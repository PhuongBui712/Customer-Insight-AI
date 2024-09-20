import json
import time
from threading import Lock
from typing import Optional, List, Dict

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate


class LLMCaller:
    """
    A class to manage the rate of requests to an LLM.
    
    This class implements a simple rate limiting mechanism to prevent exceeding the maximum number of requests per minute allowed by the LLM API.
    
    Attributes:
        max_request_per_minute (int): The maximum number of requests allowed per minute.
        _request_counter (int): The number of requests made in the current minute.
        _last_reset_time (float): The timestamp of the last time the request counter was reset.
        _state_lock (Lock): A lock to protect the request counter and last reset time from concurrent access.
    """
    _request_counter = 0
    _last_reset_time = 0.0
    _state_lock = Lock()

    def __init__(self, max_request_per_minute: int):
        self.max_request_per_minute = max_request_per_minute

    def _reset_counter(self) -> None:
        current_time = time.time()
        if self._last_reset_time == 0.0 or current_time - self._last_reset_time >= 60:
            self._request_counter = 0
            self._last_reset_time = current_time

    def _wait_to_next_minute(self) -> None:
        """
        Wait until the start of the next minute.
        """
        wait_time = max(0, self._last_reset_time + 60 - time.time())
        time.sleep(wait_time)
        self._reset_counter()

    def _increment_counter(self, num_request: int) -> None:
        """
        Increment the request counter and wait if the limit is reached.

        This method increments the internal request counter by the specified number of requests.
        If the total number of requests exceeds the maximum allowed per minute, it waits until the start of the next minute before incrementing the counter.
        """
        with self._state_lock:
            self._reset_counter()
            if self._request_counter + num_request > self.max_request_per_minute:
                self._wait_to_next_minute()
            self._request_counter += num_request


class GroqAICaller(LLMCaller):
    """
    A class to manage calls to the Groq AI LLM.

    This class inherits from `LLMCaller` and provides a wrapper for interacting with the Groq AI LLM.
    It handles rate limiting, error handling, and provides a simple interface for invoking the LLM.

    Attributes:
        llm_config (dict): A dictionary containing the configuration for the Groq AI LLM.
        prompt (ChatPromptTemplate): The prompt template to use for interacting with the LLM.
        chain (LLMChain): A LangChain LLMChain object that combines the prompt and the LLM.
    """
    def __init__(self, llm_config: dict, prompt: ChatPromptTemplate):
        super().__init__(max_request_per_minute=30)
        
        config = {'max_retries': 0}
        config.update(llm_config)

        llm = ChatGroq(**config)
        self.chain = prompt | llm

    def _extract_error_code(self, exception: Exception) -> Optional[int]:
        """
        Extract the error code from an exception.

        This method attempts to extract the error code from an exception raised during LLM invocation.
        If the exception does not have a status code attribute, it returns None.

        Args:
            exception (Exception): The exception raised during LLM invocation.

        Returns:
            Optional[int]: The error code extracted from the exception, or None if no error code is found.
        """
        try:
            error_code = exception.status_code
        except Exception:
            error_code = None
        
        return error_code

    def invoke(self, input: dict) -> str:
        """
        Invoke the Groq AI LLM with the given input.

        This method increments the request counter, invokes the LLM with the given input, and handles potential errors.
        If a 429 error (rate limit exceeded) is encountered, it waits until the next minute before retrying the request.

        Args:
            input (dict): The input to provide to the LLM.

        Returns:
            str: The response from the LLM.
        """
        self._increment_counter(1)
        try:
            result = self.chain.invoke(input).content
        except Exception as exc:
            if self._extract_error_code(exc) == 429:
                print('Reaching maximum resources, wait to next minutes!')
                self._wait_to_next_minute()
            
            result = self.chain.invoke(input).content
        
        return result


class GoogleAICaller(LLMCaller):
    """
    A class to call the Google Generative AI LLM.

    This class inherits from LLMCaller and provides a wrapper for invoking the Google Generative AI LLM.
    It handles rate limiting and error handling, and provides a consistent interface for invoking the LLM.
    """
    def __init__(self, llm_config: dict, prompt: PromptTemplate):
        """
        Initialize the GoogleAICaller object.

        Args:
            llm_config (dict): A dictionary containing the configuration for the Google Generative AI LLM.
            prompt (PromptTemplate): The prompt template to use for invoking the LLM.
        """
        super().__init__(max_request_per_minute=15)
        
        config = {'max_retries': 0}
        config.update(llm_config)
        
        llm = GoogleGenerativeAI(**config)
        self.chain = prompt | llm

    
    def _extract_error_code(self, exception: Exception) -> Optional[int]:
        """
        Extract the error code from an exception.

        This method attempts to extract the error code from an exception raised during LLM invocation.
        If the exception does not have a code attribute, it returns None.

        Args:
            exception (Exception): The exception raised during LLM invocation.

        Returns:
            Optional[int]: The error code extracted from the exception, or None if no error code is found.
        """
        try:
            error_code = exception.code.value
        except Exception:
            error_code = None

        return error_code
        

    def invoke(self, input: dict) -> str:
        """
        Invoke the Google Generative AI LLM with the given input.

        This method increments the request counter, invokes the LLM with the given input, and handles potential errors.
        If a 429 error (rate limit exceeded) is encountered, it waits until the next minute before retrying the request.

        Args:
            input (dict): The input to provide to the LLM.

        Returns:
            str: The response from the LLM.
        """
        self._increment_counter(1)
        try:
            result = self.chain.invoke(input)
        except Exception as exc:
            if self._extract_error_code(exc) == 429:
                print('Reaching maximum resources, wait to next minutes!')
                self._wait_to_next_minute()
                result = self.chain.invoke(input)

            raise exc
        
        return result


def parse_llm_output(output: str) -> List[Dict]:
    """
    Parse the output of the LLM.
    
    The output of the LLM is expected to be in either '```python' or '```json' format.
    This function will parse the output and return the result as a dictionary.
    
    Args:
        output (str): The output of the LLM.
    
    Returns:
        dict: The parsed output of the LLM.
    
    Raises:
        Exception: If the output is not in the expected format.
    """
    start = output.index('[')
    end =  len(output) - output[::-1].index(']')

    error_comma = end - 2 if output[end - 1] == ',' else end - 3
    if output[error_comma] == ',':
        output = output[:error_comma] + output[error_comma + 1:]

    try:
        res = json.loads(output[start:end])
    except Exception:
        try:
            res = json.loads(output[start:end].lower())
        except Exception:
            raise Exception(f"Could not parse output. Received: \n{output}")
    return res