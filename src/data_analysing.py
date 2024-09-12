import json
import time
import re
from time import sleep
from tqdm import tqdm
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

from utils import *
from prompt import inquiry_classifying_prompt, keyword_extracting_prompt


load_dotenv()
LLM_CONFIG = load_yaml(os.path.join(PROJECT_DIRECTORY, 'config.yaml'))['llm']


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

    def _reset_counter(self):
        current_time = time.time()
        if self._last_reset_time == 0.0 or current_time - self._last_reset_time >= 60:
            self._request_counter = 0
            self._last_reset_time = current_time


    def _increment_counter(self, num_request):
        with self._state_lock:
            self._reset_counter()
            if self._request_counter + num_request > self.max_request_per_minute:
                time.sleep(max(0, self._last_reset_time + 60 - time.time()))
                self._reset_counter()
            self._request_counter += num_request


class GroqAICaller(LLMCaller):
    def __init__(self, llm_config: dict, prompt: ChatPromptTemplate):
        super().__init__(max_request_per_minute=30)

        llm = ChatGroq(**llm_config)
        self.chain = prompt | llm

    def invoke(self, input: dict):
        self._increment_counter(1)

        return self.chain.invoke(input).content


class GoogleAICaller(LLMCaller):
    def __init__(self, llm_config: dict, prompt: PromptTemplate):
        super().__init__(max_request_per_minute=15)

        llm = GoogleGenerativeAI(**llm_config)
        self.chain = prompt | llm


    def invoke(self, input: dict):
        self._increment_counter(1)

        result = self.chain.invoke(input)

        return result


def _parse_llm_output(output: str):
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


def keyword_filter_message(patterns: List[dict], messages: List[str]) -> List[dict]:
    """
    Filter messages that contain any of the given keywords.
    
    Args:
        patterns (List[dict]): A list of keywords to filter.
        messages (List[str]): A list of messages to filter.
    
    Returns:
        List[dict]: A list of messages that do not contain any of the given keywords.
    """
    synthetic_pattern = r'\b(' + '|'.join(patterns) + r')\b'
    result = [m for m in messages if not re.search(synthetic_pattern, m['message'])]

    return result


def handle_template_message(templates: Dict[str, Dict[str, str]], messages: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Handle template messages.

    This function iterates through a list of messages and checks if each message is a key in the `templates` dictionary.
    If a message is found in the `templates` dictionary, it updates the message with the corresponding template information
    and appends it to the `template_message` list. Otherwise, it appends the message to the `other_message` list.

    Args:
        templates (Dict[str, Dict[str, str]]): A dictionary of template messages, where the key is the message string
            and the value is a dictionary containing the user and purpose information.
        messages (List[dict]): A list of messages to be processed.

    Returns:
        Tuple[List[dict], List[dict]]: A tuple containing two lists:
            - `template_message`: A list of messages that were found in the `templates` dictionary.
            - `other_message`: A list of messages that were not found in the `templates` dictionary.
    """
    template_message = []
    other_message = []
    for m in messages:
        key = m['message'].lower()
        if key in templates:
            m.update(templates[key])
            template_message.append(m)
        else:
            other_message.append(m)

    return template_message, other_message


def classify_inquiry_pipeline(messages: List[dict],
                              min_score: float,
                              batch_size: int = 50,
                              provider: Literal['google', 'groq'] = 'groq') -> Tuple[List[dict], List[dict]]:
    # classify by LLM
    if provider == 'google':
        chain = GoogleAICaller(LLM_CONFIG[provider], inquiry_classifying_prompt)
    else:
        chain = GroqAICaller(LLM_CONFIG[provider], inquiry_classifying_prompt)

    mask = []
    for i in tqdm(range(0, len(messages), batch_size), desc='Detecting insightful inquiry'):
        end_idx = min(len(messages), i + batch_size)
        try:
            response = chain.invoke({'input': str([m['message'] for m in messages[i : end_idx]])})
        except Exception:
            print(f'Error while generating response for batch {i} - {end_idx}')
            
            mask += ['error' for _ in range(i, end_idx)]
            continue

        try:
            parsed_response = _parse_llm_output(response)
            mask += [list(r.items())[0][1] for r in parsed_response]
        except Exception:
            print(f'Error while parsing LLM output for batch {i} - {end_idx}')

            mask += ['error' for _ in range(i, end_idx)]
    
    # get output to return
    classified_messages = [m for m, l in zip(messages, mask) if l >= min_score]
    error_messages = [m for m, l in zip(messages, mask) if l == 'error']

    return classified_messages, error_messages


def extract_keyword_pipeline(messages: List, 
                             batch_size: int = 50, 
                             provider: Literal['google', 'groq'] = 'groq') -> Tuple[List[dict], List[dict]]:
    if provider == 'google':
        chain = GoogleAICaller(LLM_CONFIG[provider], keyword_extracting_prompt)
    else:
        chain = GroqAICaller(LLM_CONFIG[provider], keyword_extracting_prompt)
    
    keywords = []
    for i in tqdm(range(0, len(messages), batch_size), desc='Extracting keywords'):
        end_idx = min(len(messages), i + batch_size)
        try:
            response = chain.invoke({'input': str([m['message'] for m in messages[i : end_idx]])})
        
        except Exception:
            print(f'Error while generating response for batch {i} - {end_idx}')
            keywords += ['error' for _ in range(i, end_idx)]
            continue

        try:
            parsed_response = _parse_llm_output(response)
            keywords += parsed_response

        except Exception as exc:
            print(f'Error while parsing LLM output for batch {i} - {end_idx}: {exc}')
            keywords += ['error' for _ in range(i, end_idx)]

    extracted_messages = []
    error_messages = []
    for mess, kw_item in zip(messages, keywords):
        if kw_item != 'error':
            k, v = list(kw_item.items())[0]
            if all(len(x) > 0 for x in v.values()):
                extracted_messages.append(mess.copy())
                extracted_messages[-1].update(v)
        else:
            error_messages.append(mess.copy())

    return extracted_messages, error_messages


def analyse_message_pipeline(messages: List[dict],
                             filter_patterns: Optional[List[str]] = None, 
                             template_messages: Optional[dict] = None,
                             important_score: Optional[float] = 0.7,
                             batch_size: int = 50,
                             provider: Literal['google', 'groq'] = 'groq'):
    # Initialize results
    processed_messages = []
    error_messages = []

    # start processing
    if filter_patterns is not None:
        messages = keyword_filter_message(filter_patterns, messages)

    if template_messages is not None:
        template, messages = handle_template_message(template_messages, messages)
        processed_messages += template

    messages, error = classify_inquiry_pipeline(messages, important_score, batch_size, provider)
    error_messages += error

    messages, error = extract_keyword_pipeline(messages, batch_size, provider)
    error_messages += error
    processed_messages += messages

    return processed_messages, error_messages