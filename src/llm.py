import json
import time
import re
from time import sleep
from tqdm import tqdm
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

from utils import *
from prompt import inquiry_classifying_prompt


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
    


# def keyword_extract_pipeline(messages: List, batch_size: int = 50) -> List[List[str]]:
#     chain = LLMCaller(LLM_CONFIG, extract_keyword_prompt)

#     for i in tqdm(range(0, len(messages), batch_size), desc='Extracting keywords'):
#         response = chain.invoke({'input': str(messages[i : min(len(messages), i + batch_size)])})
#         result = []
#         try:
#             result += _parse_llm_output(response)
#         except Exception as exc:
#             print(f'UserWarning: Failed to parse the output of batch {i}-{i + batch_size}')
            
#             result += ['retry' for _ in range(50)]

#     return result


def prefilter_messages(patterns: List[str], messages: List[str]):
    synthetic_pattern = r'\b(' + '|'.join(patterns) + r')\b'
    result = []
    for m in messages:
        if re.search(synthetic_pattern, m.lower()):
            result.append(False)
        else:
            result.append(True)

    return result


def classify_inquiry_pipeline(messages: List, batch_size: int = 50, provider: Literal['google', 'groq'] = 'groq') -> List[bool]:
    # pre-filter common messages by keywords
    common_keywords = ['giao', 'ship', 'địa chỉ', 'dia chi', 'giá', 'stk', 'số tài khoản', 
                       'so tai khoan', 'thanh toán', 'chuyển khoản', 'tiền mặt', 'menu', 'alo', 'hi',
                       'ok', 'shjp', 'quét']
    result = prefilter_messages(common_keywords, messages)

    redetect_message = [m for m, r in zip(messages, result) if r]
    redetect_index = [i for i in range(len(result)) if result[i]]
    
    # classify by LLM
    if provider == 'google':
        chain = GoogleAICaller(LLM_CONFIG[provider], inquiry_classifying_prompt)
    else:
        chain = GroqAICaller(LLM_CONFIG[provider], inquiry_classifying_prompt)

    mask = []
    for i in tqdm(range(0, len(redetect_message), batch_size), desc='Detecting insightful inquiry'):
        end_idx = min(len(redetect_message), i + batch_size)
        try:
            response = chain.invoke({'input': str(redetect_message[i : end_idx])})
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

    # synthesize mask
    for i, j in enumerate(redetect_index):
        result[j] = mask[i]

    return result