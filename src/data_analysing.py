import json
import time
import re
from time import sleep
from tqdm import tqdm
from threading import Lock
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from dotenv import load_dotenv

from utils import *
from prompt import (
    classifying_inquiry_prompt,
    reclassifying_inquiry_prompt,
    extracting_user_purpose_prompt
)
from llm import *


load_dotenv()
LLM_CONFIG = load_yaml(os.path.join(PROJECT_DIRECTORY, 'config.yaml'))['llm']


def keyword_filter(patterns: List[str], messages: List[dict], get_keyword: Optional[bool] = True) -> List[dict]:
    """
    Filter messages based on the presence or absence of specified keywords.

    This function iterates through a list of messages and checks if each message contains any of the given keywords.
    It returns a list of messages that either contain or do not contain the specified keywords, depending on the `get_keyword` flag.

    Args:
        patterns (List[str]): A list of keywords to filter by.
        messages (List[dict]): A list of messages to filter.
        get_keyword (Optional[bool], optional): If True, returns messages containing the keywords. 
            If False, returns messages not containing the keywords. Defaults to True.

    Returns:
        List[dict]: A list of messages that meet the filtering criteria.
    """
    synthetic_pattern = r'\b(' + '|'.join(patterns) + r')\b'
    result = [m for m in messages 
              if bool(re.search(synthetic_pattern, m['message'].lower())) == get_keyword]

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


def classify_inquiry_pipeline(
    messages: List[dict],
    min_score: float,
    batch_size: int = 50,
    provider: Literal["google", "groq"] = "groq",
) -> Tuple[List[dict], List[dict]]:
    """
    Classifies messages as insightful inquiries using an LLM.

    This function iterates through a list of messages and uses an LLM (either Google Generative AI or Groq AI) to determine if each message is an insightful inquiry.
    It uses a prompt template to guide the LLM's classification and returns two lists: one containing messages classified as insightful inquiries and another containing messages that encountered errors during processing.

    Args:
        messages (List[dict]): A list of messages to be classified.
        min_score (float): The minimum score required for a message to be considered an insightful inquiry.
        batch_size (int, optional): The number of messages to process in each batch. Defaults to 50.
        provider (Literal["google", "groq"], optional): The LLM provider to use. Defaults to "groq".

    Returns:
        Tuple[List[dict], List[dict]]: A tuple containing two lists:
            - `classified_messages`: A list of messages classified as insightful inquiries.
            - `error_messages`: A list of messages that encountered errors during processing.
    """
    
    if provider == "google":
        chain = GoogleAICaller(LLM_CONFIG[provider], classifying_inquiry_prompt)
    else:
        chain = GroqAICaller(LLM_CONFIG[provider], classifying_inquiry_prompt)

    @lru_cache(maxsize=None)
    def cached_invoke(input_str):
        return chain.invoke({"input": input_str})

    mask = [None for _ in range(len(messages))]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future = {
            executor.submit(
                lambda: cached_invoke(str([m["message"] for m in messages[i : min(i + batch_size, len(messages))]]))
            ): i
            for i in range(0, len(messages), batch_size)
        }
        for f in tqdm(as_completed(future), total=len(future), desc="Detecting insightful inquiry"):
            i = future[f]
            end_idx = min(i + batch_size, len(messages))
            try:
                response = f.result()
            except Exception as exc:
                print(f"Error while generating response for batch {i} - {end_idx - 1}")
                for idx in range(i, end_idx):
                    mask[idx] = 'error'
                continue

            try:
                parsed_response = parse_llm_output(response)
                for j, idx in enumerate(range(i, end_idx)):
                    mask[idx] = parsed_response[j][messages[idx]['message']]
            except Exception:
                print(f"Error while parsing LLM output for batch {i} - {end_idx - 1}")
                for idx in range(i, end_idx):
                    mask[idx] = 'error'

    classified_messages = []
    error_messages = []
    for message, score in zip(messages, mask):
        if score == 'error':
            error_messages.append(message)
        elif score >= min_score:
            classified_messages.append(message)

    return classified_messages, error_messages


def reclassify_inquiry_pipeline(
    messages: List[dict],
    batch_size: int = 50,
    provider: Literal["google", "groq"] = "groq",
) -> Tuple[List[dict], List[dict]]:
    """
    Reclassifies a list of messages using an LLM.

    This function takes a list of messages, each represented as a dictionary, and uses an LLM to reclassify them.
    It batches the messages and sends them to the LLM in parallel using a thread pool.
    The LLM's response is then parsed and used to update a mask indicating whether each message is classified as insightful or not.
    Finally, the function returns two lists: one containing the classified messages and another containing messages that encountered errors during processing.

    Args:
        messages (List[dict]): A list of messages, each represented as a dictionary.
        batch_size (int, optional): The number of messages to send to the LLM in each batch. Defaults to 50.
        provider (Literal["google", "groq"], optional): The LLM provider to use. Defaults to "groq".

    Returns:
        Tuple[List[dict], List[dict]]: A tuple containing two lists: the first list contains the classified messages, and the second list contains the messages that encountered errors during processing.
    """
    
    # classify by LLM
    if provider == "google":
        chain = GoogleAICaller(LLM_CONFIG[provider], reclassifying_inquiry_prompt)
    else:
        chain = GroqAICaller(LLM_CONFIG[provider], reclassifying_inquiry_prompt)

    @lru_cache(maxsize=None)
    def cached_invoke(input_str):
        return chain.invoke({"input": input_str})

    mask = [None for _ in range(len(messages))]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future = {
            executor.submit(
                lambda: cached_invoke(str([m["message"] for m in messages[i : min(i + batch_size, len(messages))]]))
            ): i
            for i in range(0, len(messages), batch_size)
        }
        for f in tqdm(as_completed(future), total=len(future), desc="Detecting insightful inquiry"):
            i = future[f]
            end_idx = min(i + batch_size, len(messages))
            try:
                response = f.result()
            except Exception as exc:
                print(f"Error while generating response for batch {i} - {end_idx - 1}")

                # update mask
                for i in range(i, end_idx):
                    mask[i] = 'error'
                continue

            try:
                parsed_response = parse_llm_output(response)
                # update mask
                for i, idx in enumerate(range(i, end_idx)):
                    mask[idx] = parsed_response[i][messages[idx]['message']]

            except Exception:
                print(f"Error while parsing LLM output for batch {i} - {end_idx - 1}")
                
                # update mask
                for i in range(i, end_idx):
                    mask[i] = 'error'
                continue

    # get output to return
    classified_messages = [
        m for m, l in zip(messages, mask) 
        if l == True
    ]
    error_messages = [m for m, l in zip(messages, mask) if l == "error"]

    return classified_messages, error_messages


def extract_user_purpose_pipeline(messages: List, 
                             batch_size: int = 50, 
                             provider: Literal['google', 'groq'] = 'groq') -> Tuple[List[dict], List[dict]]:
    """
    Extracts the user's purpose from a list of messages using a specified LLM provider.

    Args:
        messages (List): A list of dictionaries, each representing a message with a "message" key.
        batch_size (int, optional): The number of messages to process in each batch. Defaults to 50.
        provider (Literal['google', 'groq'], optional): The LLM provider to use. Defaults to 'groq'.

    Returns:
        Tuple[List[dict], List[dict]]: A tuple containing two lists:
            - The first list contains the extracted messages with the user's purpose added.
            - The second list contains the messages that failed to be processed.
    """
    if provider == 'google':
        chain = GoogleAICaller(LLM_CONFIG[provider], extracting_user_purpose_prompt)
    else:
        chain = GroqAICaller(LLM_CONFIG[provider], extracting_user_purpose_prompt)
    
    @lru_cache(maxsize=None)
    def cached_invoke(input_str):
        return chain.invoke({"input": input_str})

    user_and_purpose = [None for _ in range(len(messages))]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future = {
            executor.submit(
                lambda : cached_invoke(str([m["message"] for m in messages[i : min(i + batch_size, len(messages))]]))
            ): i
            for i in range(0, len(messages), batch_size)
        }
        for f in tqdm(as_completed(future), total=len(future), desc='Extracting keywords'):
            i = future[f]
            end_idx = min(len(messages), i + batch_size)
            try:
                response = f.result()
            
            except Exception:
                print(f'Error while generating response for batch {i} - {end_idx - 1}')
                # update `user_and_purpose`
                for idx in range(i, end_idx):
                    user_and_purpose[idx] = 'error'
                continue

            try:
                parsed_response = parse_llm_output(response)
                # update `user_and_purpose`
                for j, idx in enumerate(range(i, end_idx)):
                    user_and_purpose[idx] = parsed_response[j][messages[idx]['message']].copy()

            except Exception as exc:
                print(f'Error while parsing LLM output for batch {i} - {end_idx - 1}')
                # update `user_and_purpose`
                for idx in range(i, end_idx):
                    user_and_purpose[idx] = 'error'

    extracted_messages = []
    error_messages = []
    for mess, u_and_p in zip(messages, user_and_purpose):
        if u_and_p == 'error':
            error_messages.append(mess)
        else:
            extracted_message = mess.copy()
            extracted_message.update(u_and_p)
            extracted_messages.append(extracted_message)
        
    return extracted_messages, error_messages


def analyse_message_pipeline(messages: List[dict],
                             remove_keywords: List[str] = None,
                             filter_keywords: List[str] = None,
                             template: Optional[dict] = None,
                             important_score: Optional[float] = 0.7,
                             batch_size: int = 50,
                             provider: Literal['google', 'groq'] = 'groq') -> Tuple[Optional[List[dict]], List[dict], List[dict]]:
    """
    This function processes a list of messages through a pipeline of analysis steps.

    Args:
        messages: A list of dictionaries, each representing a message.
        remove_keywords: A list of keywords to remove from the messages.
        filter_keywords: A list of keywords to filter the messages by.
        template: A dictionary representing a template message.
        important_score: The threshold for classifying a message as important.
        batch_size: The number of messages to process in each batch.
        provider: The LLM provider to use.

    Returns:
        A tuple containing:
            - template_messages: A list of dictionaries representing the template messages, if any.
            - processed_messages: A list of dictionaries representing the processed messages.
            - error_messages: A list of dictionaries representing the messages that failed to process.
    """
    # Initialize results
    processed_messages = []
    error_messages = []

    # start processing
    if remove_keywords:
        messages = keyword_filter(remove_keywords, messages, get_keyword=False)

    if filter_keywords:
        messages = keyword_filter(filter_keywords, messages, get_keyword=True)

    template_messages = None
    if template is not None:
        template_messages, messages = handle_template_message(template, messages)

    messages, error = classify_inquiry_pipeline(messages, important_score, batch_size, provider)
    error_messages += error

    messages, error = extract_user_purpose_pipeline(messages, batch_size, provider)
    error_messages += error
    processed_messages += messages

    return template_messages, processed_messages, error_messages