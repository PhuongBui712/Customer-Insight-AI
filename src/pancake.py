import os
import re
import json
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Literal, Any, Union, List

from utils import *


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def call_pancake_api(
    url: str,
    parameters: Optional[dict] = None,
    call_type: Literal['get', 'post'] = 'get',
    add_access_token: bool = True,
    return_type: Literal['raw', 'string', 'dictionary'] = 'string',
    **kwargs: Any
) -> Union[bytes, str, dict]:
    
    # Initiate request parameters
    request_params = {}
    if add_access_token:
        request_params['access_token'] = os.getenv('PANCAKE_API')
    if parameters is not None:
        request_params.update(parameters)

    # Prepare the request arguments
    request_args = {
        'url': url,
        'params': request_params
    }

    # Add any additional keyword arguments
    request_args.update(kwargs)

    # Make the request
    if call_type == 'get':
        response = requests.get(**request_args)
    elif call_type == 'post':
        response = requests.post(**request_args)
    else:
        raise ValueError(f"Unsupported call_type: {call_type}")

    # check request status
    response.raise_for_status()

    # Process the response based on return_type
    if return_type == 'raw':
        return response.content
    elif return_type == 'string':
        return response.text
    elif return_type == 'dictionary':
        return json.loads(response.content)
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")


def get_page(return_type: Literal['id', 'standard', 'full'] = 'standard') -> Union[dict, list]:
    request_page_list_url = 'https://pages.fm/api/v1/pages'
    response = call_pancake_api(url=request_page_list_url,
                                return_type='dictionary')
    if not response['success']:
        raise Exception('Failed to get pages data from Pancake API.')

    pages = response['categorized']

    if return_type == 'full':
        return {page['id']: page for page in pages['activated']}

    elif return_type == 'standard':
        return{page['id']: {'name': page['name'], 'page_access_token': page['settings'].get('page_access_token', None)} 
               for page in pages['activated']}

    return pages['activated_page_id']


def get_page_access_token(page_info: dict) -> tuple[dict[str, str], list[str]]:
    page_ids = list(page_info.keys())

    # generate page's access token
    request_page_access_token = (
        lambda page_id: call_pancake_api(url=f'https://pages.fm/api/v1/pages/{page_id}/generate_page_access_token',
                                         call_type='post',
                                         return_type='dictionary')
    )
    page_access_tokens = {}
    not_accessible_page_ids = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_access_token = {executor.submit(request_page_access_token, id) : id for id in page_ids}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_access_token), total=len(future_to_access_token), desc='Generate page access tokens'):
            page_id = future_to_access_token[future]
            response = future.result()
            
            if not response['success']:
                print(f'UserWarning: Cannot generate access token for page "{page_info[page_id]['name']}"')
                not_accessible_page_ids.append(id)
                continue
            
            page_access_tokens[page_id] = response['page_access_token']

    return page_access_tokens, not_accessible_page_ids


def get_page_conversations(page_id: str,
                           page_access_token: str,
                           since: int,
                           until: int,
                           order_by: Literal['insert', 'update'] = 'update',
                           filter: List[str] = ['inbox', 'comment', 'rating']):
    # split the timestamp in case of the time range is longer than 1 month
    timestamp_list = split_time_stamp(since, until)
    
    # loop over each time range
    result = []
    for s, u in timestamp_list:
        # in each time range, go through all pages of response
        page_number = 1
        while True:
            response = call_pancake_api(
                url=f'https://pages.fm/api/public_api/v1/pages/{page_id}/conversations',
                parameters={'page_access_token': page_access_token,
                            'since': s,
                            'until': u,
                            'page_id': page_id,
                            'page_number': page_number,
                            'order_by': 'updated_at' if order_by == 'update' else 'inserted_at'},
                return_type='dictionary',
                add_access_token=False
            )

            if not response['success']:
                print(f"UserWarning: Failed to get conversations for page {page_id}. Respone's message: {response['message']}")
                break
            elif len(response['conversations']) == 0:
                break
            
            # update result
            result += response['conversations']

            # update `page_number`
            page_number += 1
    
    result = [c for c in result if c['type'].lower() in filter]
    return result


def get_sender(sender_dict: dict, is_sender_patterns: Optional[list] = None) -> Literal['customer', 'admin']:
    if 'admin_id' in sender_dict:
        return 'staff'

    admin_patterns = ['Trường Bào Ngư'] + (is_sender_patterns or [])
    for pattern in admin_patterns:
        if re.search(pattern, sender_dict['name']):
            return 'staff'

    return 'customer'


def filter_message(message_response: dict, attrs: Optional[list] = None) -> dict:
    if not message_response['original_message']:
        return None
    
    result = {}
    # message
    result['message'] = message_response['original_message']
    # time
    result['inserted_at'] = message_response['inserted_at']
    # from
    result['from'] = get_sender(message_response['from'])
    
    # additional attributes
    if attrs is not None:
        for k in attrs:
            result[k] = message_response.get(k, None)
         
    return result


def get_messages(page_id: str,
                 page_access_token: str,
                 conversation_id: str,
                 customer_id: str,
                 since: int,
                 until: int):
    # Get messages
    message_cnt = 0
    update_timestamp = None
    messages = []
    while update_timestamp is None or update_timestamp > since:
        # call API
        response = call_pancake_api(
            url=f'https://pages.fm/api/public_api/v1/pages/{page_id}/conversations/{conversation_id}/messages',
            parameters={
                'current_count': message_cnt,
                'page_access_token': page_access_token,
                'customer_id': customer_id,
                'conversation_id': conversation_id,
                'page_id': page_id
            },
            return_type='dictionary',
            add_access_token=False
        )

        # check response status
        if not response['success']:
            print(f"UserWarning: Failed to get conversations for page {page_id}. Respone's message: {response['message']}")

        # check messages
        if len(response['messages']) == 0:
            break

        # store messages
        messages = response['messages'] + messages

        # update `update_timstamp` and `message_cnt`
        update_timestamp = string_to_unix_second(response['messages'][0]['inserted_at'])
        message_cnt += len(response['messages'])

    # filter messages
    messages = [filter_message(m) for m in messages \
                if since <= string_to_unix_second(m['inserted_at']) <= until]
    messages = [m for m in messages if m]
    
    return messages