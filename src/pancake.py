import os
import json
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Literal, Any, Union, List

from utils import *


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
    

def get_page(return_type: Literal['id', 'full'] = 'full') -> dict:
    request_page_list_url = 'https://pages.fm/api/v1/pages'
    response = call_pancake_api(url=request_page_list_url,
                                return_type='dictionary')
    pages = response['categorized']

    if return_type == 'full':
        return {page['id']: page for page in pages['activated']}
    return pages['activated_page_id']


def get_page_access_token(page_info: dict, config: Optional[dict] = None) -> tuple[dict[str, str], list[str]]:
    if config is not None and config['excluded_page']:
        page_ids = list(page_info.keys())
        excluded_ids = [id for id, info in page_info.items() if info['name'] in config['excluded_page']]
        page_ids = list(set(page_ids) - set(excluded_ids))

    # generate page's access token
    request_page_access_token = (
        lambda page_id: call_pancake_api(url=f'https://pages.fm/api/v1/pages/{page_id}/generate_page_access_token',
                                         call_type='post',
                                         return_type='dictionary')
    )
    page_access_tokens = {}
    not_accessible_page_ids = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_access_token = {executor.submit(request_page_access_token, id) : id for id in page_ids}
        
        for future in concurrent.futures.as_completed(future_to_access_token):
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
                           page_number: int = 1,
                           order_by: Literal['insert', 'update'] = 'update',
                           filter: List[str] = ['inbox', 'comment', 'rating']):
    response = call_pancake_api(
        url=f'https://pages.fm/api/public_api/v1/pages/{page_id}/conversations',
        parameters={'page_access_token': page_access_token,
                    'since': since,
                    'until': until,
                    'page_id': page_id,
                    'page_number': page_number},
        return_type='dictionary',
        add_access_token=False
    )
    
    result = None
    if not response['success']:
        print('UserWarning: Failed to get conversations for page {}.'.format(page_id))
    else:
        result = [c for c in response['conversations'] if c['type'].lower() in filter]
    
    return result


