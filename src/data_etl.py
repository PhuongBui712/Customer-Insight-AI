import os
import re
import time
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from typing import List, Dict

from pancake import *
from llm import *
from google_sheet import *
from utils import *


# --------------------- ETL Subprocess---------------------
def filter_pages(name: str, patterns: List):
    for p in patterns:
        if re.search(p, name):
            return True
        
    return False


def update_page(schema_path: str, page_filter_patterns: Optional[List[str]] = None):
    # check exitent schema
    page_schema = {}
    if os.path.exists(schema_path):
        page_schema = load_json(schema_path)

    # call page list
    pages = get_page()
    # filter proper pages
    if page_filter_patterns:
        pages = {k: v for k, v in pages.items() if filter_pages(v['name'], page_filter_patterns)}
    
    # update `page_access_token`
    # update available `page_access_token` in case of it was renewed
    no_access_token_pages = []
    for k, v in pages.items():
        if v['page_access_token'] is None or k not in page_schema:
            no_access_token_pages.append(k)
        elif page_schema[k]['page_access_token'] != v['page_access_token']:
            page_schema[k]['page_access_token'] != v['page_access_token']

    # generate `page_access_token`
    no_access_token_pages = {k: v for k, v in pages.items() if v['page_access_token'] is None}
    if no_access_token_pages:
        page_access_tokens, _ = get_page_access_token(no_access_token_pages)
        for k, v in page_access_tokens.items():
            pages[k]['page_access_token'] = v

    # update latest data schema
    iter_keys = list(set(pages.keys()) - set(page_schema.keys()))
    for k in iter_keys:
        if not pages[k]['page_access_token']:
            continue

        # others case
        page_schema[k] = {}
        page_schema[k]['page_access_token'] = pages[k]['page_access_token']
        # set up for conversations, messages scheme
        page_schema[k]['last_check'] = get_day_before(30)
        page_schema[k]['conversations'] = {}
    
    return page_schema


def update_conversation(page_schema: dict, new_check: int):
    # call newest conversations of all pages
    conversations = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_conversations = {
            executor.submit(
                get_page_conversations,
                k, # page_id
                v['page_access_token'], # page_access_token
                v['last_check'], # since
                new_check, # until,
                'update', # order_by
                ['inbox'] # filter
            ): k
            for k, v in page_schema.items()
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_conversations), total=len(future_to_conversations), desc='Request conversations'):
            page_id = future_to_conversations[future]
            data = []
            try:
                data = future.result()
            except Exception as exc:
                print(f'Error occurred while fetching conversations for page {page_id}: {exc}')

            conversations[page_id] = data

    # check which conversation has new messages
    update_conversations = []
    for page_id, conversation_list in conversations.items():
        page_access_token = page_schema[page_id]['page_access_token']
        last_conversations = page_schema[page_id]['conversations']
        
        for con in conversation_list:
            con_id = con['id']
            if con_id not in last_conversations:
                # update `last_crawled_conversations`
                last_conversations[con_id] = {
                    'customer_id': con['customer_id'],
                    'last_updated': con['updated_at'],
                }

                # append to list to crawl data
                update_conversations.append(
                    # append page_id, page_access_token, conversation_id, customer_id, last_update
                    (page_id, page_access_token, con_id, con['customer_id'], get_day_before(30))
                )

            elif con['updated_at'] != last_conversations[con_id]['last_updated']:
                # update last crawl time
                last_conversations[con_id]['last_updated'] = con['updated_at']
                
                # add to crawling list
                last_update_timestamp = string_to_unix_second(con['updated_at'])
                update_conversations.append(
                    (page_id, page_access_token, con_id, con['customer_id'], last_update_timestamp)
                )

    return update_conversations


def update_messages(conversations: dict, new_check: int):
    messages = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_messages = {
            executor.submit(
                get_messages,
                m[0],
                m[1],
                m[2],
                m[3],
                m[4],
                new_check
            ): m[2] for m in conversations
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_messages), total=len(future_to_messages), desc='Request messages'):
            con_id = future_to_messages[future]
            try:
                mess = future.result()
                if mess:
                    messages += mess
            except Exception as exc:
                print(f'Error occurred while fetching messages for conversation {con_id}: {exc}')

    return messages



# --------------------- ETL Main process ---------------------
def data_etl(schema_path: str, filter_patterns: Optional[List] = None):
    # 1. Update pages
    page_schema = update_page(schema_path, filter_patterns)

    # 2. Update conversations
    new_check = int(time.time())
    conversations = update_conversation(page_schema, new_check)

    # 3. Completing update schema, save it.
    save_json(schema_path, page_schema)
    
    # 3. Get new messages
    messages = update_messages(conversations, new_check)

    return messages


# --------------------- ETL Utilities ---------------------
def create_dataframe(messages: List[dict]) -> DataFrame:
    df = pd.DataFrame(data=messages)

    # convert string to datetime
    df['inserted_at'] = df['inserted_at'].str.replace('T', ' ')
    df['inserted_at'] = df['inserted_at'].str.replace(r'\.\d{6}', '', regex=True)
    df['inserted_at'] = pd.to_datetime(df['inserted_at'], format='%Y-%m-%d %H:%M:%S')
    
    # transform timezone
    df['inserted_at'] = df['inserted_at'].dt.tz_localize('UTC')
    df['inserted_at'] = df['inserted_at'].dt.tz_convert('Asia/Bangkok')

    return df


def setup_data_directory(root_dir: str, sub_dir_list: List[str]) -> None:
    for dir in sub_dir_list:
        os.makedirs(os.path.join(root_dir, dir), exist_ok=True)


if __name__ == '__main__':
    # 1. setup directory for etl pipeline
    config = load_yaml(os.path.join(
        PROJECT_DIRECTORY, 'config.yaml'
    ))

    setup_data_directory(PROJECT_DIRECTORY,
                         sub_dir_list=[config['data-directory']])

    # 2. start getting data
    schema = os.path.join(PROJECT_DIRECTORY, config['schema'])
    filter_patterns = config['filter-page-keywords']
    messages = data_etl(schema, filter_patterns)

    # store messages lists
    total_message_path = os.path.join(PROJECT_DIRECTORY, config['total-messages'])
    if os.path.exists(total_message_path):
        total_messages = load_json(total_message_path) + messages

    save_json(total_message_path, total_messages)

    # 3. analyse messages
    # load messages needed to analyse
    message_table_path = os.path.join(PROJECT_DIRECTORY, config['message-table'])
    if not os.path.exists(message_table_path):
        messages = total_messages

    # analysing
    customer_messages = [m for m in messages if m['from'] == 'customer']
    message_list = [m['message'] for m in customer_messages]
    important_mask = classify_inquiry_pipeline(message_list)

    # handle failed cases
    error_messages = [m for i, m in zip(important_mask, customer_messages) if i == 'error']
    save_json(os.path.join(PROJECT_DIRECTORY, config['error-messages']))

    # handle success cases
    important_messages = [m for i, m in zip(important_mask, customer_messages) if i == True]
    important_message_df = create_dataframe(important_messages)
    inquiry_path = os.path.join(PROJECT_DIRECTORY, config['insight-inquiry-table'])
    important_message_df.to_csv(inquiry_path, index=False)

    # 3.1 Extracting keywords
    # # adding retry messages
    # retry_keyword_path = os.path.join(PROJECT_DIRECTORY, config['keyword-retry'])
    # if os.path.exists(retry_keyword_path):
    #     customer_messages += load_json(retry_keyword_path)

    # # extracting keywords
    # message_list = [m['message'] for m in customer_messages]
    # keywords_list = keyword_extract_pipeline(message_list)
    
    # # retry_keywords = []
    # for keywords, message in zip(keywords_list, customer_messages):
    #         message['keywords'] = keywords
    # customer_messages = [m for m in customer_messages if m.get('keywords', None)]

    # # save retry keywords
    # # save_json(retry_keyword_path, retry_keywords)
    
    # # convert to table
    # keywords_df = pd.DataFrame(data=customer_messages)
    
    # # update table
    # keywords_path = os.path.join(PROJECT_DIRECTORY, config['keyword-table'])
    # if os.path.exists(keywords_path):
    #     past_keyword_df = pd.read_csv(keywords_path)
    #     keywords_df = pd.concat([keywords_df, past_keyword_df], axis=0)
    
    # keywords_df.to_csv(keywords_path, index=False)


