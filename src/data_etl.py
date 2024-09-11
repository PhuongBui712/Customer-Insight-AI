import os
import re
import time
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from typing import List, Dict

from pancake import *
from data_analysing import *
from data_presentation import *
from utils import *


# --------------------- Crawling Messages sub-processes ---------------------
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


# --------------------- Get Messages main process ---------------------
def pancake_etl(schema_path: str, filter_patterns: Optional[List] = None):
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
def post_process_messages(messages: Dict[str, List[dict]]):
    result = {}
    for con_id, mess_list in messages.items():
        result[con_id] = {}
        result[con_id]['conversation'] = [f'{m['from']}: {m['message']}' for m in mess_list]
        result[con_id]['update'] = string_to_unix_second(mess_list[-1]['updated_at'])

    return result


def setup_data_directory(root_dir: str, sub_dir_list: List[str]) -> None:
    for dir in sub_dir_list:
        os.makedirs(os.path.join(root_dir, dir), exist_ok=True)


# --------------------- ETL tasks ---------------------
# task: remove old data at beginning of a day
def remove_old_data(table_path: str, num_days_before: int):
    # check data has been collected
    if not os.path.exists(table_path):
        return
    
    # just remove data once per day
    now = get_current_time_utc_plus_7()
    if now.hour != 0 and now.minute > 30:
        return
    
    # remove
    message_df = load_worksheet(0)
    date_bound = get_day_before(num_days_before, return_type='date')
    message_df = message_df.loc[message_df['inserted_at'] > date_bound]

    # udpate worksheet
    update_worksheet(message_df, sheet_idx=0)

    # update local sheet
    message_df.to_csv(table_path, index=False)


# task 2: update new data (pages, conversations, messages)
def update_new_data(config: dict):
    schema = os.path.join(PROJECT_DIRECTORY, config['schema'])
    filter_patterns = config['filter-page-keywords']
    messages = pancake_etl(schema, filter_patterns)

    # store messages lists
    total_message_path = os.path.join(PROJECT_DIRECTORY, config['total-message'])
    total_messages = messages.copy()
    if os.path.exists(total_message_path):
        total_messages = load_json(total_message_path) + messages

    save_json(total_message_path, total_messages)

    return messages, total_messages


# task 3: load data to be analysed
def load_analyse_data(config: dict, messages: List[str], total_messages: List[str]):
    # load messages needed to analyse
    message_table_path = os.path.join(PROJECT_DIRECTORY, config['message-table'])
    error_path = os.path.join(PROJECT_DIRECTORY, config['error-message'])

    # load total messages if not analysing yet
    if not os.path.exists(message_table_path):
        messages = [m for m in total_messages if m['from'] == 'customer']

    # load error messages if exist
    if os.path.exists(error_path):
        messages += load_json(error_path)

    return messages


# task 6: update tables
def update_table(config: dict, processed_messages: List[dict], error_messages: Optional[List[dict]]):
    # update message df
    message_sheet_idx = config['google-sheet']['message-sheet-iddx']
    last_message_df = load_worksheet(message_sheet_idx)
    new_message_df = create_dataframe(processed_messages)
    updated_message_df = pd.concat([last_message_df, new_message_df], axis=0)

    update_worksheet(updated_message_df, message_sheet_idx, mode='replace')
    updated_message_df.to_csv(os.path.join(PROJECT_DIRECTORY, config['message-table']), index=False)

    # update 2 stats sheet
    user_sheet_idx = config['google-sheet']['user-sheet-idx']
    purpose_sheet_idx = config['google-sheet']['purpose-sheet-idx']
    user_df, purpose_df = analyse_data(updated_message_df)

    update_worksheet(user_df, user_sheet_idx, mode='replace')
    user_df.to_csv(os.path.join(PROJECT_DIRECTORY, config['user-table']))
    update_worksheet(purpose_df, purpose_sheet_idx, mode='replace')
    purpose_df.to_csv(os.path.join(PROJECT_DIRECTORY, config['purpose-table']))

    # store failed messages
    if error_messages:
        save_json(os.path.join(PROJECT_DIRECTORY, config['error-message']))
    

# complete ETL
def analyse_customer_message_pipeline():
    # 1. setup directory for etl pipeline
    config = load_yaml(os.path.join(
        PROJECT_DIRECTORY, 'config.yaml'
    ))

    setup_data_directory(
        PROJECT_DIRECTORY,
        sub_dir_list=[config['data-directory']]
    )

    # 2. Check new day & delete old data
    remove_old_data(table_path=os.path.join(PROJECT_DIRECTORY, config['message-table']), num_days_before=90)
    
    # 3. get new messages
    messages, total_messages = update_new_data(config)

    # 4. load analyse messages
    messages = load_analyse_data(config, messages, total_messages)

    # 5. analysing
    processed_messages, error_messages = analyse_message_pipeline(
        messages,
        filter_patterns=config['filter-message-keywords'],
        template_messages=config['template-message'],
        important_score=config['important-score']
    )

    # 6. update tables
    update_table(config, processed_messages, error_messages)


if __name__ == '__main__':
    analyse_customer_message_pipeline()