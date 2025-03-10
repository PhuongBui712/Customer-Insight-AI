import os
import sys

sys.path.append(os.path.dirname(__file__))

import re
import time
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from data_analysing import *
from data_presentation import *
from pancake import *
from utils import *


# --------------------- Crawling Messages sub-processes ---------------------
def filter_pages(name: str, patterns: List):
    for p in patterns:
        if re.search(p, name):
            return True

    return False


def update_page(
    schema_path: str,
    default_last_check: int = 30,
    page_filter_patterns: Optional[List[str]] = None,
):
    # check exitent schema
    page_schema = {}
    if os.path.exists(schema_path):
        page_schema = load_json(schema_path)

    # call page list
    pages = get_page()
    # filter proper pages
    if page_filter_patterns:
        pages = {
            k: v
            for k, v in pages.items()
            if filter_pages(v["name"], page_filter_patterns)
        }

    # update `page_access_token`
    # update available `page_access_token` in case of it was renewed
    no_access_token_pages = []
    for k, v in pages.items():
        if v["page_access_token"] is None or k not in page_schema:
            no_access_token_pages.append(k)
        elif page_schema[k]["page_access_token"] != v["page_access_token"]:
            page_schema[k]["page_access_token"] = v["page_access_token"]

    # generate `page_access_token`
    no_access_token_pages = {
        k: v for k, v in pages.items() if v["page_access_token"] is None
    }
    if no_access_token_pages:
        page_access_tokens, _ = get_page_access_token(no_access_token_pages)
        for k, v in page_access_tokens.items():
            pages[k]["page_access_token"] = v

    # update latest data schema
    iter_keys = list(set(pages.keys()) - set(page_schema.keys()))
    for k in iter_keys:
        if not pages[k]["page_access_token"]:
            continue

        # others case
        page_schema[k] = {}
        page_schema[k]["name"] = pages[k]["name"]
        page_schema[k]["page_access_token"] = pages[k]["page_access_token"]
        # set up for conversations, messages scheme
        page_schema[k]["last_check"] = get_day_before(default_last_check)
        page_schema[k]["conversations"] = {}

    return page_schema


def update_conversation(page_schema: dict, new_check: int):
    # call newest conversations of all pages
    conversations = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_conversations = {
            executor.submit(
                get_page_conversations,
                k,  # page_id
                v["page_access_token"],  # page_access_token
                v["last_check"],  # since
                new_check,  # until,
                "update",  # order_by
                ["inbox"],  # filter
            ): k
            for k, v in page_schema.items()
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_conversations),
            total=len(future_to_conversations),
            desc="Request conversations",
        ):
            page_id = future_to_conversations[future]
            data = []
            try:
                data = future.result()
            except Exception as exc:
                print(
                    f"Error occurred while fetching conversations for page {page_id}: {exc}"
                )

            conversations[page_id] = data

    # check which conversation has new messages
    update_conversations = []
    for page_id, conversation_list in conversations.items():
        page_access_token = page_schema[page_id]["page_access_token"]
        last_conversations = page_schema[page_id]["conversations"]
        page_last_check = page_schema[page_id]["last_check"]

        # update last check timestamp
        page_schema[page_id]["last_check"] = new_check

        # get the conversations that need to be scraped
        for con in conversation_list:
            con_id = con["id"]
            # if con_id not in last_conversations:
            #     # update `last_crawled_conversations`
            #     last_conversations[con_id] = {
            #         'customer_id': con['customer_id'],
            #         'last_updated': con['updated_at'],
            #     }

            #     # append to list to crawl data
            #     update_conversations.append(
            #         # append page_id, page_access_token, conversation_id, customer_id, last_update
            #         (page_id, page_access_token, con_id, con['customer_id'], page_last_check)
            #     )

            # elif con['updated_at'] != last_conversations[con_id]['last_updated']:
            #     # update last crawl time
            #     last_conversations[con_id]['last_updated'] = con['updated_at']

            #     # add to crawling list
            #     last_update_timestamp = string_to_unix_second(con['updated_at'])
            #     update_conversations.append(
            #         (page_id, page_access_token, con_id, con['customer_id'], last_update_timestamp)
            #     )

            # add to crawl list
            last_update_timestamp = (
                page_last_check
                if con_id not in last_conversations
                else string_to_unix_second(last_conversations[con_id]["last_updated"])
            )
            update_conversations.append(
                # append page_id, page_access_token, conversation_id, customer_id, last_update
                (
                    page_id,
                    page_access_token,
                    con_id,
                    con["customer_id"],
                    last_update_timestamp,
                )
            )

            # update `last_conversations`
            last_conversations[con_id] = {
                "customer_id": con["customer_id"],
                "last_updated": con["updated_at"],
            }

    return update_conversations


def update_messages(conversations: dict, new_check: int):
    messages = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_messages = {
            executor.submit(get_messages, m[0], m[1], m[2], m[3], m[4], new_check): m[2]
            for m in conversations
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_messages),
            total=len(future_to_messages),
            desc="Request messages",
        ):
            con_id = future_to_messages[future]
            try:
                mess = future.result()
                if mess:
                    messages += mess
            except Exception as exc:
                print(
                    f"Error occurred while fetching messages for conversation {con_id}: {exc}"
                )

    return messages


# --------------------- Get Messages main process ---------------------
def pancake_etl(
    schema_path: str,
    default_last_check: int = 30,
    filter_patterns: Optional[List] = None,
):
    # 1. Update pages
    page_schema = update_page(schema_path, default_last_check, filter_patterns)

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
        result[con_id]["conversation"] = [
            f"{m['from']}: {m['message']}" for m in mess_list
        ]
        result[con_id]["update"] = string_to_unix_second(mess_list[-1]["updated_at"])

    return result


def setup_data_directory(root_dir: str, sub_dir_list: List[str]) -> None:
    for dir in sub_dir_list:
        os.makedirs(os.path.join(root_dir, dir), exist_ok=True)


def load_table(
    path: str,
    list_cols: Optional[List[str]] = None,
    datetime_cols: Optional[List[str]] = None,
    datetime_format: str = "%Y-%m-%d %H:%M:%S",
) -> DataFrame:
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame()

    # Process list columns
    if list_cols:
        for col in list_cols:
            # Apply the following steps to clean and convert the string into a list
            def process_list(item):
                if pd.isna(item):  # Handle missing values
                    return []

                # Remove square brackets if present, and then split by comma
                item = item.strip("[]")  # Remove surrounding brackets
                return [
                    x.strip().strip("'") for x in item.split(",")
                ]  # Split by commas and strip extra spaces

            df[col] = df[col].apply(process_list)

    # Convert datetime columns
    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(
                df[col], format=datetime_format, errors="coerce"
            )  # Handle format and errors

    return df


# --------------------- ETL tasks ---------------------
# task 2: remove old data at beginning of a day
def remove_old_data(config: dict):
    # check data has been collected
    if not os.path.exists(get_project_path(config["queue-message"])):
        return

    # just remove data once per day
    now = get_current_time_utc_plus_7()
    if now.hour != 0 and now.minute > 30:
        return

    date_bound = get_day_before(config["oldest-date"], return_type="date")

    # remove data on queue
    if os.path.exists(get_project_path(config["queue-message"])):
        queue_messsages = load_json(get_project_path(config["queue-message"]))
        print(f"Before removing old data, queue size {len(queue_messsages)}")

        queue_messsages = [
            m
            for m in queue_messsages
            if string_to_unix_second(m["inserted_at"]) > int(date_bound.timestamp())
        ]
        print(f"After removing old data, queue size {len(queue_messsages)}")

        save_json(get_project_path(config["queue-message"]), queue_messsages)

    # remove data on sheet
    for path, sheet_name in zip(
        (
            get_project_path(config["message-table"]),
            get_project_path(config["question-table"]),
        ),
        (config["message-sheet"], config["question-sheet"]),
    ):
        # remove
        load_table_args = {"path": path, "datetime_cols": ["inserted_at"]}
        name = "question"
        if sheet_name == config["message-sheet"]:
            load_table_args.update({"list_cols": ["user", "purpose"]})
            name = "message"
        message_df = load_table(**load_table_args)

        if not message_df.empty:
            print(f"Before removing old data, {name} table: {message_df.shape}")
            message_df = message_df.loc[message_df["inserted_at"] > date_bound]
            print(f"After removing old data, {name} table: {message_df.shape}")

            # update
            update_worksheet(message_df, sheet_name=sheet_name, mode="replace")
            message_df.to_csv(path, index=False)


# task 3: update new data (pages, conversations, messages)
def update_new_data(
    config: dict,
    remove_keywords: List[str] = None,
    filter_keywords: List[str] = None,
    templates: List[Dict[str, Any]] = None,
) -> List[dict]:
    # get new messages from pancake
    schema = os.path.join(PROJECT_DIRECTORY, config["schema"])
    default_last_check = config["default-last-check"]
    filter_patterns = config["filter-page-keywords"]
    messages = pancake_etl(schema, default_last_check, filter_patterns)
    messages = [m for m in messages if m["from"] == "customer"]

    # analyse messages by keywords
    if remove_keywords:
        messages = keyword_filter(remove_keywords, messages, get_keyword=False)

    if filter_keywords:
        messages = keyword_filter(filter_keywords, messages, get_keyword=True)

    # handle template messages
    template_messages = []
    if templates:
        template_messages, messages = handle_template_message(templates, messages)

    # store messages lists
    queue_message_path = get_project_path(config["queue-message"])
    if os.path.exists(queue_message_path):
        messages = load_json(queue_message_path) + messages

    return template_messages, messages


# task 4: load data to be analysed
def load_analyse_data(config: dict, messages: List[str], num_sample: int = 500):
    # load queue messages
    queue_path = get_project_path(config["queue-message"])
    queue_messages = messages
    if os.path.exists(queue_path):
        queue_messages = load_json(queue_path) + messages
        # deduplicate
        queue_messages = [
            json.loads(item)
            for item in {json.dumps(d, sort_keys=True) for d in queue_messages}
        ]

    # load messages needed to analyse
    analyse_messages = queue_messages[-num_sample:]
    queue_messages = queue_messages[:-num_sample]

    # store queue messages
    save_json(queue_path, queue_messages)

    return analyse_messages


# task 6: update tables
def update_table(config: dict, extracted_messages: List[dict], questions: List[dict]):
    # update message and question sheet
    updated_message_df = None  # keep this df for later
    for new_table, sheet_name, path in zip(
        (extracted_messages, questions),
        (config["message-sheet"], config["question-sheet"]),
        (
            get_project_path(config["message-table"]),
            get_project_path(config["question-table"]),
        ),
    ):
        if new_table:  # avoid empty list
            # load existence df
            load_table_args = {"path": path, "datetime_cols": ["inserted_at"]}
            name = "question"
            if sheet_name == config["message-sheet"]:
                load_table_args.update({"list_cols": ["user", "purpose"]})
                name = "message"
            last_df = load_table(**load_table_args)

            print(f"Before updating, {name} table: {last_df.shape}")

            # convert new extracted messages to df
            new_df = create_dataframe(new_table)

            # concat to newest df
            updated_df = pd.concat([last_df, new_df], axis=0)
            print(f"After updating, {name} table: {updated_df.shape}")

            # deduplicate
            updated_df = drop_dataframe_duplicates(updated_df)
            if sheet_name == config["message-sheet"]:
                updated_message_df = updated_df.copy()

            # upload to google sheet and store local
            update_worksheet(
                updated_df,
                sheet_name=sheet_name,
                mode="replace",
                sorted_by="inserted_at",
            )
            updated_df.to_csv(os.path.join(PROJECT_DIRECTORY, path), index=False)

    # update 2 stats sheet
    if extracted_messages:  # avoid empty list
        user_sheet_name = config["user-sheet"]
        purpose_sheet_name = config["purpose-sheet"]
        user_df, purpose_df = quantify_data(updated_message_df)

        update_worksheet(user_df, sheet_name=user_sheet_name, mode="replace")
        user_df.to_csv(get_project_path(config["user-table"]))
        update_worksheet(purpose_df, sheet_name=purpose_sheet_name, mode="replace")
        purpose_df.to_csv(get_project_path(config["purpose-table"]))

    # clean spreadsheet
    clean_spreadsheet(
        sheets=[
            config["message-sheet"],
            config["question-sheet"],
            config["user-sheet"],
            config["purpose-sheet"],
        ]
    )


# complete ETL
def analyse_customer_message_pipeline():
    # 1. setup directory for etl pipeline
    config = load_yaml(get_project_path("config.yaml"))

    setup_data_directory(PROJECT_DIRECTORY, sub_dir_list=[config["data-directory"]])

    # 2. Check new day & delete old data
    remove_old_data(config)

    # 3. get new messages
    important_keywords = (
        config["product-keywords"] + config["important-message-keywords"]
    )
    template_messages, messages = update_new_data(
        config,
        remove_keywords=config["unimportant-message-keywords"],
        filter_keywords=important_keywords,
        templates=config["template-message"],
    )

    # 4. load analyse messages
    messages = load_analyse_data(config, messages, config["num-sample"])
    if not messages:
        return

    # 5. analysing
    extracted_messages, questions, error_messages = analyse_message_pipeline(
        messages,
        question_keywords=config["question-keywords"],
        important_score=config["important-score"],
        provider=config["provider"],
    )
    extracted_messages.extend(template_messages)

    # 6. store error messages to queue
    queue_path = get_project_path(config["queue-message"])
    if error_messages:
        queue_message = load_json(queue_path) + error_messages
        save_json(queue_path, queue_message)

    # 7. update tables
    update_table(config, extracted_messages, questions)

    # # 8. update api key
    # update_env_variable(config['provider'])


if __name__ == "__main__":
    analyse_customer_message_pipeline()
