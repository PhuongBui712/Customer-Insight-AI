import os
import yaml
import json
import pytz
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal, List, Tuple, Union

from dotenv import load_dotenv


load_dotenv()


NUM_WORKERS = min(os.cpu_count() - 1, max(7, os.cpu_count() // 2))
PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        res = yaml.safe_load(file)

    return res


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        result = json.load(file)

    return result


def save_json(path: str, content: Union[dict, list], indent: int = 4) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(content, file, indent=indent, ensure_ascii=False)


def get_project_item_path(relevant_path: str) -> str:
    return os.path.join(PROJECT_DIRECTORY, relevant_path)


def string_to_unix_second(s: str,
                          format: str = "%Y-%m-%dT%H:%M:%S",
                          tz: str = 'UTC',
                          return_tz: str = 'Asia/Bangkok') -> int:
    added_second = 0
    try:
        rounded_s = s.split('.')[0]
        added_second = 1 if int(s.split('.')[1]) >= 0.5 else 0
    except Exception:
        rounded_s = s

    # convert to `datetime`
    dt = datetime.strptime(rounded_s, format)
    tz = pytz.timezone(tz)
    dt = tz.localize(dt)

    if return_tz:
        tz = pytz.timezone(return_tz)
        dt = dt.astimezone(tz)

    timestamp = int(dt.timestamp())

    return timestamp + added_second


def get_current_time_utc_plus_7():
    # Define UTC+7 timezone
    utc_plus_7 = timezone(timedelta(hours=7))
    
    # Get the current time in UTC+7
    current_time_utc_plus_7 = datetime.now(utc_plus_7)
    
    # Remove microseconds and timezone information
    current_time_utc_plus_7 = current_time_utc_plus_7.replace(microsecond=0, tzinfo=None)
    
    return current_time_utc_plus_7   


def get_day_before(num_days: int, return_type: Literal['date', 'timestamp'] = 'timestamp'):
    now = get_current_time_utc_plus_7()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    result = start_of_day - timedelta(days=num_days)
    if return_type == 'timestamp':
        result = int(result.timestamp())

    return result


def split_time_stamp(since: int, until: int, time_delta: int = 30*24*60*60) -> List[Tuple[int, int]]:
    result = [(t, min(t + time_delta, until)) for t in range(since, until, time_delta)]
    return result 


def drop_dataframe_duplicates(df: DataFrame) -> DataFrame:
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Dictionary to track columns that need conversion back
    conversion_columns = []

    # Iterate over columns
    for col in df_copy.columns:
        # Check if the column contains unhashable types (e.g., lists or dictionaries)
        if df_copy[col].apply(lambda x: isinstance(x, (list, dict))).any():
            # Convert the unhashable column to tuples
            df_copy[col] = df_copy[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
            # Track the columns that were converted
            conversion_columns.append(col)

    # Drop duplicates
    df_copy = df_copy.drop_duplicates()

    # Convert the columns that were modified back to their original types (lists, etc.)
    for col in conversion_columns:
        df_copy[col] = df_copy[col].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    return df_copy


def update_env_variable(provider: Literal['groq', 'google'] = 'groq'):
    # dotenv path
    dotenv_path = get_project_item_path(".env")
    
    # key's name pattern
    key = provider.upper() + "_API_KEY"
    
    # Read all the lines from the .env file
    with open(dotenv_path, 'r') as file:
        lines = file.readlines()

    # Find the current value of the key
    current_value = os.getenv(key)
    
    # Incremental check for keys (key1, key2, etc.)
    idx = 1
    while True:
        next_key = f"{key}{idx}"
        next_value = os.getenv(next_key)
        
        # If the next key doesn't exist, go back to key1
        if next_value is None:
            next_key = f"{key}1"
            next_value = os.getenv(next_key)
            break
        elif current_value == next_value:
            idx += 1
        else:
            break

    # Check if the key exists and update it
    with open(dotenv_path, 'w') as file:
        key_found = False
        for line in lines:
            if line.startswith(f"{key}="):
                file.write(f"{key}={next_value}\n")
                key_found = True
            else:
                file.write(line)
        
        # If the key wasn't found, add it at the end
        if not key_found:
            file.write(f"{key}={next_value}\n")

    # Reload the environment variables
    load_dotenv(dotenv_path, override=True)