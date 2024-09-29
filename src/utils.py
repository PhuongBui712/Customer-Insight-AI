import os
import yaml
import json
import pytz
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal, List, Tuple, Union


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