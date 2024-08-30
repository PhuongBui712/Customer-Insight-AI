import os
import yaml
import json
import pytz
from datetime import datetime, timedelta
from typing import Optional, Literal, List, Tuple

NUM_WORKERS = os.cpu_count() 


def load_yaml_file(path: str) -> dict:
    with open(path, 'r') as file:
        res = yaml.safe_load(file)

    return res


def read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as file:
        result = json.load(file)

    return result


def save_json(path: str, content: dict, indent: int = 4) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(content, file, indent=indent)


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
    dt = datetime.strptime(rounded_s, format=format)
    tz = pytz.timezone(tz)
    dt = tz.localize(dt)

    if return_tz:
        tz = pytz.timezone(return_tz)
        dt = dt.astimezone(tz)

    timestamp = int(dt.timestamp())

    return timestamp + added_second


def get_day_before(num_days: int, return_type: Literal['date', 'timestamp'] = 'timestamp', timezone: str = 'Asisa/Bangkok'):
    now = datetime.now()
    tz = pytz.timezone(timezone)
    now = tz.localize(now)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    result = start_of_day - timedelta(days=num_days)
    if return_type == 'timestamp':
        result = int(result.timestamp)

    return result


def split_time_stamp(since: int, until: int, time_delta: int = 30*24*60*60) -> List[Tuple[int, int]]:
    result = [(t, min(t + time_delta, until)) for t in range(since, until, time_delta)]
    return result