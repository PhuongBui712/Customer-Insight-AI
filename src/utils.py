import os
import yaml
import json
import time
from datetime import datetime


NUM_THREADS = os.cpu_count() 


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


def string_to_unix_second(s: str, format: str = "%Y-%m-%dT%H:%M:%S") -> int:
    dt = datetime.strptime(s, format=format)
    timestamp = int(dt.timestamp())

    return timestamp