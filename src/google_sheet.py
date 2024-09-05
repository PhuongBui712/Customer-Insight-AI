import os
import gspread
from google.oauth2.service_account import Credentials

import pandas as pd
from pandas import DataFrame
from typing import Optional, Literal

from dotenv import load_dotenv


load_dotenv()

_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets'
]

_creds = Credentials.from_service_account_file("credentials.json", scopes=_SCOPES)
_client = gspread.authorize(_creds)
sheet_id = os.environ['SHEET_ID']
sheet = _client.open_by_key(sheet_id)


def load_worksheet(sheet_id: Optional[str] = None, sheet_idx: Optional[int] = None) -> DataFrame:
    if sheet_id is None and sheet_idx is None:
        raise Exception("Please provide either a sheet id or sheet index.")

    worksheet = None
    if sheet_id:
        worksheet = sheet.get_worksheet_by_id(sheet_id)
    else:
        worksheet = sheet.get_worksheet(sheet_idx)

    return pd.DataFrame(worksheet.get_all_records())


def update_worksheet(dataframe: DataFrame, sheet_id: Optional[str] = None, sheet_idx: Optional[int] = None, mode: Literal['update', 'replace'] = 'update') -> None:
    if sheet_id is None and sheet_idx is None:
        raise Exception("Please provide either `sheet_id` or `sheet_idx")
    
    # get worksheet
    if sheet_id:
        worksheet = sheet.get_worksheet_by_id(sheet_id)
    else:
        worksheet = sheet.get_worksheet(sheet_idx)

    # update worksheet
    if mode == 'replace':
        worksheet.clear()
    
    worksheet.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())


if __name__ == '__main__':
    # 1. load worksheet
    df = load_worksheet(sheet_idx=0)
    print(df)

    # 2. update new worksheet
    new_df = pd.DataFrame({'col a': [1,2,3,4,5], 'col b': [11,12,13,14,15]})
    update_worksheet(new_df, sheet_idx=0, mode='replace')

    # reload worksheet (updated)
    df = load_worksheet(sheet_idx=0)
    print(df)