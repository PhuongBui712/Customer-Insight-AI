import os
import gspread
from google.oauth2.service_account import Credentials

import pandas as pd
from pandas import DataFrame
from typing import Optional, Literal, List

from dotenv import load_dotenv

from utils import PROJECT_DIRECTORY


load_dotenv()

_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets'
]

_credential_file = os.path.join(PROJECT_DIRECTORY, "credentials.json")
_creds = Credentials.from_service_account_file(_credential_file, scopes=_SCOPES)
_client = gspread.authorize(_creds)
sheet_id = os.environ['SHEET_ID']
spreadsheet = _client.open_by_key(sheet_id)


def load_worksheet(sheet_name: str, return_type: Literal['sheet', 'dataframe'] = 'dataframe') -> DataFrame:
    """
    Load a worksheet by name. If it doesn't exist, create it.
    
    Args:
    sheet_name (str): The name of the worksheet to load or create.
    spreadsheet (gspread.Spreadsheet): The spreadsheet object containing the worksheet.
    
    Returns:
    DataFrame: A DataFrame containing the worksheet data.
    """
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=100)
    
    if return_type == 'sheet':
        return worksheet

    # Get all records from the worksheet
    records = worksheet.get_all_records()
    
    if not records:
        return DataFrame()
    return DataFrame(records)


def update_worksheet(
        dataframe: DataFrame,
        sheet_name: str,
        mode: Literal['update', 'replace'] = 'update',
        sorted_by: Optional[str] = None,
        ascending: bool = False
) -> None:
    worksheet = load_worksheet(sheet_name=sheet_name, return_type='sheet')

    # update worksheet
    if mode == 'replace':
        worksheet.clear()
    
    # preprocess df
    upload_df = dataframe.copy()

    # sort df by datetime
    if sorted_by is not None:
        upload_df = upload_df.sort_values(by=sorted_by, ascending=ascending)
    
    # astype datetime to string
    datetime_columns = upload_df.select_dtypes(include=['datetime64[ns]']).columns
    upload_df[datetime_columns] = upload_df[datetime_columns].astype(str)
    
    # convert a list to string
    for col in upload_df.columns:
        if upload_df[col].apply(lambda x: isinstance(x, list)).any():
            upload_df[col] = upload_df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

    try:
        worksheet.update([upload_df.columns.values.tolist()] + upload_df.values.tolist())
    except Exception as exc:
        print(exc)
        print(f"Data:\n{[upload_df.columns.values.tolist()] + upload_df.values.tolist()}")


def clean_spreadsheet(sheets: List[str]) -> None:
    all_sheets = spreadsheet.worksheets()
    for worksheet in all_sheets:
        if worksheet.title not in sheets:
            spreadsheet.del_worksheet(worksheet)

# --------------------- Specific functions ---------------------
# These functions are specific to the data structure used in this
# project and may not be applicable to all data.
def create_dataframe(messages: List[dict]) -> DataFrame:
    """
    Creates a Pandas DataFrame from a list of messages.

    This function takes a list of dictionaries representing messages, each containing the following keys:
    - `message`: The message content.
    - `from`: The sender of the message.
    - `inserted_at`: The timestamp of the message insertion.
    - `user`: The user associated with the message.
    - `purpose`: The purpose of the message.

    The function converts the list of dictionaries into a Pandas DataFrame and performs the following transformations on the 'inserted_at' column:

    1. Converts the string representation of the timestamp to a datetime object.
    2. Localizes the timestamp to UTC.
    3. Converts the timestamp to the Asia/Bangkok timezone.

    Args:
        messages (List[dict]): A list of dictionaries representing messages.

    Returns:
        DataFrame: A Pandas DataFrame containing the messages.
    """
    df = pd.DataFrame(data=messages)

    # convert string to datetime
    df['inserted_at'] = df['inserted_at'].str.replace('T', ' ')
    df['inserted_at'] = df['inserted_at'].str.replace(r'\.\d{6}', '', regex=True)
    df['inserted_at'] = pd.to_datetime(df['inserted_at'], format='%Y-%m-%d %H:%M:%S')
    
    # transform timezone
    df['inserted_at'] = df['inserted_at'].dt.tz_localize('UTC')
    df['inserted_at'] = df['inserted_at'].dt.tz_convert('Asia/Bangkok')
    df['inserted_at'] = df['inserted_at'].dt.tz_localize(None)

    return df


def sheet_to_df(raw_sheet: DataFrame) -> DataFrame:
    if raw_sheet.empty:
        return raw_sheet
    
    # convert to datetime
    df = raw_sheet.copy()
    df['inserted_at'] = pd.to_datetime(df['inserted_at'], format='%Y-%m-%d %H:%M:%S')
    
    # convert string -> list[str]
    for col in ['user', 'purpose']:
        try:
            df[col] = df[col].apply(lambda x: x.split(',') if ',' in x else [x])
        except Exception as exc:
            continue

    return df


def quantify_data(message_df: DataFrame, user_col: str = 'user', purpose_col: str = 'purpose') -> DataFrame:
    user = message_df[[user_col]]
    count_user = user.explode([user_col])
    count_user = count_user[count_user[user_col] != '']
    count_user = count_user.value_counts(user_col, ascending=False).reset_index()

    purpose = message_df[[purpose_col]]
    count_purpose = purpose.explode([purpose_col])
    count_purpose = count_purpose[count_purpose[purpose_col] != '']
    count_purpose = count_purpose.value_counts(purpose_col, ascending=False).reset_index()

    return count_user, count_purpose


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