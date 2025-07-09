import os
import pandas as pd


def write_csv(df: pd.DataFrame, folder_path: str, file_name: str) -> None:
    """
    Записывает список словарей в CSV-файл.
    """
    if df.empty:
        raise ValueError("DataFrame пуст. Нечего записывать.")

    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)


def read_csv(folder_path: str, file_name: str) -> pd.DataFrame:
    """
    Считывает CSV-файл в DataFrame.

    """
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")

    return pd.read_csv(file_path)