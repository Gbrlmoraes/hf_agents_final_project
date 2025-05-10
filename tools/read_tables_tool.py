import os
from langchain_core.tools import tool
import pandas as pd
from typing import Literal


@tool
def ReadTablesTool(file_path: str, file_type: Literal['csv', 'excel']) -> str:
    """
    Returns the rows of a table from a CSV or Excel file.
    This tool is expected to work with small files.

    Args:
        file_path: The path to the file to be read.
        file_type: The type of the file to be read. Must be 'csv' or 'excel'.
    """

    READ_FUNCTIONS = {
        'csv': pd.read_csv,
        'excel': pd.read_excel,
    }

    if file_type not in READ_FUNCTIONS:
        return f"Unsupported file type: '{file_type}'. Only 'csv' and 'excel' are supported."

    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at '{file_path}'."

        read_function = READ_FUNCTIONS[file_type]
        df = read_function(file_path)

        # Define limits for table size
        MAX_ROWS = 50
        MAX_COLS = 10

        if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
            return f'The file is too large to process. It has {df.shape[0]} rows and {df.shape[1]} columns. Maximum allowed is {MAX_ROWS} rows and {MAX_COLS} columns.'

        if df.empty:
            return 'The file is empty or contains no data.'

        table_content = df.to_string(index=False)
        return f'Table content:\n{table_content}'

    except pd.errors.EmptyDataError:
        return f"Error reading the file '{file_path}': The file is empty or contains no data to parse."
    except pd.errors.ParserError:
        return f"Error reading the file '{file_path}': Could not parse the file. Ensure it is a valid {file_type} file."
    except Exception as e:
        return f"An unexpected error occurred while reading the file '{file_path}': {e}"


if __name__ == '__main__':
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')

    files = ['test_table.xlsx']

    for file in files:
        file_path = os.path.join(source_dir, file)
        print(
            ReadTablesTool.invoke(input={'file_path': file_path, 'file_type': 'excel'})
        )
