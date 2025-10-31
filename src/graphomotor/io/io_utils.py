"""This module contains utility functions for reading data files."""

import pandas as pd

DTYPE_MAP = {
    "line_number": "int",
    "x": "float",
    "y": "float",
    "UTC_Timestamp": "float",
    "seconds": "float",
    "epoch_time_in_seconds_start": "float",
    "error": "str",
    "correct_path": "str",
    "actual_path": "str",
    "total_time": "float",
    "total_number_of_errors": "int",
}


def _check_missing_columns(data: pd.DataFrame, task_name: str) -> None:
    """Check for missing columns in the DataFrame.

    Args:
        data: DataFrame containing spiral drawing data.
        task_name: Name of the drawing task.

    Raises:
        KeyError: If any required columns are missing.
    """
    if "trail" in task_name.lower():
        required_columns = list(DTYPE_MAP.keys())
    else:
        required_columns = list(DTYPE_MAP.keys())[:6]

    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")
