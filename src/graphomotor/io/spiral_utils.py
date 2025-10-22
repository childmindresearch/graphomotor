"""This module contains utility functions for reading data files."""

import pandas as pd

DTYPE_MAP = {
    "line_number": "int",
    "x": "float",
    "y": "float",
    "UTC_Timestamp": "float",
    "seconds": "float",
    "epoch_time_in_seconds_start": "float",
}


def _check_missing_columns(data: pd.DataFrame) -> None:
    """Check for missing columns in the DataFrame.

    Args:
        data: DataFrame containing spiral drawing data.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing_columns = set(DTYPE_MAP.keys()) - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")
