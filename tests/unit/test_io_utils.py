"""Unit tests for the io_utils module."""

import pandas as pd
import pytest

from graphomotor.io import io_utils


@pytest.mark.parametrize("missing_column", list(io_utils.DTYPE_MAP.keys())[:6])
def test_check_missing_columns(
    valid_spiral_data: pd.DataFrame, missing_column: str
) -> None:
    """Test that missing columns raise a KeyError."""
    valid_spiral_data = valid_spiral_data.drop(columns=[missing_column])
    with pytest.raises(KeyError, match=f"Missing required columns: {missing_column}"):
        io_utils._check_missing_columns(valid_spiral_data, task_name="spiral")


def test_check_missing_columns_trails() -> None:
    """Test that missing columns raise a KeyError for trails data."""
    data = pd.DataFrame(
        {
            "line_number": [1, 2],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "UTC_Timestamp": [1620000000.0, 1620000001.0],
            "seconds": [0.0, 1.0],
            "epoch_time_in_seconds_start": [1620000000.0, 1620000000.0],
            "error": ["", ""],
            "correct_path": ["A", "B"],
            "actual_path": ["A", "C"],
            "total_time": [10.0, 10.0],
            "total_number_of_errors": [0, 1],
        }
    )
    data = data.drop(columns=["error"])
    with pytest.raises(KeyError, match="Missing required columns: error"):
        io_utils._check_missing_columns(data, task_name="trail")
