"""Test cases for reader.py functions."""

import pathlib
import re

import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.io import reader


def test_parse_filename_valid(sample_data: pathlib.Path) -> None:
    """Test that valid filenames are parsed correctly."""
    expected_metadata = {
        "id": "5123456",
        "hand": "Dom",
        "task": "spiral_trace1",
    }
    metadata = reader._parse_filename(sample_data.stem)
    assert metadata == expected_metadata, (
        f"Expected {expected_metadata}, but got {metadata}"
    )


@pytest.mark.parametrize(
    "invalid_filename",
    [
        "asdf123-spiral_trace1_Dom.csv",  # missing ID
        "[5123456]-spiral_trace1_Dom.csv",  # missing mindloggerID
        "[5123456]asdf123-Dom.csv",  # missing task
        "[5123456]asdf123-spiral_trace1.csv",  # missing hand
    ],
)
def test_parse_filename_invalid(invalid_filename: str) -> None:
    """Test that invalid filenames raise a ValueError."""
    filename = re.escape(invalid_filename)
    with pytest.raises(
        ValueError,
        match=f"Filename does not match expected pattern: {filename}",
    ):
        reader._parse_filename(invalid_filename)


@pytest.mark.parametrize("missing_column", list(reader.DTYPE_MAP.keys()))
def test_check_missing_columns(
    valid_spiral_data: pd.DataFrame, missing_column: str
) -> None:
    """Test that missing columns raise a KeyError."""
    valid_spiral_data = valid_spiral_data.drop(columns=[missing_column])
    with pytest.raises(KeyError, match=f"Missing required columns: {missing_column}"):
        reader._check_missing_columns(valid_spiral_data)


def test_convert_start_time() -> None:
    """Test that start time is converted correctly."""
    dummy_data = pd.DataFrame({"epoch_time_in_seconds_start": [10**15]})
    with pytest.raises(ValueError, match="Error converting 'start_time' to datetime"):
        reader._convert_start_time(dummy_data)


def test_load_spiral(sample_data: pathlib.Path) -> None:
    """Test that spiral loads with string input and start time is moved to metadata."""
    spiral = reader.load_spiral(str(sample_data))
    assert isinstance(spiral, models.Spiral)
    assert "epoch_time_in_seconds_start" not in spiral.data.columns
    assert "start_time" in spiral.metadata


def test_load_spiral_invalid_extension(sample_data: pathlib.Path) -> None:
    """Test that loading a non-CSV file raises an error."""
    invalid_file = sample_data.with_suffix(".txt")
    filename = re.escape(str(invalid_file))
    with pytest.raises(IOError, match=f"Error reading file {filename}"):
        reader.load_spiral(invalid_file)
