"""Test cases for reader.py functions."""

import pathlib

import pandas as pd
import pytest

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
    filename = invalid_filename.replace("[", "\\[").replace("]", "\\]")
    with pytest.raises(
        ValueError,
        match=f"Filename does not match expected pattern: {filename}",
    ):
        reader._parse_filename(invalid_filename)


@pytest.mark.parametrize("missing_column", list(reader.DTYPE_MAP.keys()))
def test_check_missing_columns(sample_data: pathlib.Path, missing_column: str) -> None:
    """Test that missing columns raise a KeyError."""
    data = pd.read_csv(sample_data)
    data = data.drop(columns=[missing_column])
    with pytest.raises(KeyError, match=f"Missing required columns: {missing_column}"):
        reader._check_missing_columns(data)


def test_convert_start_time(sample_data: pathlib.Path) -> None:
    """Test that start time is converted correctly."""
    data = pd.read_csv(sample_data)
    data["epoch_time_in_seconds_start"] = data["epoch_time_in_seconds_start"] * 1000000
    with pytest.raises(ValueError, match="Error converting 'start_time' to datetime"):
        reader._convert_start_time(data)


def test_load_spiral_str_path(sample_data: pathlib.Path) -> None:
    """Test that loading a spiral file with a string path works."""
    spiral = reader.load_spiral(str(sample_data))
    assert spiral is not None


def test_load_spiral_invalid_extension(sample_data: pathlib.Path) -> None:
    """Test that loading a non-CSV file raises an error."""
    invalid_file = sample_data.with_suffix(".txt")
    filename = invalid_file.as_posix().replace("[", "\\[").replace("]", "\\]")
    with pytest.raises(IOError, match=f"Error reading file {filename}"):
        reader.load_spiral(invalid_file)


def test_load_spiral_drops_start_time(sample_data: pathlib.Path) -> None:
    """Test that the 'epoch_time_in_seconds_start' column is dropped."""
    spiral = reader.load_spiral(sample_data)
    assert "epoch_time_in_seconds_start" not in spiral.data.columns
