"""Test cases for reader.py functions."""

from pathlib import Path

import pandas as pd
import pytest

from graphomotor.io import reader


def test_parse_filename_valid(sample_data: Path) -> None:
    """Test that valid filenames are parsed correctly."""
    expected_metadata = {
        "id": "5123456",
        "hand": "Dom",
        "task": "spiral_trace1",
    }
    metadata = reader.parse_filename(sample_data.stem)
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
        reader.parse_filename(invalid_filename)


def test_load_spiral_invalid_extension(sample_data: Path) -> None:
    """Test that loading a non-CSV file raises an error."""
    invalid_file = sample_data.with_suffix(".txt")
    filename = invalid_file.as_posix().replace("[", "\\[").replace("]", "\\]")
    with pytest.raises(IOError, match=f"Error reading file {filename}"):
        reader.load_spiral(invalid_file)


@pytest.mark.parametrize("missing_column", reader.DATA_COLUMNS)
def test_load_spiral_missing_columns(sample_data: Path, missing_column: str) -> None:
    """Test validation error when required columns are missing."""
    data = pd.read_csv(sample_data)
    data = data.drop(columns=[missing_column])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_csv", lambda *args, **kwargs: data)
        with pytest.raises(
            KeyError,
            match=r"Missing required columns: \['" + missing_column + r"'\]",
        ):
            reader.load_spiral(sample_data)


@pytest.mark.parametrize(
    "column,expected_type",
    [
        ("line_number", "int"),
        ("x", "float"),
        ("y", "float"),
        ("UTC_Timestamp", "float"),
        ("seconds", "float"),
    ],
)
def test_load_spiral_invalid_data_types(
    sample_data: Path, column: str, expected_type: str
) -> None:
    """Test validation error when data types are incorrect."""
    data = pd.read_csv(sample_data)
    data[column] = data[column].astype(str)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_csv", lambda *args, **kwargs: data)
        with pytest.raises(
            ValueError, match=f"'{column}' should be of type {expected_type}"
        ):
            reader.load_spiral(sample_data)


def test_load_spiral_invalid_timestamp(sample_data: Path) -> None:
    """Test validation error when UTC_Timestamp is invalid."""
    data = pd.read_csv(sample_data)
    data["UTC_Timestamp"] = data["UTC_Timestamp"] * 1000000
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_csv", lambda *args, **kwargs: data)
        with pytest.raises(
            ValueError,
            match="Error converting 'UTC_Timestamp' to datetime",
        ):
            reader.load_spiral(sample_data)


def test_load_spiral_drops_start_time(sample_data: Path) -> None:
    """Test that the 'epoch_time_in_seconds_start' column is dropped."""
    spiral = reader.load_spiral(sample_data)
    assert "epoch_time_in_seconds_start" not in spiral.data.columns
