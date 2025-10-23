"""Test cases for reader.py functions."""

import pathlib
import re

import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.io import reader


@pytest.mark.parametrize(
    "invalid_filename",
    [
        "asdf123-spiral_trace1_Dom.csv",  # missing ID
        "[5123456]asdf123_Dom.csv",  # missing task
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


def test_convert_start_time() -> None:
    """Test that start time is converted correctly."""
    dummy_data = pd.DataFrame({"epoch_time_in_seconds_start": [10**15]})
    with pytest.raises(ValueError, match="Error converting 'start_time' to datetime"):
        reader._convert_start_time(dummy_data)


def test_load_spiral(sample_spiral_data: pathlib.Path) -> None:
    """Test that spiral loads with string input and start time is moved to metadata."""
    spiral = reader.load_drawing_data(str(sample_spiral_data))
    assert isinstance(spiral, models.Drawing)
    assert "epoch_time_in_seconds_start" not in spiral.data.columns
    assert "start_time" in spiral.metadata
    assert "source_path" in spiral.metadata
    assert spiral.metadata["task"] == "spiral_trace1_Dom"


def test_load_trails(sample_trails_data: pathlib.Path) -> None:
    """Test that trails loads with string input and start time is moved to metadata."""
    trails = reader.load_drawing_data(str(sample_trails_data))
    assert isinstance(trails, models.Drawing)
    assert "epoch_time_in_seconds_start" not in trails.data.columns
    assert "start_time" in trails.metadata
    assert "source_path" in trails.metadata
    assert trails.metadata["task"] == "trail4"


def test_load_alpha(sample_alpha_data: pathlib.Path) -> None:
    """Test that alpha loads with string input and start time is moved to metadata."""
    alpha = reader.load_drawing_data(str(sample_alpha_data))
    assert isinstance(alpha, models.Drawing)
    assert "epoch_time_in_seconds_start" not in alpha.data.columns
    assert "start_time" in alpha.metadata
    assert "source_path" in alpha.metadata
    assert alpha.metadata["task"] == "Alpha_AtoZ"


def test_load_dsym(sample_dsym_data: pathlib.Path) -> None:
    """Test that dsym loads with string input and start time is moved to metadata."""
    dsym = reader.load_drawing_data(str(sample_dsym_data))
    assert isinstance(dsym, models.Drawing)
    assert "epoch_time_in_seconds_start" not in dsym.data.columns
    assert "start_time" in dsym.metadata
    assert "source_path" in dsym.metadata
    assert dsym.metadata["task"] == "dsym_2"


def test_load_spiral_invalid_extension(sample_spiral_data: pathlib.Path) -> None:
    """Test that loading a non-CSV file raises an error."""
    invalid_file = sample_spiral_data.with_suffix(".txt")
    filename = re.escape(str(invalid_file))
    with pytest.raises(IOError, match=f"Error reading file {filename}"):
        reader.load_drawing_data(invalid_file)
