"""Test cases for drawing_metrics.py functions."""

import numpy as np
import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.features.trails import drawing_metrics
from graphomotor.io import reader


def test_get_total_errors() -> None:
    """Test ValueError when total_number_of_errors column doesn't exist."""
    invalid_df = pd.DataFrame({
        "some_other_column": [0, 1, 2],
        "seconds": [0.0, 1.0, 2.0],
    })
    drawing = models.Drawing(
        data=invalid_df, task_name="trails", metadata={"id": "5555555"}
    )

    with pytest.raises(
        ValueError,
        match="Drawing data does not contain 'total_number_of_errors' column.",
    ):
        drawing_metrics.get_total_errors(drawing)


def test_valid_total_errors() -> None:
    """Test case with valid total_number_of_errors column."""
    filepath = "tests/sample_data/[5000000]648b6b868819c1120b4f6ce3-trail4.csv"
    drawing = reader.load_drawing_data(filepath)

    result = drawing_metrics.get_total_errors(drawing)
    assert result == {"total_errors": 1.0}


def test_percent_accurate_paths_missing_columns() -> None:
    """Test ValueError when required columns are missing."""
    invalid_df = pd.DataFrame({
        "some_other_column": [0, 1, 2],
        "seconds": [0.0, 1.0, 2.0],
    })
    drawing = models.Drawing(
        data=invalid_df, task_name="trails", metadata={"id": "5555555"}
    )

    with pytest.raises(
        ValueError,
        match="DataFrame must contain 'correct_path' and 'actual_path' columns.",
    ):
        drawing_metrics.percent_accurate_paths(drawing)


def test_percent_accurate_paths_sample_data() -> None:
    """Test percent_accurate_paths with sample drawing data."""
    filepath = "tests/sample_data/[5000000]648b6b868819c1120b4f6ce3-trail4.csv"
    drawing = reader.load_drawing_data(filepath)

    result = drawing_metrics.percent_accurate_paths(drawing)
    assert result == {"percent_accurate_paths": 100.0}
