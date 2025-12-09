"""Tests for trails time.py."""

import pandas as pd

from graphomotor.core import models
from graphomotor.features.trails import time


def test_total_error_time_no_errors() -> None:
    """Test case with no errors."""
    df = pd.DataFrame({
        "error": ["E0", "E0", "E0", "E0"],
        "seconds": [0, 1, 2, 3],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})

    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 0.0}


def test_single_error_chunk() -> None:
    """Test case with a single error chunk."""
    df = pd.DataFrame({
        "error": ["E0", "E1", "E1", "E0", "E0"],
        "seconds": [0.0, 1.0, 3.0, 5.0, 6.0],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})

    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 3.5}


def test_multiple_error_chunks() -> None:
    """Test case with multiple error chunks."""
    df = pd.DataFrame({
        "error": ["E0", "E1", "E1", "E0", "E2", "E2", "E0"],
        "seconds": [0, 1, 3, 5, 6, 7, 9],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 6.0}


def test_error_at_end() -> None:
    """Test case with an error chunk that goes to the end of the drawing."""
    df = pd.DataFrame({
        "error": ["E0", "E0", "E2", "E2"],
        "seconds": [0.0, 1.0, 2.0, 4.0],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 2.5}


def test_error_at_start() -> None:
    """Test case with an error chunk that starts at the beginning of the drawing."""
    df = pd.DataFrame({
        "error": ["E1", "E1", "E0", "E0"],
        "seconds": [0.0, 1.0, 3.0, 4.0],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 2.0}


def test_calculate_correct_time() -> None:
    """Test calculation of correct drawing time."""
    df = pd.DataFrame({
        "error": ["E0", "E1", "E0", "E2", "E0"],
        "seconds": [0.0, 1.0, 3.0, 5.0, 7.0],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_correct_time(drawing)
    assert result == {"total_correct_time": 3.5}
