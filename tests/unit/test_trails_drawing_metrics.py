"""Test cases for drawing_metrics.py functions."""

import pandas as pd
import pytest
import scipy.spatial.distance as dist

from graphomotor.core import models
from graphomotor.features.trails import drawing_metrics
from graphomotor.io import reader


def test_no_pen_lifts() -> None:
    """Test case with no pen lifts."""
    df = pd.DataFrame({"line_number": [1, 1, 1, 1]})
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = drawing_metrics.detect_pen_lifts(drawing)
    assert result == {"pen_lifts": 0}


def test_valid_pen_lifts() -> None:
    """Test case with valid pen lifts."""
    df = pd.DataFrame({"line_number": [1, 2, 3, 4]})
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = drawing_metrics.detect_pen_lifts(drawing)
    assert result == {"pen_lifts": 3}


def test_get_total_errors() -> None:
    """Test ValueError when total_number_of_errors column doesn't exist."""
    invalid_df = pd.DataFrame({"some_other_column": [0, 1, 2]})
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


def test_path_optimality_positive() -> None:
    """Test case for path optimality with positive optimal distance."""
    start = models.CircleTarget(order=1, label="1", center_x=0, center_y=0, radius=1)
    end = models.CircleTarget(order=2, label="2", center_x=10, center_y=0, radius=1)
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=pd.DataFrame(),
        is_error=False,
        line_number=1,
        distance=8,
    )

    drawing_metrics.calculate_path_optimality(segment, start, end)

    expected_optimal_distance = 10 - 1 - 1
    assert segment.path_optimality == expected_optimal_distance / 8


def test_path_optimality_non_positive_distance() -> None:
    """Test case where optimal distance is zero or negative, so no assignment occurs."""
    start = models.CircleTarget(order=1, label="1", center_x=0, center_y=0, radius=5)
    end = models.CircleTarget(order=2, label="2", center_x=8, center_y=0, radius=5)
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=pd.DataFrame(),
        is_error=False,
        line_number=1,
        distance=5,
    )

    drawing_metrics.calculate_path_optimality(segment, start, end)

    assert segment.path_optimality == 0.0
