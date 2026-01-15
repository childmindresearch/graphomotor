"""Test cases for drawing_metrics.py functions."""

import numpy as np
import pandas as pd
import pytest

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


def test_smoothness_less_than_three_points() -> None:
    """Less than 3 points cannot define curvature."""
    points = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    assert drawing_metrics.calculate_smoothness(points) == 0.0


def test_smoothness_straight_line() -> None:
    """Collinear points have zero curvature."""
    points = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 0, 0, 0]})
    assert drawing_metrics.calculate_smoothness(points) == 0.0


def test_smoothness_single_right_angle() -> None:
    """A single 90-degree corner should produce a large smoothness value.

    (non-zero), since sharp turns are penalized.
    """
    points = pd.DataFrame({"x": [0, 1, 1], "y": [0, 0, 1]})
    expected = np.pi / 2
    smoothness = drawing_metrics.calculate_smoothness(points)
    assert np.isclose(smoothness, expected)


def test_smoothness_varied_angles() -> None:
    """Multiple angles should produce RMS curvature.

    Path has a 90° turn followed by a 45° turn.
    """
    points = pd.DataFrame({"x": [0, 1, 1, 2], "y": [0, 0, 1, 2]})
    c1 = np.pi / 2
    c2 = (np.pi / 4) / ((1 + np.sqrt(2)) / 2)
    expected = np.sqrt((c1**2 + c2**2) / 2)

    smoothness = drawing_metrics.calculate_smoothness(points)

    assert np.isclose(smoothness, expected)


def test_smoothness_zero_length_segments() -> None:
    """Zero-length segments should be skipped; no angles → smoothness 0."""
    points = pd.DataFrame({"x": [0, 1, 1, 2], "y": [0, 0, 0, 0]})
    smoothness = drawing_metrics.calculate_smoothness(points)
    assert smoothness == 0.0


def test_smoothness_single_180_degree_turn() -> None:
    """A single 180-degree turn should produce a very large smoothness value.

    Since it represents maximal curvature.
    """
    points = pd.DataFrame({
        "x": [0, 1, 0],
        "y": [0, 0, 0],
    })
    expected = np.pi
    smoothness = drawing_metrics.calculate_smoothness(points)
    assert np.isclose(smoothness, expected)
