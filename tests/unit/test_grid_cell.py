"""Test cases for the GridCell model."""

import pandas as pd
import pytest

from graphomotor.core import models


@pytest.fixture
def cell() -> models.GridCell:
    """Create a grid cell representing one letter region."""
    return models.GridCell(
        x_min=10.0, x_max=25.0, y_min=80.0, y_max=97.0, index=0, label="A"
    )


@pytest.mark.parametrize(
    "x_min,x_max,y_min,y_max,expected_error",
    [
        (10.0, 5.0, 0.0, 1.0, "x_min .* must be less than x_max"),
        (5.0, 5.0, 0.0, 1.0, "x_min .* must be less than x_max"),
        (0.0, 1.0, 10.0, 5.0, "y_min .* must be less than y_max"),
        (0.0, 1.0, 5.0, 5.0, "y_min .* must be less than y_max"),
    ],
)
def test_grid_cell_invalid_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    expected_error: str,
) -> None:
    """Test that invalid or equal bounds raise ValueError."""
    with pytest.raises(ValueError, match=expected_error):
        models.GridCell(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def test_stroke_centroid_inside_cell(cell: models.GridCell) -> None:
    """Stroke whose centroid falls inside the cell should be contained."""
    stroke_points = pd.DataFrame({"x": [15.0, 20.0, 17.0], "y": [85.0, 90.0, 95.0]})
    assert cell.contains_points(stroke_points)


def test_stroke_centroid_outside_cell(cell: models.GridCell) -> None:
    """Stroke whose centroid falls outside the cell should not be contained."""
    stroke_points = pd.DataFrame({"x": [33.8, 37.8, 35.3], "y": [85.8, 95.4, 90.1]})
    assert not cell.contains_points(stroke_points)


def test_stroke_centroid_on_lower_boundary(cell: models.GridCell) -> None:
    """Stroke whose centroid lands on the lower/left boundary (min) is included."""
    stroke_points = pd.DataFrame({"x": [10.0, 10.0], "y": [88.0, 89.0]})
    assert cell.contains_points(stroke_points)


def test_stroke_centroid_on_upper_boundary(cell: models.GridCell) -> None:
    """Stroke whose centroid lands on the upper/right boundary (max) is excluded."""
    stroke_points = pd.DataFrame({"x": [17.0, 18.0], "y": [97.0, 97.0]})
    assert not cell.contains_points(stroke_points)


def test_stroke_points_span_outside_but_centroid_inside(
    cell: models.GridCell,
) -> None:
    """Stroke with points outside the cell but centroid inside should be contained."""
    stroke_points = pd.DataFrame({"x": [8.0, 22.0], "y": [78.0, 98.0]})
    assert cell.contains_points(stroke_points)


def test_single_point_stroke(cell: models.GridCell) -> None:
    """Single-point stroke should use that point as its centroid."""
    stroke_points = pd.DataFrame({"x": [17.5], "y": [90.0]})
    assert cell.contains_points(stroke_points)
