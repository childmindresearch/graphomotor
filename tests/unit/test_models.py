"""Test cases for the Spiral model."""

import datetime

import pandas as pd
import pytest

from graphomotor.core import models


def test_valid_spiral_creation(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
) -> None:
    """Test creating a valid Drawing instance."""
    spiral = models.Drawing(
        data=valid_spiral_data,
        task_name="spiral_drawing",
        metadata=valid_spiral_metadata,
    )
    assert spiral.data.equals(valid_spiral_data)
    assert spiral.metadata == valid_spiral_metadata
    assert spiral.task_name == "spiral_drawing"


def test_empty_dataframe(
    valid_spiral_metadata: dict[str, str | datetime.datetime],
) -> None:
    """Test validation error when DataFrame is empty."""
    empty_data = pd.DataFrame(
        columns=["line_number", "x", "y", "UTC_Timestamp", "seconds"]
    )

    with pytest.raises(ValueError, match="DataFrame is empty"):
        models.Drawing(
            data=empty_data, task_name="spiral", metadata=valid_spiral_metadata
        )


@pytest.mark.parametrize(
    "key,invalid_value,expected_error",
    [
        ("id", "1001", "'id' must start with digit 5"),
        ("id", "512345", "'id' must be 7 digits long"),
    ],
)
def test_invalid_metadata_values(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
    key: str,
    invalid_value: str | datetime.datetime,
    expected_error: str,
) -> None:
    """Test validation errors for various invalid metadata values."""
    invalid_metadata = valid_spiral_metadata.copy()
    invalid_metadata[key] = invalid_value

    with pytest.raises(ValueError, match=expected_error):
        models.Drawing(
            data=valid_spiral_data, task_name="spiral", metadata=invalid_metadata
        )


@pytest.mark.parametrize(
    "key,time,duplicate_time,expected_error",
    [
        (
            "UTC_Timestamp",
            16.596,
            16.596,
            "duplicate timestamps in 'UTC_Timestamp'.",
        ),
        ("seconds", 1.73, 1.73, "duplicate timestamps in 'seconds'."),
    ],
)
def test_duplicate_timestamps(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
    key: str,
    time: str | float,
    duplicate_time: str | float,
    expected_error: str,
) -> None:
    """Test that duplicate timestamps in the DataFrame aren't allowed."""
    invalid_data = valid_spiral_data.copy()
    invalid_data[key][0] = time
    invalid_data[key][1] = duplicate_time

    with pytest.raises(ValueError, match=expected_error):
        models.Drawing(
            data=invalid_data, task_name="spiral", metadata=valid_spiral_metadata
        )


@pytest.fixture
def circle() -> models.CircleTarget:
    """Create a standard circle at origin with radius 10."""
    return models.CircleTarget(
        order=1, label="test_circle", center_x=0.0, center_y=0.0, radius=10.0
    )


@pytest.mark.parametrize(
    "x,y,description",
    [
        (0.0, 0.0, "center"),
        (10.0, 0.0, "right edge"),
        (0.0, 10.0, "top edge"),
        (-10.0, 0.0, "left edge"),
        (0.0, -10.0, "bottom edge"),
        (5.0, 0.0, "inside horizontally"),
        (0.0, 5.0, "inside vertically"),
    ],
)
def test_point_inside_circle(
    circle: models.CircleTarget,
    x: float,
    y: float,
    description: str,
) -> None:
    """Point at center, on edge, or just inside should be contained."""
    assert circle.contains_point(x, y)


def test_point_outside_with_default_tolerance(circle: models.CircleTarget) -> None:
    """Point outside default tolerance boundary should not be contained."""
    assert not circle.contains_point(16.0, 0.0)
    assert not circle.contains_point(0.0, 16.0)


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
    expected_optimal_distance = (
        end.center_x - start.center_x - start.radius - end.radius
    )
    expected_path_optimality = expected_optimal_distance / segment.distance

    segment.calculate_path_optimality(start, end)

    assert segment.path_optimality == expected_path_optimality


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

    segment.calculate_path_optimality(start, end)

    assert segment.path_optimality == 0.0
