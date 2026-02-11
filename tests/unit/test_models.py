"""Test cases for the Spiral model."""

import datetime
from typing import Dict, cast

import numpy as np
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


@pytest.mark.parametrize(
    "points_data,start_params,end_params,expected_start,expected_end,test_id",
    [
        (
            {"x": [0, 1, 2, 3, 4, 5], "y": [0, 1, 2, 3, 4, 5]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 0.5},
            {"order": 2, "label": "B", "center_x": 5, "center_y": 5, "radius": 0.5},
            1,
            5,
            "valid_trajectory",
        ),
        (
            {"x": [0.1, 0.2, 0.3], "y": [0.1, 0.2, 0.3]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 1.0},
            {"order": 2, "label": "B", "center_x": 10, "center_y": 10, "radius": 1.0},
            None,
            None,
            "no_exit_from_start",
        ),
        (
            {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 0.5},
            {"order": 2, "label": "B", "center_x": 10, "center_y": 10, "radius": 0.5},
            1,
            None,
            "never_reaches_end",
        ),
        (
            {"x": [], "y": []},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 1.0},
            {"order": 2, "label": "B", "center_x": 5, "center_y": 5, "radius": 1.0},
            None,
            None,
            "empty_dataframe",
        ),
        (
            {"x": [0.1], "y": [0.1]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 1.0},
            {"order": 2, "label": "B", "center_x": 5, "center_y": 5, "radius": 1.0},
            None,
            None,
            "single_point_in_start",
        ),
        (
            {"x": [2], "y": [2]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 0.5},
            {"order": 2, "label": "B", "center_x": 5, "center_y": 5, "radius": 0.5},
            0,
            None,
            "single_point_outside_start",
        ),
        (
            {"x": [0, 2.5, 5], "y": [0, 2.5, 5]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 0.5},
            {"order": 2, "label": "B", "center_x": 2.5, "center_y": 2.5, "radius": 1.0},
            1,
            1,
            "immediate_transition",
        ),
        (
            {"x": [3, 4, 5], "y": [3, 4, 5]},
            {"order": 1, "label": "A", "center_x": 0, "center_y": 0, "radius": 0.5},
            {"order": 2, "label": "B", "center_x": 5, "center_y": 5, "radius": 0.5},
            0,
            2,
            "first_point_outside_start",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_valid_ink_trajectory(
    points_data: Dict[str, list[float]],
    start_params: dict[str, int | str | float],
    end_params: dict[str, int | str | float],
    expected_start: int | None,
    expected_end: int | None,
    test_id: str,
) -> None:
    """Test valid_ink_trajectory method with various point configurations.

    Tests behavior with different circle boundaries.
    """
    points_df = pd.DataFrame(points_data)

    start_circle = models.CircleTarget(**start_params)  # type: ignore[arg-type]
    end_circle = models.CircleTarget(**end_params)  # type: ignore[arg-type]

    line_segment = models.LineSegment(
        start_label=cast(str, start_params["label"]),
        end_label=cast(str, end_params["label"]),
        points=points_df,
        is_error=False,
        line_number=1,
    )
    result_start, result_end = line_segment.valid_ink_trajectory(
        start_circle, end_circle
    )

    assert result_start == expected_start, f"Start index mismatch for {test_id}"
    assert result_end == expected_end, f"End index mismatch for {test_id}"


def test_uniform_motion() -> None:
    """Test with points moving at constant velocity."""
    points = pd.DataFrame({
        "x": [0, 1, 2, 3],
        "y": [0, 0, 0, 0],
        "seconds": [0, 1, 2, 3],
    })
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    segment.calculate_velocity_metrics(points)

    assert segment.distance == 3.0
    assert segment.mean_speed == 1.0
    assert segment.speed_variance == 0.0
    assert np.all(segment.velocities) == 1.0
    assert np.all(segment.accelerations) == 0.0


def test_accelerating_motion() -> None:
    """Test with motion accelerating over time."""
    points = pd.DataFrame({
        "x": [0, 1, 4, 9],
        "y": [0, 0, 0, 0],
        "seconds": [0, 1, 2, 3],
    })
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    segment.calculate_velocity_metrics(points)

    assert segment.distance == 9.0
    assert segment.mean_speed == 3.0
    assert segment.speed_variance > 0.0
    assert segment.velocities == [1.0, 3.0, 5.0]
    assert segment.accelerations == [2.0, 2.0]


def test_velocity_two_points_only() -> None:
    """Test velocity calculation with only two points."""
    points = pd.DataFrame({
        "x": [0, 3],
        "y": [0, 4],
        "seconds": [0, 2],
    })
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    segment.calculate_velocity_metrics(points)

    assert segment.distance == 5.0
    assert segment.mean_speed == 2.5
    assert segment.speed_variance == 0.0
    assert segment.velocities == [2.5]
    assert segment.accelerations == []


def test_decelerating_motion() -> None:
    """Test with decelerating motion (negative acceleration)."""
    points = pd.DataFrame({
        "x": [0, 4, 7, 9],
        "y": [0, 0, 0, 0],
        "seconds": [0, 1, 2, 3],
    })
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    segment.calculate_velocity_metrics(points)

    assert segment.distance == 9.0
    assert segment.mean_speed == 3.0
    assert segment.speed_variance > 0.0
    assert segment.velocities == [4.0, 3.0, 2.0]
    assert segment.accelerations == [-1.0, -1.0]


def test_stationary_motion() -> None:
    """Test with no movement (all points the same)."""
    points = pd.DataFrame({
        "x": [1, 1, 1],
        "y": [1, 1, 1],
        "seconds": [0, 1, 2],
    })
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    segment.calculate_velocity_metrics(points)

    assert segment.distance == 0.0
    assert segment.mean_speed == 0.0
    assert segment.speed_variance == 0.0
    assert segment.velocities == [0.0, 0.0]
    assert segment.accelerations == [0.0]
