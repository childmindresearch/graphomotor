"""Test cases for the Spiral model."""

import datetime
from typing import Dict, cast
from unittest.mock import patch

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
    assert "path_optimality" in segment.available_metrics


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
    assert "path_optimality" not in segment.available_metrics


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
    points = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [0, 0, 0, 0],
            "seconds": [0, 1, 2, 3],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()

    assert segment.distance == 3.0
    assert segment.mean_speed == 1.0
    assert segment.speed_variance == 0.0
    assert segment.velocities == [1.0, 1.0, 1.0]
    assert segment.min_velocity == 1.0
    assert segment.max_velocity == 1.0
    assert segment.accelerations == [0.0, 0.0]
    assert {"distance", "mean_speed", "speed_variance"} <= segment.available_metrics


def test_accelerating_motion() -> None:
    """Test with motion accelerating over time."""
    points = pd.DataFrame(
        {
            "x": [0, 1, 4, 9],
            "y": [0, 0, 0, 0],
            "seconds": [0, 1, 2, 3],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()

    assert segment.distance == 9.0
    assert segment.mean_speed == 3.0
    assert segment.speed_variance > 0.0
    assert segment.velocities == [1.0, 3.0, 5.0]
    assert segment.min_velocity == 1.0
    assert segment.max_velocity == 5.0
    assert {"min_velocity", "max_velocity"} <= segment.available_metrics
    assert segment.accelerations == [2.0, 2.0]


def test_velocity_two_points_only() -> None:
    """Test velocity calculation with only two points."""
    points = pd.DataFrame(
        {
            "x": [0, 3],
            "y": [0, 4],
            "seconds": [0, 2],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()

    assert segment.distance == 5.0
    assert segment.mean_speed == 2.5
    assert segment.speed_variance == 0.0
    assert segment.velocities == [2.5]
    assert segment.min_velocity == 0.0
    assert segment.max_velocity == 0.0
    assert "min_velocity" not in segment.available_metrics
    assert "max_velocity" not in segment.available_metrics
    assert segment.accelerations == []  # No acceleration with only one velocity point
    assert "accelerations" not in segment.available_metrics
    assert "mean_acceleration" not in segment.available_metrics


@pytest.mark.parametrize("seconds", [[0.0, 2.0, 4.0, 6.0], [0.0, 1.0, 3.0, 6.0]])
@pytest.mark.parametrize("decelerating", [False, True])
def test_acceleration_uses_elapsed_time(
    seconds: list[float], decelerating: bool
) -> None:
    """Acceleration uses midpoint time differences, including unequal intervals."""
    timestamps = np.asarray(seconds)
    positions = timestamps**2
    expected = 2.0
    if decelerating:
        positions = 20 * timestamps - positions
        expected = -2.0
    points = pd.DataFrame({"x": positions, "y": 0.0, "seconds": seconds})
    segment = models.LineSegment("1", "2", points, False, 0, ink_points=points)

    segment.calculate_velocity_metrics()

    assert segment.accelerations == pytest.approx([expected, expected])
    assert "accelerations" in segment.available_metrics
    assert segment.mean_acceleration == pytest.approx(expected)
    assert "mean_acceleration" in segment.available_metrics


@pytest.mark.parametrize(
    "positions,expected",
    [
        ([0, 1, 2, 3], 0.0),
        ([0, 0, 0, 0], 0.0),
        ([0, 1, 4], 2.0),
        ([0, 1, 4, 9], 2.0),
        ([0, 4, 7, 9], -1.0),
        ([0, 1, 4, 5], 0.0),
        ([0, 1, 4, 10], 2.5),
    ],
)
def test_mean_acceleration_is_signed_arithmetic_mean(
    positions: list[float], expected: float
) -> None:
    """Mean acceleration retains signs and genuine zeros, including cancellation."""
    points = pd.DataFrame({"x": positions, "y": 0.0, "seconds": range(len(positions))})
    segment = models.LineSegment("1", "2", points, False, 0, ink_points=points)
    segment.calculate_velocity_metrics()

    assert segment.mean_acceleration == pytest.approx(expected)
    assert "mean_acceleration" in segment.available_metrics


def test_decelerating_motion() -> None:
    """Test with decelerating motion (negative acceleration)."""
    points = pd.DataFrame(
        {
            "x": [0, 4, 7, 9],
            "y": [0, 0, 0, 0],
            "seconds": [0, 1, 2, 3],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()

    assert segment.distance == 9.0
    assert segment.mean_speed == 3.0
    assert segment.speed_variance > 0.0
    assert segment.velocities == [4.0, 3.0, 2.0]
    assert segment.accelerations == [-1.0, -1.0]


def test_stationary_motion() -> None:
    """Test with no movement (all points the same)."""
    points = pd.DataFrame(
        {
            "x": [1, 1, 1],
            "y": [1, 1, 1],
            "seconds": [0, 1, 2],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()

    assert segment.distance == 0.0
    assert segment.mean_speed == 0.0
    assert segment.speed_variance == 0.0
    assert segment.velocities == [0.0, 0.0]
    assert segment.min_velocity == 0.0
    assert segment.max_velocity == 0.0
    assert {"min_velocity", "max_velocity"} <= segment.available_metrics
    assert segment.accelerations == [0.0]


def test_no_hesitations_uniform_motion() -> None:
    """Test with uniform motion where all velocities are equal."""
    points = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [0, 0, 0, 0],
            "seconds": [0, 1, 2, 3],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()
    segment.detect_hesitations()

    assert segment.hesitation_count == 0
    assert segment.hesitation_duration == 0.0


def test_hesitation_at_start() -> None:
    """Test when the line starts with a hesitation."""
    points = pd.DataFrame(
        {
            "x": [0, 0.1, 1, 2],
            "y": [0, 0.1, 0, 0],
            "seconds": [0, 1, 2, 3],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()
    segment.detect_hesitations()

    assert segment.hesitation_count == 1
    assert segment.hesitation_duration == 1.0


def test_multiple_hesitations() -> None:
    """Test when there are multiple hesitation periods."""
    points = pd.DataFrame(
        {
            "x": [0, 100, 100.1, 200, 200.1, 300, 400, 500, 600],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "seconds": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()
    segment.detect_hesitations()

    assert segment.hesitation_count == 2
    assert segment.hesitation_duration == 2.0


def test_less_than_three_velocities() -> None:
    """Test early return when velocities length is less than 3."""
    points = pd.DataFrame(
        {
            "x": [0, 1],
            "y": [0, 0],
            "seconds": [0, 1],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for velocity calculation
    )

    segment.calculate_velocity_metrics()
    segment.detect_hesitations()

    assert segment.hesitation_count == 0
    assert segment.hesitation_duration == 0.0


def test_smoothness_less_than_three_points() -> None:
    """Less than 3 points cannot define curvature."""
    points = pd.DataFrame(
        {
            "x": [0, 1],
            "y": [0, 0],
            "seconds": [0, 1],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for smoothness calculation
    )
    segment.calculate_smoothness()
    assert segment.smoothness == 0.0


def test_smoothness_straight_line() -> None:
    """Collinear points have zero curvature."""
    points = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 0, 0, 0]})
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for smoothness calculation
    )
    segment.calculate_smoothness()
    assert segment.smoothness == 0.0


def test_smoothness_single_right_angle() -> None:
    """A single 90-degree corner should produce a large smoothness value.

    (non-zero), since sharp turns are penalized.
    """
    points = pd.DataFrame({"x": [0, 1, 1], "y": [0, 0, 1]})
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for smoothness calculation
    )
    expected = np.pi / 2
    segment.calculate_smoothness()
    assert np.isclose(segment.smoothness, expected)


def test_smoothness_varied_angles() -> None:
    """Multiple angles should produce RMS curvature.

    Path has a 90° turn followed by a 45° turn.
    """
    points = pd.DataFrame({"x": [0, 1, 1, 2], "y": [0, 0, 1, 2]})
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for smoothness calculation
    )
    c1 = np.pi / 2
    c2 = (np.pi / 4) / ((1 + np.sqrt(2)) / 2)
    expected = np.sqrt((c1**2 + c2**2) / 2)

    segment.calculate_smoothness()

    assert np.isclose(segment.smoothness, expected)


def test_smoothness_zero_length_segments() -> None:
    """Zero-length segments should be skipped; no angles → smoothness 0."""
    points = pd.DataFrame({"x": [0, 1, 1, 2], "y": [0, 0, 0, 0]})
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for smoothness calculation
    )
    segment.calculate_smoothness()
    assert segment.smoothness == 0.0


def test_smoothness_single_180_degree_turn() -> None:
    """A single 180-degree turn should produce a very large smoothness value.

    Since it represents maximal curvature.
    """
    points = pd.DataFrame(
        {
            "x": [0, 1, 0],
            "y": [0, 0, 0],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
        ink_points=points,  # Pre-assign ink_points for smoothness calculation
    )
    expected = np.pi
    segment.calculate_smoothness()
    assert np.isclose(segment.smoothness, expected)


def test_compute_segment_metrics_less_than_two_points() -> None:
    """Test early return when segment has less than 2 points."""
    points = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "seconds": [0],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )
    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert segment.ink_time == 0.0
    assert segment.distance == 0.0
    assert len(segment.velocities) == 0


def test_compute_segment_metrics_invalid_start_label() -> None:
    """Test early return when start_label not in trail circles."""
    points = pd.DataFrame(
        {
            "x": [0, 50, 100],
            "y": [0, 0, 0],
            "seconds": [0, 1, 2],
        }
    )
    segment = models.LineSegment(
        start_label="999",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert segment.ink_time == 0.0
    assert segment.distance == 0.0
    assert len(segment.velocities) == 0


def test_compute_segment_metrics_invalid_end_label() -> None:
    """Test early return when end_label not in trail circles."""
    points = pd.DataFrame(
        {
            "x": [0, 50, 100],
            "y": [0, 0, 0],
            "seconds": [0, 1, 2],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="999",
        points=points,
        is_error=False,
        line_number=1,
    )
    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert segment.ink_time == 0.0
    assert segment.distance == 0.0
    assert len(segment.velocities) == 0


def test_compute_segment_metrics_valid_trajectory() -> None:
    """Test successful computation of all metrics with valid trajectory."""
    points = pd.DataFrame(
        {
            "x": [0, 25, 50, 75, 100],
            "y": [0, 0, 0, 0, 0],
            "seconds": [0, 1, 2, 3, 4],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    with patch.object(segment, "valid_ink_trajectory", return_value=(0, 4)):
        segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert np.isclose(segment.ink_time, 4.0)
    assert len(segment.ink_points) == 5
    assert segment.distance > 0.0
    assert segment.mean_speed > 0.0
    assert len(segment.velocities) > 0
    assert segment.path_optimality > 0.0
    assert np.isclose(segment.smoothness, 0.0)


def test_compute_segment_metrics_ink_end_equals_ink_start() -> None:
    """Test when ink_end_idx equals ink_start_idx (no valid trajectory)."""
    points = pd.DataFrame(
        {
            "x": [0, 50, 100],
            "y": [0, 0, 0],
            "seconds": [0, 1, 2],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    with patch.object(segment, "valid_ink_trajectory", return_value=(1, 1)):
        segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert segment.distance == 0.0
    assert len(segment.velocities) == 0


def test_compute_segment_metrics_ink_end_before_ink_start() -> None:
    """Test when ink_end_idx is before ink_start_idx (invalid trajectory)."""
    points = pd.DataFrame(
        {
            "x": [0, 50, 100],
            "y": [0, 0, 0],
            "seconds": [0, 1, 2],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    with patch.object(segment, "valid_ink_trajectory", return_value=(2, 0)):
        segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert segment.distance == 0.0
    assert len(segment.velocities) == 0


def test_compute_segment_metrics_only_start_index_found() -> None:
    """Test when only ink_start_idx is found (end is None)."""
    points = pd.DataFrame(
        {
            "x": [0, 25, 50, 75],
            "y": [0, 0, 0, 0],
            "seconds": [0, 1, 2, 3],
        }
    )
    segment = models.LineSegment(
        start_label="1",
        end_label="2",
        points=points,
        is_error=False,
        line_number=1,
    )

    circles = {
        "A": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=100, center_y=0, radius=10
            ),
        }
    }

    with patch.object(segment, "valid_ink_trajectory", return_value=(1, None)):
        segment.compute_segment_metrics(circles=circles, trail_id="A")

    assert np.isclose(segment.ink_time, 2.0)
    assert len(segment.ink_points) == 3
    assert segment.distance == 0.0
    assert len(segment.velocities) == 0
    assert segment.available_metrics == {"ink_time"}


@pytest.mark.parametrize(
    "positions,expected",
    [
        ([0], set()),
        ([0, 0.5], set()),
        ([0, 2, 4], {"ink_time"}),
        (
            [0, 2, 10],
            {
                "ink_time",
                "distance",
                "mean_speed",
                "speed_variance",
                "path_optimality",
            },
        ),
        (
            [0, 2, 4, 10],
            {
                "ink_time",
                "distance",
                "mean_speed",
                "speed_variance",
                "min_velocity",
                "max_velocity",
                "path_optimality",
                "smoothness",
                "accelerations",
                "mean_acceleration",
            },
        ),
        (
            [0, 2, 4, 6, 10],
            {
                "ink_time",
                "distance",
                "mean_speed",
                "speed_variance",
                "min_velocity",
                "max_velocity",
                "path_optimality",
                "smoothness",
                "hesitation_count",
                "hesitation_duration",
                "accelerations",
                "mean_acceleration",
            },
        ),
    ],
)
def test_segment_metric_availability(
    positions: list[float], expected: set[str]
) -> None:
    """The producer records successful calculations, including calculated zeros."""
    points = pd.DataFrame(
        {"x": positions, "y": [0] * len(positions), "seconds": range(len(positions))}
    )
    segment = models.LineSegment("1", "2", points, False, 0)
    circles = {
        "trail": {
            "1": models.CircleTarget(1, "1", 0, 0, 1),
            "2": models.CircleTarget(2, "2", 10, 0, 1),
        }
    }
    segment.compute_segment_metrics(circles, "trail")
    assert segment.available_metrics == expected
    if "smoothness" in expected:
        assert segment.smoothness == 0

    # A new computation must not publish stale values from an earlier trajectory.
    segment.points = points.iloc[:1]
    segment.compute_segment_metrics(circles, "trail")
    assert not segment.available_metrics


def test_segment_availability_sets_are_independent() -> None:
    """Calculation status is local to each segment, not shared between instances."""
    first = models.LineSegment("1", "2", pd.DataFrame(), False, 0)
    second = models.LineSegment("1", "2", pd.DataFrame(), False, 1)
    first.available_metrics.add("ink_time")
    assert not second.available_metrics
