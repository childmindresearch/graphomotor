"""Tests for trails time.py."""

import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.features.trails import time


def test_total_error_time_no_errors() -> None:
    """Test case with no errors."""
    df = pd.DataFrame(
        {
            "error": ["E0", "E0", "E0", "E0"],
            "seconds": [0, 1, 2, 3],
        }
    )
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})

    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 0.0}


def test_single_error_chunk() -> None:
    """Test case with a single error chunk."""
    df = pd.DataFrame(
        {
            "error": ["E0", "E1", "E1", "E0", "E0"],
            "seconds": [0.0, 1.0, 3.0, 5.0, 6.0],
        }
    )
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})

    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 3.5}


def test_multiple_error_chunks() -> None:
    """Test case with multiple error chunks."""
    df = pd.DataFrame(
        {
            "error": ["E0", "E1", "E1", "E0", "E2", "E2", "E0"],
            "seconds": [0, 1, 3, 5, 6, 7, 9],
        }
    )
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 6.0}


def test_error_at_end() -> None:
    """Test case with an error chunk that goes to the end of the drawing."""
    df = pd.DataFrame(
        {
            "error": ["E0", "E0", "E2", "E2"],
            "seconds": [0.0, 1.0, 2.0, 4.0],
        }
    )
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 2.5}


def test_error_at_start() -> None:
    """Test case with an error chunk that starts at the beginning of the drawing."""
    df = pd.DataFrame(
        {
            "error": ["E1", "E1", "E0", "E0"],
            "seconds": [0.0, 1.0, 3.0, 4.0],
        }
    )
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 2.0}


def _make_segment(
    start_label: str,
    end_label: str,
    points: list[dict],
    is_error: bool = False,
    line_number: int = 0,
) -> models.LineSegment:
    """Helper to construct a LineSegment with a points DataFrame."""
    return models.LineSegment(
        start_label=start_label,
        end_label=end_label,
        points=pd.DataFrame(points),
        is_error=is_error,
        line_number=line_number,
    )


def _make_circle(
    label: str,
    center_x: float,
    center_y: float,
    radius: float,
    order: int = 0,
) -> models.CircleTarget:
    """Helper to construct a CircleTarget."""
    return models.CircleTarget(
        order=order,
        label=label,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )


@pytest.fixture
def create_segment_and_circles() -> tuple[
    list[models.LineSegment], list[models.CircleTarget]
]:
    """Fixture to create a sample LineSegment and CircleTarget for testing."""
    first_segment = _make_segment(
        start_label="A",
        end_label="B",
        points=[
            {"x": 0.0, "y": 0.0, "seconds": 1.0},
            {"x": 0.1, "y": 0.0, "seconds": 1.1},
            {"x": 5.0, "y": 5.0, "seconds": 2.0},
            {"x": 10.0, "y": 10.0, "seconds": 3.0},
            {"x": 10.1, "y": 10.1, "seconds": 3.05},
        ],
    )
    second_segment = _make_segment(
        start_label="B",
        end_label="C",
        points=[
            {"x": 10.0, "y": 10.0, "seconds": 3.5},
            {"x": 10.1, "y": 10.1, "seconds": 3.55},
            {"x": 15.0, "y": 15.0, "seconds": 4.5},
            {"x": 20.0, "y": 20.0, "seconds": 5.5},
            {"x": 20.1, "y": 20.1, "seconds": 5.55},
        ],
    )

    third_segment = _make_segment(
        start_label="C",
        end_label="D",
        points=[
            {"x": 20.0, "y": 20.0, "seconds": 6.0},
            {"x": 20.1, "y": 20.1, "seconds": 6.05},
            {"x": 22.0, "y": 22.0, "seconds": 6.5},
            {"x": 25.0, "y": 25.0, "seconds": 7.0},
        ],
    )
    first_circle = _make_circle(label="A", center_x=0.0, center_y=0.0, radius=1.0)
    second_circle = _make_circle(label="B", center_x=10.0, center_y=10.0, radius=1.0)
    third_circle = _make_circle(label="C", center_x=20.0, center_y=20.0, radius=1.0)
    fourth_circle = _make_circle(label="D", center_x=25.0, center_y=25.0, radius=1.0)
    return [first_segment, second_segment, third_segment], [
        first_circle,
        second_circle,
        third_circle,
        fourth_circle,
    ]


@pytest.mark.parametrize(
    "circle_number, segment_number, expected_entry_time",
    [
        (0, 0, 1.0),
        (1, 0, 3.0),
        (1, 1, 3.5),
        (2, 1, 5.5),
    ],  # converting A B C labels to 0, 1, 2 index for list
)
def test_entry_time(
    circle_number: int,
    segment_number: int,
    expected_entry_time: float,
    create_segment_and_circles: tuple[
        list[models.LineSegment], list[models.CircleTarget]
    ],
) -> None:
    """Test entry time when entering a circle."""
    segments, circles = create_segment_and_circles

    entry_time = time._find_circle_entry_time(
        segments[segment_number].points, circles[circle_number]
    )
    assert entry_time == expected_entry_time


@pytest.mark.parametrize(
    "circle_number, segment_number, expected_exit_time",
    [
        (0, 0, 2.0),
        (1, 0, 1.0),
        (1, 1, 4.5),
        (2, 1, 3.5),
    ],  # converting A B C labels to 0, 1, 2 index for list
)
def test_exit_time(
    circle_number: int,
    segment_number: int,
    expected_exit_time: float,
    create_segment_and_circles: tuple[
        list[models.LineSegment], list[models.CircleTarget]
    ],
) -> None:
    """Test exit time when exiting a circle."""
    segments, circles = create_segment_and_circles

    exit_time = time._find_circle_exit_time(
        segments[segment_number].points, circles[circle_number]
    )
    assert exit_time == expected_exit_time


def test_calculate_think_times(
    create_segment_and_circles: tuple[
        list[models.LineSegment], list[models.CircleTarget]
    ],
) -> None:
    """Test the calculate_think_times function."""
    segments, circles = create_segment_and_circles
    circle_mapping = {"5555555": {circle.label: circle for circle in circles}}

    time.calculate_think_times(segments, circle_mapping, "5555555")

    assert segments[0].think_time == 1.0
    assert segments[0].think_circle_label == "A"
    assert segments[1].think_time == 1.5
    assert segments[1].think_circle_label == "B"
    assert segments[2].think_time == 1.0
    assert segments[2].think_circle_label == "C"
