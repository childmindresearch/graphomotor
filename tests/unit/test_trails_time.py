"""Tests for trails time.py."""

from typing import Any, Dict, List, Tuple

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
    points: List[Dict],
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


@pytest.mark.parametrize(
    "segments_data, circles_data, config_data, expected",
    [
        pytest.param(
            [
                (
                    "A",
                    "B",
                    [
                        {"x": 20.0, "y": 20.0, "seconds": 1.0},
                        {"x": 30.0, "y": 30.0, "seconds": 2.0},
                    ],
                )
            ],
            {"A": ("A", 0.0, 0.0, 1.0)},
            [{"label": "A", "order": 1}],
            [],
            id="single_segment_no_intermediate_think_time",
        ),
        pytest.param(
            [
                (
                    "A",
                    "B",
                    [
                        {"x": 5.0, "y": 5.0, "seconds": 1.0},
                        {"x": 5.0, "y": 5.0, "seconds": 2.0},
                    ],
                ),
                (
                    "B",
                    "C",
                    [
                        {"x": 5.0, "y": 5.0, "seconds": 3.0},
                        {"x": 20.0, "y": 20.0, "seconds": 5.0},
                    ],
                ),
            ],
            {"B": ("B", 5.0, 5.0, 1.0)},
            [{"label": "A", "order": 1}, {"label": "B", "order": 2}],
            [("B", "C", 3.0, "B")],
            id="think_time_assigned_between_consecutive_segments",
        ),
        pytest.param(
            [
                ("A", "B", [{"x": 5.0, "y": 5.0, "seconds": 1.0}]),
                ("C", "D", [{"x": 20.0, "y": 20.0, "seconds": 3.0}]),
            ],
            {"B": ("B", 5.0, 5.0, 1.0)},
            [{"label": "A", "order": 1}, {"label": "C", "order": 2}],
            [],
            id="no_think_time_when_segments_not_connected",
        ),
        pytest.param(
            [
                ("A", "B", [{"x": 5.0, "y": 5.0, "seconds": 1.0}]),
                ("B", "C", [{"x": 20.0, "y": 20.0, "seconds": 3.0}]),
            ],
            {},
            [{"label": "A", "order": 1}, {"label": "B", "order": 2}],
            [],
            id="no_think_time_when_circle_label_missing",
        ),
        pytest.param(
            [
                ("A", "B", [{"x": 5.0, "y": 5.0, "seconds": 5.0}]),
                ("B", "C", [{"x": 5.0, "y": 5.0, "seconds": 3.0}]),
            ],
            {"B": ("B", 5.0, 5.0, 1.0)},
            [{"label": "A", "order": 1}, {"label": "B", "order": 2}],
            [],
            id="no_think_time_when_exit_not_after_entry",
        ),
    ],
)
def test_intermediate_think_times(
    segments_data: List[Tuple],
    circles_data: Dict[str, Tuple],
    config_data: List[Dict],
    expected: List[Tuple],
) -> None:
    """Test think time assignment between consecutive segment pairs."""
    segments = [_make_segment(s, e, p) for s, e, p in segments_data]
    circles = {
        "trail1": {
            label: _make_circle(*params) for label, params in circles_data.items()
        }
    }
    config = {"trail1": {"items": config_data}}

    result = time.calculate_think_times(segments, circles, config, "trail1")

    expected_by_key = {(s, e): (tt, lbl) for s, e, tt, lbl in expected}
    for seg in result:
        key = (seg.start_label, seg.end_label)
        if key in expected_by_key:
            think_time, circle_label = expected_by_key[key]
            assert seg.think_time == think_time
            assert seg.think_circle_label == circle_label
        else:
            assert seg.think_time == 0.0


def test_first_segment_think_time() -> None:
    """Test think time is assigned to the first segment based on its start circle."""
    seg = _make_segment(
        "A",
        "B",
        [
            {"x": 0.0, "y": 0.0, "seconds": 1.0},
            {"x": 0.0, "y": 0.0, "seconds": 2.0},
            {"x": 20.0, "y": 20.0, "seconds": 4.0},
        ],
    )
    circles = {"trail1": {"A": _make_circle("A", 0.0, 0.0, 1.0)}}
    config = {"trail1": {"items": [{"label": "A", "order": 1}]}}

    result = time.calculate_think_times([seg], circles, config, "trail1")

    assert result[0].think_time == 3.0
    assert result[0].think_circle_label == "A"


def test_segment_sorting() -> None:
    """Test that segments are sorted correctly by circle order."""
    segments = [
        _make_segment("B", "C", [{"x": 5.0, "y": 5.0, "seconds": 2.0}]),
        _make_segment("A", "B", [{"x": 0.0, "y": 0.0, "seconds": 1.0}]),
    ]
    circles: Dict[str, Dict[str, models.CircleTarget]] = {"trail1": {}}
    config: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "trail1": {"items": [{"label": "A", "order": 1}, {"label": "B", "order": 2}]}
    }

    result = time.calculate_think_times(segments, circles, config, "trail1")

    assert [seg.start_label for seg in result] == ["A", "B"]


def test_empty_segments_returns_empty() -> None:
    """Empty segment list returns empty list without error."""
    result = time.calculate_think_times(
        [],
        {"trail1": {}},
        {"trail1": {"items": []}},
        "trail1",
    )
    assert result == []
