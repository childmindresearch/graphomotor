"""Tests for trails_utils.py."""

from typing import Any, Dict, List

import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.utils import trails_utils


@pytest.fixture
def circles() -> Dict[str, Dict[str, models.CircleTarget]]:
    """Return a minimal set of CircleTarget dictionaries for each trail."""
    return {
        "trail2": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=1, center_y=0, radius=10
            ),
        },
        "trail4": {
            "1": models.CircleTarget(
                order=1, label="1", center_x=0, center_y=0, radius=10
            ),
            "2": models.CircleTarget(
                order=2, label="2", center_x=1, center_y=0, radius=10
            ),
        },
    }


def test_multiple_unique_paths(
    circles: Dict[str, Dict[str, models.CircleTarget]],
) -> None:
    """Multiple unique actual_path values produce correct segments."""
    df = pd.DataFrame(
        {
            "actual_path": ["1 ~ 2", "2 ~ 1"],
            "line_number": [0, 1],
            "is_error": [False, True],
            "x": [0, 1],
            "y": [0, 1],
        }
    )
    segments = trails_utils.segment_lines(df, "trail2", circles)
    assert len(segments) == 2
    assert segments[0].start_label == "1"
    assert segments[0].end_label == "2"
    assert not segments[0].is_error
    assert segments[0].line_number == 0
    assert segments[1].start_label == "2"
    assert segments[1].end_label == "1"
    assert segments[1].is_error
    assert segments[1].line_number == 1


def test_single_path_fallback_line_number(
    circles: Dict[str, Dict[str, models.CircleTarget]],
) -> None:
    """Single unique path falls back to line_number segmentation."""
    df = pd.DataFrame(
        {
            "actual_path": ["1 ~ 2", "1 ~ 2"],
            "line_number": [0, 1],
            "is_error": [False, False],
            "x": [0, 1],
            "y": [0, 1],
        }
    )
    segments = trails_utils.segment_lines(df, "trail2", circles)
    assert len(segments) == 2
    for idx, seg in enumerate(segments):
        assert seg.start_label == "1"
        assert seg.end_label == "2"
        assert seg.line_number == idx
        assert not seg.is_error


def test_empty_dataframe(
    circles: Dict[str, Dict[str, models.CircleTarget]],
) -> None:
    """Empty DataFrame produces no segments."""
    df = pd.DataFrame(columns=["actual_path", "line_number", "is_error", "x", "y"])
    segments = trails_utils.segment_lines(df, "trail2", circles)
    assert segments == []


def test_invalid_trail_id(
    circles: Dict[str, Dict[str, models.CircleTarget]],
) -> None:
    """Passing an invalid trail_id raises KeyError."""
    df = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "actual_path": ["1 ~ 2"],
            "line_number": [0],
            "is_error": [False],
        }
    )
    with pytest.raises(KeyError, match="Trail ID not found in circles dictionary."):
        trails_utils.segment_lines(df, "invalid", circles)


@pytest.mark.parametrize(
    "data, match_msg",
    [
        (
            {
                "actual_path": ["invalid", "newpath"],
                "line_number": [0, 1],
                "is_error": [False, True],
                "x": [0, 1],
                "y": [0, 1],
            },
            "Invalid actual_path value encountered",
        ),
        (
            {
                "actual_path": [None],
                "line_number": [0],
                "is_error": [False],
                "x": [0],
                "y": [0],
            },
            "Invalid actual_path value encountered",
        ),
    ],
)
def test_invalid_paths_raise(
    data: Dict[str, List[Any]],
    match_msg: str,
    circles: Dict[str, Dict[str, models.CircleTarget]],
) -> None:
    """Invalid actual_path values trigger ValueError."""
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match=match_msg):
        trails_utils.segment_lines(df, "trail2", circles)
