"""Tests for trails_utils.py."""

import typing

import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.utils import trails_utils


@pytest.fixture
def circles() -> typing.Dict[str, typing.Dict[str, models.CircleTarget]]:
    """Return a minimal set of CircleTarget dicts for each trail."""
    return {
        "trail2": {
            "1": models.CircleTarget(
                order=1,
                label="1",
                center_x=0,
                center_y=0,
                radius=10,
            ),
            "2": models.CircleTarget(
                order=2,
                label="2",
                center_x=1,
                center_y=0,
                radius=10,
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
    circles: typing.Dict[str, typing.Dict[str, models.CircleTarget]],
) -> None:
    """Test that segment_lines correctly segments multiple unique paths."""
    df = pd.DataFrame({
        "actual_path": ["1 ~ 2", "2 ~ 1"],
        "line_number": [0, 1],
        "is_error": [False, True],
        "x": [0, 1],
        "y": [0, 1],
    })
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
    circles: typing.Dict[str, typing.Dict[str, models.CircleTarget]],
) -> None:
    """Test that segment_lines handles fallback for line_number for identical paths."""
    df = pd.DataFrame({
        "actual_path": ["1 ~ 2", "1 ~ 2"],
        "line_number": [0, 1],
        "is_error": [False, False],
        "x": [0, 1],
        "y": [0, 1],
    })
    segments = trails_utils.segment_lines(df, "trail2", circles)
    assert len(segments) == 2
    for idx, seg in enumerate(segments):
        assert seg.start_label == "1"
        assert seg.end_label == "2"
        assert seg.line_number == idx
        assert not seg.is_error


def test_empty_dataframe(
    circles: typing.Dict[str, typing.Dict[str, models.CircleTarget]],
) -> None:
    """Test that segment_lines returns an empty list for an empty DataFrame."""
    df = pd.DataFrame(columns=["actual_path", "line_number", "is_error", "x", "y"])
    segments = trails_utils.segment_lines(df, "trail2", circles)
    assert segments == []


def test_invalid_trail_id(
    circles: typing.Dict[str, typing.Dict[str, models.CircleTarget]],
) -> None:
    """Test that segment_lines raises KeyError for invalid trail_id."""
    df = pd.DataFrame({
        "actual_path": ["1 ~ 2"],
        "line_number": [0],
        "is_error": [False],
    })
    with pytest.raises(KeyError, match="Trail ID 'invalid' not found"):
        trails_utils.segment_lines(df, "invalid", circles)


@pytest.mark.parametrize(
    "df_data, match_msg",
    [
        (
            {
                "actual_path": ["invalid"],
                "line_number": [0],
                "is_error": [False],
                "x": [0],
                "y": [0],
            },
            "Invalid path value 'invalid' encountered",
        ),
        (
            {
                "actual_path": [None],
                "line_number": [0],
                "is_error": [False],
                "x": [0],
                "y": [0],
            },
            "Invalid path value 'None' encountered",
        ),
    ],
)
def test_invalid_paths_raise(
    df_data: pd.DataFrame,
    match_msg: str,
    circles: typing.Dict[str, typing.Dict[str, models.CircleTarget]],
) -> None:
    """Test that segment_lines raises ValueError for invalid path values."""
    df = pd.DataFrame(df_data)
    with pytest.raises(ValueError, match=match_msg):
        trails_utils.segment_lines(df, "trail2", circles)
