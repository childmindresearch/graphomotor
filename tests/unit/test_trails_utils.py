"""Tests for trails_utils.py."""

import typing

import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.utils import trails_utils


@pytest.fixture
def circles() -> typing.Dict[str, typing.List[models.CircleTarget]]:
    """Return a minimal set of CircleTarget lists for each trail."""
    return {
        "trail2": [
            models.CircleTarget(order=1, label="1", center_x=0, center_y=0, radius=10),
            models.CircleTarget(order=2, label="2", center_x=1, center_y=0, radius=10),
        ],
        "trail4": [
            models.CircleTarget(order=1, label="1", center_x=0, center_y=0, radius=10),
            models.CircleTarget(order=2, label="2", center_x=1, center_y=0, radius=10),
        ],
    }


def test_multiple_unique_paths(
    circles: typing.Dict[str, typing.List[models.CircleTarget]],
) -> None:
    """Test segmentation when df has multiple unique actual_path values."""
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
    assert segments[0].is_error is False
    assert segments[0].line_number == 0

    assert segments[1].start_label == "2"
    assert segments[1].end_label == "1"
    assert segments[1].is_error is True
    assert segments[1].line_number == 1


def test_single_path_fallback_line_number(
    circles: typing.Dict[str, typing.List[models.CircleTarget]],
) -> None:
    """Test fallback segmentation when only one unique actual_path exists."""
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
        assert seg.is_error is False


def test_empty_dataframe(
    circles: typing.Dict[str, typing.List[models.CircleTarget]],
) -> None:
    """Empty DataFrame should produce no segments."""
    df = pd.DataFrame(columns=["actual_path", "line_number", "is_error", "x", "y"])
    segments = trails_utils.segment_lines(df, "trail2", circles)
    assert segments == []


def test_invalid_paths_skipped(
    circles: typing.Dict[str, typing.List[models.CircleTarget]],
) -> None:
    """Rows with invalid paths should be skipped."""
    df = pd.DataFrame({
        "actual_path": ["invalid", None, "1 ~ 2"],
        "line_number": [0, 1, 2],
        "is_error": [False, True, False],
        "x": [0, 1, 2],
        "y": [0, 1, 2],
    })

    segments = trails_utils.segment_lines(df, "trail2", circles)

    # Only the valid "1 ~ 2" path should be converted to a segment
    assert len(segments) == 1
    seg = segments[0]
    assert seg.start_label == "1"
    assert seg.end_label == "2"
    assert seg.line_number == 0
    assert seg.is_error is False
