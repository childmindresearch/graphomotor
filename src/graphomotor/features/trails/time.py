"""Feature extraction module for time-based metrics in trails drawing data."""

from typing import Dict, List, Optional

import pandas as pd

from graphomotor.core import models


def calculate_total_error_time(drawing: models.Drawing) -> dict[str, float]:
    """Calculate the total time spent making errors.

    A contiguous "error chunk" is any sequence of rows where df["error"] != "E0".
    The start and end of each chunk is defined as the midpoint between the last
    timestamp with a "correct" entry and the first timestamp of an "error". The total
    error time is the sum of the durations of all error chunks.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the total time (s) spent in error states.
    """
    mask = drawing.data["error"] != "E0"
    if not mask.any():
        return {"total_error_time": 0.0}

    error_change = mask.astype(int).diff()
    chunk_starts = error_change[error_change == 1].index.tolist()
    chunk_ends = error_change[error_change == -1].index.tolist()

    if mask.iloc[0]:
        chunk_starts = [0] + chunk_starts

    if mask.iloc[-1]:
        chunk_ends = chunk_ends + [len(drawing.data)]

    seconds = drawing.data["seconds"].to_numpy()
    total_error_time = 0.0

    for start_idx, end_idx in zip(chunk_starts, chunk_ends):
        start_time = (
            (seconds[start_idx - 1] + seconds[start_idx]) / 2
            if start_idx > 0
            else seconds[0]
        )

        end_time = (
            (seconds[end_idx - 1] + seconds[end_idx]) / 2
            if end_idx < len(seconds)
            else seconds[-1]
        )

        total_error_time += end_time - start_time

    return {"total_error_time": float(total_error_time)}


def _find_circle_entry_time(
    points: pd.DataFrame, circle: models.CircleTarget
) -> Optional[float]:
    """Find when a point last entered the circle by scanning backwards."""
    return next(
        (
            points.iloc[i]["seconds"]
            for i in range(len(points) - 1, -1, -1)
            if circle.contains_point(points.iloc[i]["x"], points.iloc[i]["y"])
        ),
        None,
    )


def _find_circle_exit_time(
    points: pd.DataFrame, circle: models.CircleTarget
) -> Optional[float]:
    """Find when a point first left the circle by scanning forward."""
    return next(
        (
            points.iloc[i]["seconds"]
            for i in range(len(points))
            if not circle.contains_point(points.iloc[i]["x"], points.iloc[i]["y"])
        ),
        points.iloc[0]["seconds"] if len(points) > 0 else None,
    )


def calculate_think_times(
    segments: List[models.LineSegment],
    circles: Dict[str, Dict[str, models.CircleTarget]],
    config: Dict[str, List[Dict]],
    trail_id: str,
) -> List[models.LineSegment]:
    """Calculate think times using consecutive segments approach.

    Think time at a circle = time from entering the circle (end of incoming segment)
                            to leaving the circle (start of outgoing segment)

    Args:
        segments: List of LineSegment objects in order
        circles: Dictionary mapping trail IDs to dictionaries of CircleTarget
            objects (output of load_scaled_circles)
        config: Configuration dictionary containing circle order for trails (output
            of create_config_from_circles)
        trail_id: Trail identifier for circle lookup

    Returns:
        Updated list of segments with think times calculated
    """
    trail_circles = circles[trail_id]
    circle_order = {item["label"]: item["order"] for item in config[trail_id]["items"]}
    segments = sorted(segments, key=lambda s: circle_order.get(s.start_label, 999))

    for current_seg, next_seg in zip(segments, segments[1:]):
        circle_label = current_seg.end_label
        if circle_label != next_seg.start_label or circle_label not in trail_circles:
            continue

        circle = trail_circles[circle_label]
        current_points = current_seg.points.reset_index(drop=True)
        next_points = next_seg.points.reset_index(drop=True)

        entry_time = _find_circle_entry_time(current_points, circle)
        exit_time = _find_circle_exit_time(next_points, circle)

        if entry_time is not None and exit_time is not None and exit_time > entry_time:
            next_seg.think_time = exit_time - entry_time
            next_seg.think_circle_label = circle_label

    if not segments or len(segments[0].points) == 0:
        return segments

    first_seg = segments[0]
    if first_seg.start_label in trail_circles:
        first_points = first_seg.points.reset_index(drop=True)
        exit_time = _find_circle_exit_time(
            first_points, trail_circles[first_seg.start_label]
        )

        if exit_time is not None:
            first_seg.think_time = exit_time - first_points.iloc[0]["seconds"]
            first_seg.think_circle_label = first_seg.start_label

    return segments
