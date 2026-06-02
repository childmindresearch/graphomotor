"""Feature extraction module for time-based metrics in trails drawing data."""

from typing import Optional

import pandas as pd

from graphomotor.core import config, models

logger = config.get_logger()


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


def calculate_think_times(
    segments: list[models.LineSegment],
    circle_mapping: dict[str, dict[str, models.CircleTarget]],
    trail_id: str,
) -> None:
    """Calculate think times using consecutive segments approach.

    This function computes all the think_times for all provided LineSegments.
    `think_time` is defined as the difference in timestamps between the first point of
    the current LineSegment entering a circle and the first point of the next
    LineSegment exiting that same circle.

    For the first segment, if it starts inside a circle, the `think_time` is calculated
    as the difference between the first point of the segment and the first point exiting
    that circle.

    This method is only for segments without errors.

    Args:
        segments: List of LineSegment objects in order.
        circle_mapping: Dictionary mapping trail IDs to dictionaries of CircleTarget
            objects (output of load_scaled_circles, provides circle locations).
        trail_id: Specific trail task identifier for circle location lookup.
    """
    target_circles = circle_mapping[trail_id]

    first_seg = segments[0]
    if first_seg.start_label in target_circles:
        exit_time = _find_circle_exit_time(
            first_seg.points, target_circles[first_seg.start_label]
        )

        if exit_time is not None:
            first_seg.think_time = exit_time - first_seg.points.iloc[0]["seconds"]
            first_seg.think_circle_label = first_seg.start_label

    for current_seg, next_seg in zip(segments, segments[1:]):
        current_circle_label = current_seg.end_label
        if current_circle_label != next_seg.start_label:
            logger.warning(
                "Mismatched segment labels: %s -> %s and %s -> %s",
                current_seg.start_label,
                current_seg.end_label,
                next_seg.start_label,
                next_seg.end_label,
            )
            continue

        if current_circle_label not in target_circles:
            logger.warning(
                "Circle label %s not found in trail circles", current_circle_label
            )
            continue

        circle_location = target_circles[current_circle_label]

        entry_time = _find_circle_entry_time(current_seg.points, circle_location)
        exit_time = _find_circle_exit_time(next_seg.points, circle_location)

        if entry_time is not None and exit_time is not None and exit_time > entry_time:
            next_seg.think_time = exit_time - entry_time
            next_seg.think_circle_label = current_circle_label

        # Provide logger warning for this strange case
        if entry_time is not None and exit_time is not None and entry_time > exit_time:
            logger.warning(
                "Entry time %s is greater than exit time %s for circle %s",
                entry_time,
                exit_time,
                current_circle_label,
            )


def _find_circle_entry_time(
    points: pd.DataFrame, circle: models.CircleTarget
) -> Optional[float]:
    """Helper function to find the timestamp when a LineSegment first entered a circle.

    This function iterates through the points in order and returns the
    timestamp of the first point that is inside the circle. If no points are inside
    the circle, it returns None.

    Args:
        points: DataFrame containing the points of a LineSegment.
        circle: CircleTarget object representing the specific circle to check against.

    Returns:
        The timestamp (in seconds) of the first point that is inside the circle,
        or None if no points are inside the circle.
    """
    for row in range(len(points)):
        if circle.contains_point(points.iloc[row]["x"], points.iloc[row]["y"]):
            return points.iloc[row]["seconds"]
    return None


def _find_circle_exit_time(
    points: pd.DataFrame, circle: models.CircleTarget
) -> Optional[float]:
    """Helper function to find the timestamp when a LineSegment first exits a circle.

    This function iterates through the points in order and returns the
    timestamp of the first point that is outside the circle. If all points are inside
    the circle, it returns None.

    Args:
        points: DataFrame containing the points of a LineSegment.
        circle: CircleTarget object representing the specific circle to check against.

    Returns:
        The timestamp (in seconds) of the first point that is outside the circle,
        or None if all points are inside the circle.
    """
    for row in range(len(points)):
        if not circle.contains_point(points.iloc[row]["x"], points.iloc[row]["y"]):
            return points.iloc[row]["seconds"]
    return None
