"""Utility functions for trails management."""

from typing import Optional, Tuple

import pandas as pd

from graphomotor.core import models


def valid_ink_trajectory(
    points: pd.DataFrame,
    start_circle: models.CircleTarget,
    end_circle: models.CircleTarget,
) -> Tuple[Optional[int], Optional[int]]:
    """Determine whether an ink trajectory exists from a start circle to an end circle.

    An "ink trajectory" is defined as the first contiguous sequence of points that:
      1. Begins **after** the pen leaves the start circle, and
      2. Ends when the pen first enters the end circle.

    The function scans point-by-point in order. The ink start index is the first point
    whose (x, y) location is *outside* the start circle. The ink end index is the first
    subsequent point whose (x, y) location falls *inside* the end circle. If either of
    these conditions never occurs, the trajectory is considered invalid.

    Args:
        points: DataFrame of points with 'x' and 'y' columns.
        start_circle: CircleTarget representing the start circle.
        end_circle: CircleTarget representing the end circle.

    Returns:
        Tuple of (ink_start_idx: int, ink_end_idx: int) if valid trajectory exists,
        else (None, None).
    """
    ink_start_idx = None
    ink_end_idx = None

    for idx in range(len(points)):
        row = points.iloc[idx]

        if (
            not start_circle.contains_point(row["x"], row["y"])
            and ink_start_idx is None
        ):
            ink_start_idx = idx

        if ink_start_idx is not None and end_circle.contains_point(row["x"], row["y"]):
            ink_end_idx = idx
            break

    return ink_start_idx, ink_end_idx
