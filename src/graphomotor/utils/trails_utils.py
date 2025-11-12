"""Utility functions for trails management."""

import typing

import pandas as pd

from graphomotor.core import models


def segment_lines(
    df: pd.DataFrame,
    trail_id: str,
    circles: typing.Dict[str, typing.Dict[str, models.CircleTarget]],
) -> typing.List[models.LineSegment]:
    """Segment data into individual lines drawn between circles.

    This function first tries to segment lines based on unique actual paths.
    If multiple unique paths are found, it creates segments for each path.
    If only one unique path exists, it falls back to grouping by line numbers.

    Args:
        df: Participant data DataFrame
        trail_id: 'trail2' or 'trail4'
        circles: Dictionary mapping trail IDs to lists of CircleTarget objects

    Returns:
        List of LineSegment objects
    """
    if trail_id not in circles:
        raise KeyError("Trail ID not found in circles dictionary.")

    segments = []
    unique_paths = df["actual_path"].unique()

    if len(unique_paths) > 1:
        segment_counter = 0
        for path in unique_paths:
            if pd.isna(path) or "~" not in path:
                raise ValueError("Invalid actual_path value encountered.")

            path_data = df[df["actual_path"] == path].copy()
            start_label, end_label = path.split(" ~ ")
            segments.append(
                models.LineSegment(
                    start_label=start_label.strip(),
                    end_label=end_label.strip(),
                    points=path_data,
                    is_error=path_data["is_error"].iloc[0],
                    line_number=segment_counter,
                )
            )
            segment_counter += 1
    else:
        for line_num in df["line_number"].unique():
            line_data = df[df["line_number"] == line_num].copy()
            path = line_data["actual_path"].iloc[0]
            if pd.isna(path) or "~" not in path:
                raise ValueError("Invalid actual_path value encountered.")

            start_label, end_label = path.split(" ~ ")
            segments.append(
                models.LineSegment(
                    start_label=start_label.strip(),
                    end_label=end_label.strip(),
                    points=line_data,
                    is_error=line_data["is_error"].iloc[0],
                    line_number=int(line_num),
                )
            )

    return segments
