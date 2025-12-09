"""Feature extraction module for drawing error-based metrics in trails drawing data."""

import scipy.spatial.distance as dist

from graphomotor.core import models


def detect_pen_lifts(drawing: models.Drawing) -> dict[str, int]:
    """Detect pen lifts during a spiral drawing task.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Integer count of pen lifts detected.
    """
    return {"pen_lifts": len(drawing.data["line_number"].unique()) - 1}


def get_total_errors(drawing: models.Drawing) -> dict[str, float]:
    """Extract the total number of errors of a trails drawing task.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the total number of errors of the task.
    """
    if "total_number_of_errors" not in drawing.data.columns:
        raise ValueError(
            "Drawing data does not contain 'total_number_of_errors' column."
        )
    return {"total_errors": drawing.data["total_number_of_errors"].iloc[0]}


def calculate_path_optimality(
    segment: models.LineSegment,
    start_circle: models.CircleTarget,
    end_circle: models.CircleTarget,
) -> None:
    """Calculate path optimality ratio.

    Args:
        segment: LineSegment object for which to calculate path optimality.
        start_circle: CircleTarget representing the start circle.
        end_circle: CircleTarget representing the end circle.

    Returns:
        Path optimality ratio.
    """
    optimal_distance = (
        dist.euclidean(
            [start_circle.center_x, start_circle.center_y],
            [end_circle.center_x, end_circle.center_y],
        )
        - start_circle.radius
        - end_circle.radius
    )

    if optimal_distance > 0:
        segment.path_optimality = optimal_distance / segment.distance
    return
