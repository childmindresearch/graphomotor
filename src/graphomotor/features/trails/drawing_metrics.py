"""Feature extraction module for drawing error-based metrics in trails drawing data."""

import numpy as np
import pandas as pd

from graphomotor.core import models


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


def percent_accurate_paths(drawing: models.Drawing) -> dict[str, float]:
    """Calculate the percentage of accurate paths in a trails drawing task.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the percentage of accurate paths of the task.

    Raises:
        ValueError: If required columns are missing in the drawing data.
    """
    if not {"correct_path", "actual_path"}.issubset(drawing.data.columns):
        raise ValueError(
            "DataFrame must contain 'correct_path' and 'actual_path' columns."
        )
    return {
        "percent_accurate_paths": (
            (drawing.data["correct_path"] == drawing.data["actual_path"]).mean() * 100
        )
    }


def calculate_smoothness(points: pd.DataFrame) -> float:
    """Calculate path smoothness based on Root Mean Square (RMS) curvature.

    Represents the curvature per unit arc length.
    Lower values indicate smoother drawings. Penalizes sharp corners (e.g., 90° turns)
    and noisy corrections. Normalized by arc length to reduce sampling-rate dependence.

    Args:
        points: DataFrame representing drawing points.

    Returns:
        Smoothness metric as a float.
    """
    if len(points) < 3:
        return 0.0

    xy = points[["x", "y"]].to_numpy()

    forward_vector = xy[1:-1] - xy[:-2]
    backward_vector = xy[2:] - xy[1:-1]

    forward_norm = np.linalg.norm(forward_vector, axis=1)
    backward_norm = np.linalg.norm(backward_vector, axis=1)

    valid = (forward_norm > 0) & (backward_norm > 0)
    if not np.any(valid):
        return 0.0

    valid_forward_vector = forward_vector[valid]
    valid_backward_vector = backward_vector[valid]
    valid_forward_norm = forward_norm[valid]
    valid_backward_norm = backward_norm[valid]

    cos_angle = (valid_forward_vector * valid_backward_vector).sum(axis=1) / (
        valid_forward_norm * valid_backward_norm
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angles = np.arccos(cos_angle)

    avg_segment_length = (valid_forward_norm + valid_backward_norm) / 2.0
    curvatures = angles / avg_segment_length

    return float(np.sqrt(np.mean(curvatures**2)))
