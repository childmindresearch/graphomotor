"""Feature extraction module for drawing error-based metrics in trails drawing data."""

import numpy as np
import pandas as pd

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


def calculate_smoothness(points: pd.DataFrame) -> float:
    """Calculate path smoothness based on RMS curvature.

    Represants the curvature per unit arc length.
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

    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]

    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)

    valid = (l1 > 0) & (l2 > 0)
    if not np.any(valid):
        return 0.0

    v1 = v1[valid]
    v2 = v2[valid]
    l1 = l1[valid]
    l2 = l2[valid]

    cos_angle = np.einsum("ij,ij->i", v1, v2) / (l1 * l2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angles = np.arccos(cos_angle)

    arc_len = (l1 + l2) / 2.0
    curvatures = angles / arc_len

    return float(np.sqrt(np.mean(curvatures**2)))
