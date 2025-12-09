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
    """Calculate path smoothness based on curvature changes.

    Lower values indicate smoother paths. This function calculates angles between
    consecutive segments and returns the standard deviation of these angles as a
    measure of smoothness.

    Args:
        points: DataFrame representing drawing points.

    Returns:
        Smoothness metric as a float.
    """
    if len(points) < 3:
        return 0.0

    x = points["x"].values
    y = points["y"].values

    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue

        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)

    if not angles:
        return 0.0

    return np.std(np.degrees(angles))
