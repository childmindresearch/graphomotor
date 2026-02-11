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
