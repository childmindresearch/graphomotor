"""Feature extraction module for metrics used across all task drawing data."""

from graphomotor.core import models


def get_task_duration(drawing: models.Drawing) -> dict[str, float]:
    """Calculate the total duration of a drawing task.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the total duration of the task in seconds.
    """
    return {"duration": drawing.data["seconds"].iloc[-1]}


def detect_pen_lifts(drawing: models.Drawing) -> dict[str, int]:
    """Detect pen lifts during a spiral drawing task.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Integer count of pen lifts detected.
    """
    return {"pen_lifts": len(drawing.data["line_number"].unique()) - 1}
