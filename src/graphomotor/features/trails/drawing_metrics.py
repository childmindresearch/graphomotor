"""Feature extraction module for drawing error-based metrics in spiral drawing data."""

from graphomotor.core import models


def detect_pen_lifts(drawing: models.Drawing) -> int:
    """Detect pen lifts during a spiral drawing task.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Integer count of pen lifts detected.
    """
    return len(drawing.data["line_number"].unique()) - 1
