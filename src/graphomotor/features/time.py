"""Feature extraction module for time-based metrics in spiral drawing data."""

from graphomotor.core import models


def get_task_duration(spiral: models.Spiral) -> dict[str, float]:
    """Calculate the total duration of a spiral drawing task.

    Args:
        spiral: Spiral object containing drawing data.

    Returns:
        Dictionary containing the total duration of the task in seconds.
    """
    return {"duration": spiral.data["seconds"].iloc[-1]}
