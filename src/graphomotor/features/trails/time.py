"""Feature extraction module for time-based metrics in trails drawing data."""

from graphomotor.core import models


def calculate_total_error_time(drawing: models.Drawing) -> dict[str, float]:
    """Calculate the total time spent making errors.

    A contiguous "error chunk" is any sequence of rows where df["error"] != "E0".
    For each chunk, we take:
        chunk_time = seconds[end_index] - seconds[start_index]
    and sum these across all detected error chunks.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the total time spent in error states.
    """
    mask = drawing.data["error"] != "E0"

    if not mask.any():
        return {"total_error_time": 0.0}

    shifted = mask.shift(fill_value=False)

    chunk_start = (~shifted & mask).to_numpy().nonzero()[0]
    chunk_end = (shifted & ~mask).to_numpy().nonzero()[0]

    if mask.iloc[-1]:
        chunk_end = list(chunk_end) + [len(drawing.data) - 1]

    total_error_time = 0.0

    for start_idx, end_idx in zip(chunk_start, chunk_end):
        start_time = drawing.data.loc[start_idx, "seconds"]
        end_time = drawing.data.loc[end_idx, "seconds"]
        total_error_time += end_time - start_time

    return {"total_error_time": float(total_error_time)}
