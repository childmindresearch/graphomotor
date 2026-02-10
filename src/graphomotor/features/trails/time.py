"""Feature extraction module for time-based metrics in trails drawing data."""

from graphomotor.core import models


def calculate_total_error_time(drawing: models.Drawing) -> dict[str, float]:
    """Calculate the total time spent making errors.

    A contiguous "error chunk" is any sequence of rows where df["error"] != "E0".
    The start and end of each chunk is defined as the midpoint between the last
    timestamp with a "correct" entry and the first timestamp of an "error". The total
    error time is the sum of the durations of all error chunks.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the total time (s) spent in error states.
    """
    mask = drawing.data["error"] != "E0"
    if not mask.any():
        return {"total_error_time": 0.0}

    error_change = mask.astype(int).diff()
    chunk_starts = error_change[error_change == 1].index.tolist()
    chunk_ends = error_change[error_change == -1].index.tolist()

    if mask.iloc[0]:
        chunk_starts = [0] + chunk_starts

    if mask.iloc[-1]:
        chunk_ends = chunk_ends + [len(drawing.data)]

    seconds = drawing.data["seconds"].to_numpy()
    total_error_time = 0.0

    for start_idx, end_idx in zip(chunk_starts, chunk_ends):
        start_time = (
            (seconds[start_idx - 1] + seconds[start_idx]) / 2
            if start_idx > 0
            else seconds[0]
        )

        end_time = (
            (seconds[end_idx - 1] + seconds[end_idx]) / 2
            if end_idx < len(seconds)
            else seconds[-1]
        )

        total_error_time += end_time - start_time

    return {"total_error_time": float(total_error_time)}
