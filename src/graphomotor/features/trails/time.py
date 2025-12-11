"""Feature extraction module for time-based metrics in trails drawing data."""

from graphomotor.core import models


def calculate_total_error_time(drawing: models.Drawing) -> dict[str, float]:
    """Calculate the total time spent making errors.

    A contiguous "error chunk" is any sequence of rows where df["error"] != "E0".
    For each chunk, we find the midpoint time when the error started and the midpoint
    time when the error ended. The total error time is the sum of the durations of all
    error chunks.

    Args:
        drawing: Drawing object containing drawing data.

    Returns:
        Dictionary containing the total time spent in error states.
    """
    mask = drawing.data["error"] != "E0"
    if not mask.any():
        return {"total_error_time": 0.0}

    chunk_start = (~mask.shift(fill_value=False) & mask).to_numpy().nonzero()[0]
    chunk_end = (mask.shift(fill_value=False) & ~mask).to_numpy().nonzero()[0]

    if mask.iloc[-1]:
        chunk_end = list(chunk_end) + [len(drawing.data) - 1]

    seconds = drawing.data["seconds"].to_numpy()
    total_error_time = 0.0

    for start_idx, end_idx in zip(chunk_start, chunk_end):
        start_mid = (
            (seconds[start_idx - 1] + seconds[start_idx]) / 2
            if start_idx > 0
            else seconds[0]
        )

        if end_idx + 1 < len(drawing.data):
            end_mid = (seconds[end_idx] + seconds[end_idx - 1]) / 2
        else:
            if mask.iloc[end_idx]:
                end_mid = seconds[end_idx]
            else:
                end_mid = (seconds[end_idx] + seconds[end_idx - 1]) / 2
        print(start_mid, end_mid)
        total_error_time += end_mid - start_mid

    return {"total_error_time": float(total_error_time)}
