"""Reader module for processing spiral drawing CSV files."""

import datetime
import re
from pathlib import Path

import pandas as pd

from graphomotor.core import models

DATA_COLUMNS = [
    "line_number",
    "x",
    "y",
    "UTC_Timestamp",
    "seconds",
    "epoch_time_in_seconds_start",
]


def parse_filename(filename: str) -> dict[str, str | datetime.datetime]:
    """Extract metadata from spiral drawing filename.

    Args:
        filename: The filename to parse, typically in format like
                 '[5123456]mindlogger-ID-spiral_trace2_NonDom' or similar pattern.

    Returns:
        dict: Dictionary containing extracted metadata such as:
             - id: Unique identifier for the participant
             - hand: Hand used ('Dom' for dominant, 'NonDom' for non-dominant)
             - task: Task name

    Raises:
        ValueError: If filename does not match expected pattern or cannot be parsed.
    """
    pattern = r"\[(\d+)\].*-([^_]+)_([^_]+)_(\w+)$"
    match = re.match(pattern, filename)

    if match:
        id, task_name, task_detail, hand = match.groups()
        metadata = {
            "id": id,
            "hand": hand,
            "task": f"{task_name}_{task_detail}",
        }
        return metadata
    else:
        raise ValueError(f"Filename does not match expected pattern: {filename}")


def load_spiral(filepath: Path) -> models.Spiral:
    """Load a single spiral drawing CSV file and return a Spiral object.

    Args:
        filepath: Path to the CSV file containing spiral drawing data.

    Returns:
        Spiral: A Spiral object containing the loaded data and metadata.

    Raises:
        IOError: If the file cannot be read.
        KeyError: If the file does not contain required columns.
        ValueError: If the file has incorrect data types.
    """
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")

    missing_columns = [col for col in DATA_COLUMNS if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if not pd.api.types.is_integer_dtype(data["line_number"]):
        raise ValueError("'line_number' should be of type int")
    if not pd.api.types.is_float_dtype(data["x"]):
        raise ValueError("'x' should be of type float")
    if not pd.api.types.is_float_dtype(data["y"]):
        raise ValueError("'y' should be of type float")
    if not pd.api.types.is_float_dtype(data["UTC_Timestamp"]):
        raise ValueError("'UTC_Timestamp' should be of type float")
    if not pd.api.types.is_float_dtype(data["seconds"]):
        raise ValueError("'seconds' should be of type float")

    try:
        data["UTC_Timestamp"] = pd.to_datetime(
            data["UTC_Timestamp"] * 1000, unit="ms", utc=True
        )
    except Exception as e:
        raise ValueError(f"Error converting 'UTC_Timestamp' to datetime: {e}")

    metadata = parse_filename(filepath.stem)

    try:
        metadata["start_time"] = datetime.datetime.fromtimestamp(
            data["epoch_time_in_seconds_start"].iloc[0], tz=datetime.timezone.utc
        )
    except Exception as e:
        raise ValueError(f"Error converting 'start_time' to datetime: {e}")

    data = data.drop(columns=["epoch_time_in_seconds_start"])

    return models.Spiral(data=data, metadata=metadata)
