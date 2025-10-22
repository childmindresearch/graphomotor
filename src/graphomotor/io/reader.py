"""Reader module for processing spiral drawing CSV files."""

import datetime
import pathlib
import re

import pandas as pd

from graphomotor.core import models

DTYPE_MAP = {
    "line_number": "int",
    "x": "float",
    "y": "float",
    "UTC_Timestamp": "float",
    "seconds": "float",
    "epoch_time_in_seconds_start": "float",
    "error": "str",
    "correct_path": "str",
    "actual_path": "str",
    "total_time": "float",
    "total_number_of_errors": "int",
}


def _parse_filename(filename: str) -> dict[str, str | datetime.datetime]:
    """Extract metadata from spiral drawing filename.

    The function parses filenames of Curious exports of drawing data that are
    typically formatted as '[5123456]curious-ID-spiral_trace2_NonDom'. It extracts
    the participant ID (the value within the brackets), task name ('spiral_trace' or
    'spiral_recall', followed by the trial number from 1 to 5), and hand used (dominant
    or non-dominant). Regular expressions are used to match the expected pattern
    and extract the relevant components.

    Note: A 'start_time' key (datetime object) will be added to the returned dictionary
    later in the load_spiral function.

    Args:
        filename: Filename of the spiral drawing CSV file from Curious export.

    Returns:
        Dictionary containing extracted metadata:
            - id: Participant ID (e.g., '5123456')
            - hand: Hand used for drawing ('Dom' or 'NonDom')
            - task: Task name and trial number (e.g., 'spiral_trace2')

    Raises:
        ValueError: If filename does not match expected pattern.
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

    raise ValueError(f"Filename does not match expected pattern: {filename}")


def _convert_start_time(data: pd.DataFrame) -> datetime.datetime:
    """Convert start time to a datetime object.

    Args:
        data: DataFrame containing spiral drawing data.

    Returns:
        Start time as a datetime object in UTC.

    Raises:
        ValueError: If there is an issue with the data format or conversion.
    """
    try:
        start_time = datetime.datetime.fromtimestamp(
            data["epoch_time_in_seconds_start"].iloc[0], tz=datetime.timezone.utc
        )
        return start_time
    except Exception as e:
        raise ValueError(f"Error converting 'start_time' to datetime: {e}")


def load_drawing_data(filepath: pathlib.Path | str) -> models.Drawing:
    """Load a single drawing CSV file and return a Drawing object.

    This function loads data from a pre-processed/cleaned CSV file containing
    drawing data. The loaded data is assumed to already have unique timestamps and
    uniform sampling, so no further validation is performed for these aspects. The
    function extracts metadata from the filename using the _parse_filename function.

    Args:
        filepath: Path to the CSV file containing  drawing data.

    Returns:
        A Drawing object containing the loaded data and metadata.

    Raises:
        IOError: If the file cannot be read.
    """
    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    try:
        data = pd.read_csv(filepath)

        if "UTC_Timestamp" in data.columns:
            data["UTC_Timestamp"] = pd.to_datetime(
                data["UTC_Timestamp"].astype(float) * 1000,
                unit="ms",
                utc=True,
                exact=True,
            )

        for col, dtype in DTYPE_MAP.items():
            if col in data.columns and col != "UTC_Timestamp":
                data[col] = data[col].astype(dtype)
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")

    metadata = _parse_filename(filepath.stem)

    metadata["start_time"] = _convert_start_time(data)
    metadata["source_path"] = str(filepath)

    data = data.drop(columns=["epoch_time_in_seconds_start"])
    task_name = metadata["task"]
    return models.Drawing(data=data, task_name=task_name, metadata=metadata)
