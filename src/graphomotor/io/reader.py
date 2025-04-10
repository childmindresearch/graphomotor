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
}


def _parse_filename(filename: str) -> dict[str, str | datetime.datetime]:
    """Extract metadata from spiral drawing filename.

    The function parses filenames of MindLogger exports of drawing data that are
    typically formatted as '[5123456]mindlogger-ID-spiral_trace2_NonDom'. It extracts
    the participant ID (the value within the brackets), task name ('spiral_trace' or
    'spiral_recall', followed by the trial number from 1 to 5), and hand used (dominant
    or non-dominant). Regular expressions are used to match the expected pattern
    and extract the relevant components.

    Note: A 'start_time' key (datetime object) will be added to the returned dictionary
    later in the load_spiral function.

    Args:
        filename: Filename of the spiral drawing CSV file from MindLogger export.

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


def _check_missing_columns(data: pd.DataFrame) -> None:
    """Check for missing columns in the DataFrame.

    Args:
        data: DataFrame containing spiral drawing data.

    Raises:
        KeyError: If any required columns are missing.
    """
    missing_columns = set(DTYPE_MAP.keys()) - set(data.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")


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


def load_spiral(filepath: pathlib.Path | str) -> models.Spiral:
    """Load a single spiral drawing CSV file and return a Spiral object.

    This function loads data from a pre-processed/cleaned CSV file containing spiral
    drawing data. The loaded data is assumed to already have unique timestamps and
    uniform sampling, so no further validation is performed for these aspects. The
    function extracts metadata from the filename using the _parse_filename function.

    Args:
        filepath: Path to the CSV file containing spiral drawing data.

    Returns:
        A Spiral object containing the loaded data and metadata.

    Raises:
        IOError: If the file cannot be read.
    """
    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)

    to_datetime_converter = {
        "UTC_Timestamp": lambda x: pd.to_datetime(
            float(x) * 1000, unit="ms", utc=True, exact=True
        ),
    }
    accepted_dtypes = {k: v for k, v in DTYPE_MAP.items() if k != "UTC_Timestamp"}

    try:
        data = pd.read_csv(
            filepath,
            dtype=accepted_dtypes,
            converters=to_datetime_converter,
            usecols=list(DTYPE_MAP.keys()),
        )
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")

    _check_missing_columns(data)

    metadata = _parse_filename(filepath.stem)

    metadata["start_time"] = _convert_start_time(data)

    data = data.drop(columns=["epoch_time_in_seconds_start"])

    return models.Spiral(data=data, metadata=metadata)
