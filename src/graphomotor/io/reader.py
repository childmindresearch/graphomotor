"""Reader module for processing spiral drawing CSV files."""

import datetime
import pathlib
import re

import pandas as pd

from graphomotor.core import models
from graphomotor.io import io_utils

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
    """Extract metadata from drawing filename.

    The function parses filenames of Curious exports of drawing data that are
    typically formatted as:
    - Spiral: '[5123456]curious-ID-spiral_trace1_Dom'
    - Trail: '[5012543]648b6b868819c1120b4f6ce3-trail4'
    - Alphabet: '[5902334]uuid-uuid-Alpha_AtoZ'
    - DSYM: '[5086403]curious-ID-dsym_2'

    It extracts the participant ID (the value within the brackets) and task name
    (everything after the last dash) from the filename.

    Note: A 'start_time' key (datetime object) will be added to the returned dictionary
    later in the load_drawing function.

    Args:
        filename: Filename of the drawing CSV file from Curious export.

    Returns:
        Dictionary containing extracted metadata:
            - id: 7-digit Participant ID ('5123456')
            - task: Task name ('spiral_trace1_Dom', 'trail4', 'Alpha_AtoZ', 'dsym_2')

    Raises:
        ValueError: If filename does not match expected pattern.
    """
    pattern = r"\[(\d+)\].*-(.+)$"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    participant_id, task_name = match.groups()

    metadata = {
        "id": participant_id,
        "task": task_name,
    }

    return metadata


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

    With trails data there are often empty values in certain columns, so we handle
    missing values when applying data types (empty string, 0.0 for numerical).

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
                if dtype == "int":
                    data[col] = data[col].fillna(0).astype(dtype)
                elif dtype == "float":
                    data[col] = data[col].fillna(0.0).astype(dtype)
                elif dtype == "str":
                    data[col] = data[col].fillna("").astype(dtype)
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")

    metadata = _parse_filename(filepath.stem)
    io_utils._check_missing_columns(data, metadata["task"])
    metadata["start_time"] = _convert_start_time(data)
    metadata["source_path"] = str(filepath)

    data = data.drop(columns=["epoch_time_in_seconds_start"])

    return models.Drawing(data=data, task_name=str(metadata["task"]), metadata=metadata)
