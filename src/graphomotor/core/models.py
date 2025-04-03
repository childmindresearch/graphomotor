"""Internal data class for spiral drawing data."""

import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class Spiral(BaseModel):
    """A class representing a spiral drawing, encapsulating both raw data and metadata.

    Attributes:
        data (pd.DataFrame): DataFrame containing drawing data with required columns
            (line_number, x, y, UTC_Timestamp, seconds)
        metadata (dict): Dictionary containing metadata about the spiral:
            - id: Unique identifier for the participant
            - hand: Hand used ('Dom' for dominant, 'NonDom' for non-dominant)
            - task: Task name
            - start_time: Start time of drawing
    """

    data: pd.DataFrame
    metadata: dict[str, str | datetime.datetime]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data")
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        """Validate that DataFrame contains required columns and correct data types."""
        required_columns = ["line_number", "x", "y", "UTC_Timestamp", "seconds"]
        missing_columns = [col for col in required_columns if col not in v.columns]

        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        if v.empty:
            raise ValueError("DataFrame is empty")

        if not pd.api.types.is_integer_dtype(v["line_number"]):
            raise ValueError("'line_number' should be of type int")
        if not pd.api.types.is_float_dtype(v["x"]):
            raise ValueError("'x' should be of type float")
        if not pd.api.types.is_float_dtype(v["y"]):
            raise ValueError("'y' should be of type float")
        if not pd.api.types.is_float_dtype(v["seconds"]):
            raise ValueError("'seconds' should be of type float")
        if not pd.api.types.is_datetime64_any_dtype(v["UTC_Timestamp"]):
            raise ValueError("'UTC_Timestamp' should be of type datetime")

        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validate metadata dictionary for required keys and correct data types."""
        required_keys = ["id", "hand", "task", "start_time"]
        missing_keys = [key for key in required_keys if key not in v]

        if missing_keys:
            raise ValueError(f"Metadata missing required keys: {missing_keys}")

        if not isinstance(v["id"], str):
            raise ValueError("'id' must be a string")
        if not v["id"].startswith("5"):
            raise ValueError("'id' must start with digit 5")
        if len(v["id"]) != 7:
            raise ValueError("'id' must be 7 digits long")

        if not isinstance(v["hand"], str):
            raise ValueError("'hand' must be a string")
        if v["hand"] not in ["Dom", "NonDom"]:
            raise ValueError(
                "'hand' must be either 'Dom' (dominant) or 'NonDom' (non-dominant)"
            )

        if not isinstance(v["task"], str):
            raise ValueError("'task' must be a string")
        valid_prefixes = ["spiral_trace", "spiral_recall"]
        valid_tasks = [f"{prefix}{i}" for prefix in valid_prefixes for i in range(1, 6)]
        if v["task"] not in valid_tasks:
            raise ValueError(
                "'task' must be either 'spiral_trace' or 'spiral_recall', numbered 1-5"
            )

        if not isinstance(v["start_time"], (datetime.datetime)):
            raise ValueError("'start_time' must be a datetime object")

        return v
