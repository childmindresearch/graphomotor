"""Internal data class for spiral drawing data."""

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class Spiral(BaseModel):
    """A class representing a spiral drawing, encapsulating both raw data and metadata.

    Attributes:
        data: DataFrame containing drawing data with required columns
            (line_number, x, y, UTC_Timestamp, seconds)
        metadata: Dictionary containing metadata about the spiral:
            - id: Unique identifier for the participant
            - hand: Hand used ('Dom' for dominant, 'NonDom' for non-dominant)
            - task: Task name
            - start_time: Start time of drawing
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    metadata: dict[str, str | datetime]

    @field_validator("data")
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        """Validate that DataFrame contains required columns and correct data types.

        Args:
            cls: The class.
            v: The dataframe to validate.

        Returns:
            The dataframe if it is not empty.

        Raises:
            ValueError: If the dataframe is empty.
        """
        if v.empty:
            raise ValueError("DataFrame is empty")

        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validate metadata dictionary for required keys and correct data types.

        Args:
            cls: The class.
            v: The metadata dictionary to validate.

        Returns:
            The metadata dictionary if it is valid.

        Raises:
            ValueError: If the metadata dictionary has invalid values.
        """
        if not v["id"].startswith("5"):
            raise ValueError("'id' must start with digit 5")
        if len(v["id"]) != 7:
            raise ValueError("'id' must be 7 digits long")

        if v["hand"] not in ["Dom", "NonDom"]:
            raise ValueError("'hand' must be either 'Dom' or 'NonDom'")

        valid_tasks = ["spiral_trace", "spiral_recall"]
        valid_tasks_trials = [
            f"{prefix}{i}" for prefix in valid_tasks for i in range(1, 6)
        ]
        if v["task"] not in valid_tasks_trials:
            raise ValueError(
                "'task' must be either 'spiral_trace' or 'spiral_recall', numbered 1-5"
            )

        return v
