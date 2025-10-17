"""Test cases for the Spiral model."""

import datetime

import pandas as pd
import pytest

from graphomotor.core import models


def test_valid_spiral_creation(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
) -> None:
    """Test creating a valid Spiral instance."""
    spiral = models.Drawing(
        data=valid_spiral_data,
        task_name="spiral_drawing",
        metadata=valid_spiral_metadata,
    )
    assert spiral.data.equals(valid_spiral_data)
    assert spiral.metadata == valid_spiral_metadata
    assert spiral.task_name == "spiral_drawing"


def test_empty_dataframe(
    valid_spiral_metadata: dict[str, str | datetime.datetime],
) -> None:
    """Test validation error when DataFrame is empty."""
    empty_data = pd.DataFrame(
        columns=["line_number", "x", "y", "UTC_Timestamp", "seconds"]
    )

    with pytest.raises(ValueError, match="DataFrame is empty"):
        models.Drawing(data=empty_data, metadata=valid_spiral_metadata)


@pytest.mark.parametrize(
    "key,invalid_value,expected_error",
    [
        ("id", "1001", "'id' must start with digit 5"),
        ("id", "512345", "'id' must be 7 digits long"),
        (
            "hand",
            "left",
            "'hand' must be either 'Dom' or 'NonDom'",
        ),
        (
            "task",
            "rey_o_copy",
            "'task' must be either 'spiral_trace' or 'spiral_recall', numbered 1-5",
        ),
        (
            "task",
            "spiral_trace6",
            "'task' must be either 'spiral_trace' or 'spiral_recall', numbered 1-5",
        ),
    ],
)
def test_invalid_metadata_values(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
    key: str,
    invalid_value: str | datetime.datetime,
    expected_error: str,
) -> None:
    """Test validation errors for various invalid metadata values."""
    invalid_metadata = valid_spiral_metadata.copy()
    invalid_metadata[key] = invalid_value

    with pytest.raises(ValueError, match=expected_error):
        models.Drawing(data=valid_spiral_data, metadata=invalid_metadata)
