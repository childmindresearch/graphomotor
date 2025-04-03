"""Test cases for the Spiral model."""

import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from graphomotor.core.models import Spiral


@pytest.fixture
def valid_spiral_data() -> pd.DataFrame:
    """Create a valid DataFrame for spiral data."""
    return pd.DataFrame(
        {
            "line_number": [0, 0, 1, 1, 2],
            "x": [
                51.91775653923541,
                51.991637323943664,
                52.07809356136821,
                52.87191901408451,
                52.87191901408451,
            ],
            "y": [
                50.268840214136375,
                49.935427997673536,
                49.58943418813663,
                49.516041561871226,
                49.448939732142854,
            ],
            "UTC_Timestamp": pd.to_datetime(
                [
                    1706896115.972,
                    1706896115.977,
                    1706896115.978,
                    1706896115.991,
                    1706896116.016,
                ],
                unit="s",
                utc=True,
            ),
            "seconds": [0, 0.005, 0.006, 0.019, 0.044],
        }
    )


@pytest.fixture
def valid_spiral_metadata() -> dict[str, str | datetime.datetime]:
    """Create valid metadata for spiral."""
    return {
        "id": "5123456",
        "hand": "Dom",
        "task": "spiral_trace2",
        "start_time": datetime.datetime.fromtimestamp(
            1706896115.972, tz=datetime.timezone.utc
        ),
    }


class TestSpiralModel:
    """Test cases for the Spiral model."""

    def test_valid_spiral_creation(
        self,
        valid_spiral_data: pd.DataFrame,
        valid_spiral_metadata: dict[str, str | datetime.datetime],
    ) -> None:
        """Test creating a valid Spiral instance."""
        spiral = Spiral(data=valid_spiral_data, metadata=valid_spiral_metadata)
        assert spiral.data.equals(valid_spiral_data)
        assert spiral.metadata == valid_spiral_metadata

    def test_missing_columns(
        self,
        valid_spiral_data: pd.DataFrame,
        valid_spiral_metadata: dict[str, str | datetime.datetime],
    ) -> None:
        """Test validation error when required columns are missing."""
        for column in valid_spiral_data.columns:
            missing_column_data = valid_spiral_data.drop(columns=[column])
            with pytest.raises(ValidationError) as excinfo:
                Spiral(data=missing_column_data, metadata=valid_spiral_metadata)
            assert f"DataFrame missing required columns: ['{column}']" in str(
                excinfo.value
            )

    def test_empty_dataframe(
        self, valid_spiral_metadata: dict[str, str | datetime.datetime]
    ) -> None:
        """Test validation error when DataFrame is empty."""
        empty_data = pd.DataFrame(
            columns=["line_number", "x", "y", "UTC_Timestamp", "seconds"]
        )

        with pytest.raises(ValidationError) as excinfo:
            Spiral(data=empty_data, metadata=valid_spiral_metadata)
        assert "DataFrame is empty" in str(excinfo.value)

    @pytest.mark.parametrize(
        "column,expected_type",
        [
            ("line_number", "int"),
            ("x", "float"),
            ("y", "float"),
            ("UTC_Timestamp", "datetime"),
            ("seconds", "float"),
        ],
    )
    def test_invalid_data_types(
        self,
        valid_spiral_data: pd.DataFrame,
        valid_spiral_metadata: dict[str, str | datetime.datetime],
        column: str,
        expected_type: str,
    ) -> None:
        """Test validation error when data types are incorrect."""
        invalid_data = valid_spiral_data.copy()
        invalid_data[column] = invalid_data[column].astype(str)

        with pytest.raises(ValidationError) as excinfo:
            Spiral(data=invalid_data, metadata=valid_spiral_metadata)

        assert f"'{column}' should be of type {expected_type}" in str(excinfo.value)

    def test_missing_metadata_keys(
        self,
        valid_spiral_data: pd.DataFrame,
        valid_spiral_metadata: dict[str, str | datetime.datetime],
    ) -> None:
        """Test validation error when required metadata keys are missing."""
        for key in list(valid_spiral_metadata.keys()):
            missing_column_metadata = valid_spiral_metadata.copy()
            del missing_column_metadata[key]
            with pytest.raises(ValidationError) as excinfo:
                Spiral(data=valid_spiral_data, metadata=missing_column_metadata)
            assert f"Metadata missing required keys: ['{key}']" in str(excinfo.value)

    @pytest.mark.parametrize(
        "key,invalid_value,expected_error",
        [
            ("id", 5123456, "'id' must be a string"),
            ("id", "1001", "'id' must start with digit 5"),
            ("id", "512345", "'id' must be 7 digits long"),
            (
                "hand",
                1,
                "'hand' must be a string",
            ),
            (
                "hand",
                "left",
                "'hand' must be either 'Dom' (dominant) or 'NonDom' (non-dominant)",
            ),
            (
                "task",
                1,
                "'task' must be a string",
            ),
            (
                "task",
                "rey_o_copy",
                "'task' must be either 'spiral_trace' or 'spiral_recall', numbered 1-5",
            ),
            (
                "task",
                "spiral_trace_6",
                "'task' must be either 'spiral_trace' or 'spiral_recall', numbered 1-5",
            ),
            (
                "start_time",
                "2025-04-03 13:36:20.521",
                "'start_time' must be a datetime object",
            ),
        ],
    )
    def test_invalid_metadata_values(
        self,
        valid_spiral_data: pd.DataFrame,
        valid_spiral_metadata: dict[str, str | datetime.datetime],
        key: str,
        invalid_value: str | datetime.datetime,
        expected_error: str,
    ) -> None:
        """Test validation errors for various invalid metadata values."""
        invalid_metadata = valid_spiral_metadata.copy()
        invalid_metadata[key] = invalid_value

        with pytest.raises(ValidationError) as excinfo:
            Spiral(data=valid_spiral_data, metadata=invalid_metadata)
        assert expected_error in str(excinfo.value)
