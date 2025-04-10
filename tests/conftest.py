"""Fixtures used by pytest."""

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

from graphomotor.core import config, models


@pytest.fixture
def sample_data() -> pathlib.Path:
    """Sample data for tests."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / "[5123456]d3afad5c-8a5d-4292-8f54-24109ea6f793-648c7b3e-8819-c112-0b4f-6f3000000000-spiral_trace1_Dom.csv"  # noqa: E501
    )


@pytest.fixture
def valid_spiral_data(sample_data: pathlib.Path) -> pd.DataFrame:
    """Create a valid DataFrame for spiral data."""
    return pd.read_csv(sample_data)


@pytest.fixture
def valid_spiral_metadata() -> dict[str, str | datetime.datetime]:
    """Create valid metadata for spiral."""
    return {
        "id": "5123456",
        "hand": "Dom",
        "task": "spiral_trace1",
        "start_time": datetime.datetime.fromtimestamp(
            1701700376.296,
            tz=datetime.timezone.utc,
        ),
    }


@pytest.fixture
def valid_spiral(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
) -> models.Spiral:
    """Create a valid Spiral object."""
    return models.Spiral(
        data=valid_spiral_data,
        metadata=valid_spiral_metadata,
    )


@pytest.fixture
def reference_spiral() -> np.ndarray:
    """Create a reference spiral for testing."""
    return config.generate_reference_spiral()
