"""Fixtures used by pytest."""

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import generate_reference_spiral


@pytest.fixture
def sample_spiral_data() -> pathlib.Path:
    """Sample spiral data for tests."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / (
            "[5000000]65318bf53c36ce79135b1049-648c7d0e8819c1120b4f708d"
            "-spiral_trace1_Dom.csv"
        )
    )


@pytest.fixture
def sample_trails_data() -> pathlib.Path:
    """Sample trails data for tests."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / ("[5000000]648b6b868819c1120b4f6ce3-trail4.csv")
    )


@pytest.fixture
def sample_alpha_data() -> pathlib.Path:
    """Sample alphabet data for tests."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / (
            "[5999999]17745689-f78c-4513-b052-ea9e3daad2f0-648c7b45-8819-c112-0b4f-6f3200000000-Alpha_AtoZ.csv"
        )
    )


@pytest.fixture
def sample_dsym_data() -> pathlib.Path:
    """Sample DSYM data for tests."""
    return (
        pathlib.Path(__file__).parent
        / "sample_data"
        / ("[5000000]64d3ba8e22d8180cf9b407b5-648c7d0e8819c1120b4f708d-dsym_2.csv")
    )


@pytest.fixture
def valid_spiral_data(sample_spiral_data: pathlib.Path) -> pd.DataFrame:
    """Create a valid DataFrame for spiral data."""
    return pd.read_csv(sample_spiral_data)


@pytest.fixture
def valid_spiral_metadata() -> dict[str, str | datetime.datetime]:
    """Create valid metadata for spiral."""
    return {
        "id": "5000000",
        "hand": "Dom",
        "task": "spiral_trace1",
        "start_time": datetime.datetime.fromtimestamp(
            1697745697.08,
            tz=datetime.timezone.utc,
        ),
        "source_path": "sample_data/[5000000]"
        "65318bf53c36ce79135b1049-648c7d0e8819c1120b4f708d-spiral_trace1_Dom.csv",
    }


@pytest.fixture
def valid_spiral(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict[str, str | datetime.datetime],
) -> models.Drawing:
    """Create a valid Spiral object."""
    return models.Drawing(
        data=valid_spiral_data,
        task_name="spiral",
        metadata=valid_spiral_metadata,
    )


@pytest.fixture
def ref_spiral() -> np.ndarray:
    """Create a reference spiral for testing."""
    return generate_reference_spiral.generate_reference_spiral(config.SpiralConfig())


@pytest.fixture
def perfect_spiral() -> models.Drawing:
    """Create a perfect Spiral object."""
    return reader.load_drawing_data(
        pathlib.Path(__file__).parent
        / "sample_data"
        / "[5000000]perfect-3000-points-spiral_trace1_Dom.csv"
    )


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create a sample features DataFrame for testing."""
    return pd.DataFrame(
        {
            "participant_id": ["5123456"],
            "task": ["spiral_trace1"],
            "hand": ["Dom"],
            "test_feature": [1.0],
        }
    )


@pytest.fixture
def sample_batch_features() -> pathlib.Path:
    """Path to the sample batch features CSV file."""
    return pathlib.Path(__file__).parent / "sample_data" / "sample_batch_features.csv"


@pytest.fixture
def sample_batch_features_df(sample_batch_features: pathlib.Path) -> pd.DataFrame:
    """Create a sample batch features DataFrame for testing."""
    return pd.read_csv(sample_batch_features)


@pytest.fixture
def sample_batch_spirals() -> pathlib.Path:
    """Path to the sample batch spirals directory."""
    return pathlib.Path(__file__).parent / "sample_batch_data"
