"""Tests for main workflow of graphomotor orchestrator."""

import pathlib

import pandas as pd
import pytest

from graphomotor.core import spiral_orchestrator


def test_orchestrator_happy_path(
    sample_spiral_data: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
) -> None:
    """Test the orchestrator with a happy path scenario."""
    output_path = tmp_path / "features.csv"
    features = spiral_orchestrator.run_pipeline(
        input_path=sample_spiral_data,
        output_path=output_path,
        feature_categories=None,
        config_params={"center_x": 0, "center_y": 0},
        verbosity=2,
    )

    assert "Using default feature categories" in caplog.text
    assert "Custom spiral configuration" in caplog.text
    assert "Graphomotor pipeline completed successfully" in caplog.text
    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (1, 29)

    assert pd.read_csv(output_path, index_col=0, dtype=str).equals(features)
