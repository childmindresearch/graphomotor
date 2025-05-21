"""Tests for main workflow of graphomotor orchestrator."""

import pathlib

import pytest

from graphomotor.core import orchestrator


def test_orchestrator_happy_path(
    sample_data: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pathlib.Path,
) -> None:
    """Test the orchestrator with a happy path scenario."""
    output_path = tmp_path / "features.csv"
    features = orchestrator.run_pipeline(
        input_path=sample_data,
        output_path=output_path,
        feature_categories=["duration", "velocity", "hausdorff", "AUC"],
        config_params={"center_x": 0, "center_y": 0},
    )
    csv_content = output_path.read_text()

    assert "Custom spiral configuration" in caplog.text
    assert "Features saved successfully to" in caplog.text
    assert "Graphomotor pipeline completed successfully" in caplog.text
    assert "ERROR" not in caplog.text
    assert "WARNING" not in caplog.text

    assert isinstance(features, dict)
    assert len(features) == 25

    assert output_path.is_file()
    assert len(csv_content.strip().split("\n")) == 29
