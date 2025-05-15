"""Tests for main workflow of graphomotor orchestrator."""

import pathlib

import pytest

from graphomotor.core import orchestrator


def test_orchestrator_happy_path(
    sample_data: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the orchestrator with a happy path scenario."""
    features = orchestrator.run_pipeline(
        input_path=sample_data,
        output_path=None,
        feature_categories=["duration", "velocity", "hausdorff", "AUC"],
        config_params=None,
    )

    assert "Graphomotor pipeline completed successfully" in caplog.text
    assert isinstance(features, dict)
    assert len(features) == 25
