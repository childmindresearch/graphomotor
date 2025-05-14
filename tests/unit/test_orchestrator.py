"""Tests for the orchestrator module."""

import pathlib

import pytest

from graphomotor.core import orchestrator

## TODO: test all functions here, then add a smoke test as well for the whole pipeline


@pytest.mark.parametrize(
    "feature_categories, expected_behavior",
    [
        (["duration", "velocity", "hausdorff", "AUC"], "multiple_valid"),
        (["duration"], "single_valid"),
        (["duration", "unknown_category"], "contains_invalid"),
        (["unknown_category"], "only_invalid"),
    ],
)
def test_extract_features_categories(
    feature_categories: list[str],
    expected_behavior: str,
    sample_data: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the feature extraction process with various categories."""
    if expected_behavior == "only_invalid":
        with pytest.raises(ValueError, match="No valid feature categories provided"):
            orchestrator.extract_features(
                input_path=sample_data,
                output_path=None,
                feature_categories=feature_categories,
            )
    else:
        features = orchestrator.extract_features(
            input_path=sample_data,
            output_path=None,
            feature_categories=feature_categories,
        )

        if expected_behavior == "multiple_valid":
            assert len(features) == 25
        elif expected_behavior == "single_valid":
            assert len(features) == 1 and "duration" in features
        elif expected_behavior == "contains_invalid":
            assert (
                "Unknown feature categories requested, these categories will be ignored"
                in caplog.text
            )
            assert len(features) > 0
