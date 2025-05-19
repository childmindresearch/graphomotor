"""Tests for the orchestrator module."""

import pathlib
from datetime import datetime

import numpy as np
import pytest

from graphomotor.core import config, models, orchestrator


@pytest.mark.parametrize(
    "feature_categories, expected_behavior",
    [
        (["duration", "velocity", "hausdorff", "AUC"], "multiple_valid"),
        (["duration"], "single_valid"),
        (["duration", "unknown_category"], "contains_invalid"),
        (["unknown_category"], "only_invalid"),
    ],
)
def test_validate_feature_categories(
    feature_categories: list[str],
    expected_behavior: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _validate_feature_categories with various categories."""
    if expected_behavior == "only_invalid":
        with pytest.raises(ValueError, match="No valid feature categories provided"):
            orchestrator._validate_feature_categories(feature_categories)
    else:
        valid_categories = orchestrator._validate_feature_categories(feature_categories)

        if expected_behavior == "multiple_valid":
            assert len(valid_categories) == 4
        else:
            if expected_behavior == "contains_invalid":
                assert (
                    "Unknown feature categories requested, "
                    "these categories will be ignored" in caplog.text
                )
            assert len(valid_categories) == 1
            assert "duration" in valid_categories


@pytest.mark.parametrize(
    "feature_categories, expected_feature_number",
    [
        (["duration"], 1),
        (["velocity"], 15),
        (["hausdorff"], 8),
        (["AUC"], 1),
        (["duration", "velocity", "hausdorff", "AUC"], 25),
    ],
)
def test_get_feature_categories(
    feature_categories: list[str],
    expected_feature_number: int,
    valid_spiral: models.Spiral,
    ref_spiral: np.ndarray,
) -> None:
    """Test _get_feature_categories with various categories."""
    features = orchestrator._get_feature_categories(
        valid_spiral, ref_spiral, feature_categories
    )

    assert len(features) == expected_feature_number


@pytest.mark.parametrize(
    "output_has_extension, dir_exists, overwrite, raise_exception, feature_values",
    [
        (
            False,
            False,
            False,
            False,
            {"feature1": "0"},
        ),
        (
            True,
            False,
            False,
            False,
            {"feature1": "0", "feature2": "42.0"},
        ),
        (
            False,
            True,
            False,
            False,
            {"feature1": "0", "feature3": "3.14"},
        ),
        (
            True,
            True,
            False,
            False,
            {"feature1": "0", "feature2": "42.0", "feature3": "3.14"},
        ),
        (
            True,
            True,
            True,
            False,
            {"feature1": "0", "feature2": "42.0", "feature3": "3.14", "feature4": "1"},
        ),
        (
            True,
            True,
            True,
            True,
            {"feature1": "0", "feature2": "42.0", "feature3": "3.14", "feature4": "1"},
        ),
    ],
)
def test_export_features_to_csv(
    output_has_extension: bool,
    dir_exists: bool,
    overwrite: bool,
    raise_exception: bool,
    feature_values: dict,
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with various scenarios."""
    input_path = pathlib.Path("/fake/input/path.csv")

    if output_has_extension:
        output_path = tmp_path / "features.csv"
    else:
        output_path = tmp_path / "output_dir"

    if not dir_exists:
        if output_has_extension:
            parent_dir = output_path.parent / "nonexistent"
            output_path = parent_dir / output_path.name
        else:
            output_path = output_path / "nonexistent"

    if overwrite:
        output_path.write_text("This should be overwritten\n")

    expected_filename = (
        f"{valid_spiral.metadata['id']}_{valid_spiral.metadata['task']}_{valid_spiral.metadata['hand']}_"
        f"features_{datetime.today().strftime('%Y%m%d')}.csv"
    )

    expected_output_path = output_path
    if not output_has_extension:
        expected_output_path = output_path / expected_filename

    expected_csv = (
        f"participant_id,{valid_spiral.metadata['id']}\n"
        f"task,{valid_spiral.metadata['task']}\n"
        f"hand,{valid_spiral.metadata['hand']}\n"
        f"source_file,{input_path}\n"
        f"feature1,{feature_values['feature1']}\n"
    )
    if output_has_extension:
        expected_csv += f"feature2,{feature_values['feature2']}\n"
    if dir_exists:
        expected_csv += f"feature3,{feature_values['feature3']}\n"
    if overwrite:
        expected_csv += f"feature4,{feature_values['feature4']}\n"

    if raise_exception:
        output_path.chmod(0o444)
        with pytest.raises(Exception):
            orchestrator._export_features_to_csv(
                valid_spiral,
                feature_values,
                input_path,
                output_path,
            )
    else:
        orchestrator._export_features_to_csv(
            valid_spiral,
            feature_values,
            input_path,
            output_path,
        )

    actual_csv = expected_output_path.read_text()

    if not dir_exists:
        if output_has_extension:
            assert "Creating parent directory that doesn't exist:" in caplog.text
        else:
            assert "Creating directory that doesn't exist:" in caplog.text
    if overwrite and not raise_exception:
        assert "Overwriting existing file:" in caplog.text
        assert "This should be overwritten" not in actual_csv
    if raise_exception:
        assert "Failed to save features to" in caplog.text
    else:
        assert actual_csv == expected_csv


@pytest.mark.parametrize(
    "output_path, spiral_config",
    [
        (None, None),
        ("temp_path", None),
        (
            None,
            config.SpiralConfig.add_custom_params(
                {"center_x": 0, "center_y": 0, "growth_rate": 0}
            ),
        ),
    ],
)
def test_extract_features(
    sample_data: pathlib.Path,
    output_path: pathlib.Path | None,
    spiral_config: config.SpiralConfig | None,
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
) -> None:
    """Test extract_features function with valid categories."""
    if output_path:
        output_path = tmp_path / "features.csv"
    feature_categories = ["duration", "velocity", "hausdorff", "AUC"]
    features = orchestrator.extract_features(
        sample_data, output_path, feature_categories, spiral_config
    )
    expected_max_hausdorff_distance = 0
    if spiral_config:
        expected_max_hausdorff_distance = max(
            np.sqrt(x**2 + y**2)
            for x, y in zip(valid_spiral.data["x"], valid_spiral.data["y"])
        )

    assert isinstance(features, dict)
    assert len(features) == 25
    assert all(isinstance(value, str) for value in features.values())
    assert all(len(value.split(".")[-1]) <= 15 for value in features.values())
    if output_path:
        assert output_path.is_file()
    if spiral_config:
        assert (
            features["hausdorff_distance_maximum"]
            == f"{expected_max_hausdorff_distance:.15f}"
        )
