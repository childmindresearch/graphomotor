"""Tests for the orchestrator module."""

import datetime
import pathlib
import typing

import numpy as np
import pytest

from graphomotor.core import config, models, orchestrator


@pytest.mark.parametrize(
    "feature_categories, expected_valid_count",
    [
        (["duration", "velocity", "hausdorff", "AUC"], 4),
        (["duration"], 1),
        (["duration", "velocity"], 2),
        (["velocity", "hausdorff"], 2),
    ],
)
def test_validate_feature_categories_valid(
    feature_categories: list[orchestrator.FeatureCategories],
    expected_valid_count: int,
) -> None:
    """Test _validate_feature_categories with valid categories."""
    valid_categories = orchestrator._validate_feature_categories(feature_categories)
    assert len(valid_categories) == expected_valid_count
    for category in feature_categories:
        assert category in valid_categories


@pytest.mark.parametrize(
    "feature_categories",
    [
        [],
        ["unknown_category"],
        ["unknown_category", "another_unknown"],
    ],
)
def test_validate_feature_categories_only_invalid(
    feature_categories: list[orchestrator.FeatureCategories],
) -> None:
    """Test _validate_feature_categories with only invalid categories."""
    with pytest.raises(ValueError, match="No valid feature categories provided"):
        orchestrator._validate_feature_categories(feature_categories)


def test_validate_feature_categories_mixed(caplog: pytest.LogCaptureFixture) -> None:
    """Test _validate_feature_categories with mix of valid and invalid categories."""
    feature_categories = typing.cast(
        list[orchestrator.FeatureCategories],
        [
            "duration",
            "meaning_of_life",
        ],
    )
    valid_categories = orchestrator._validate_feature_categories(feature_categories)

    assert len(valid_categories) == 1
    assert "duration" in valid_categories
    assert "Unknown feature categories requested" in caplog.text
    assert "meaning_of_life" in caplog.text


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
    feature_categories: list[orchestrator.FeatureCategories],
    expected_feature_number: int,
    valid_spiral: models.Spiral,
    ref_spiral: np.ndarray,
) -> None:
    """Test _get_feature_categories with various categories."""
    features = orchestrator._get_feature_categories(
        valid_spiral, ref_spiral, feature_categories
    )

    assert len(features) == expected_feature_number


def test_export_features_to_csv_extension_parent_dir(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
) -> None:
    """Test _export_features_to_csv with extension and parent directory."""
    input_path = pathlib.Path("/fake/input/path.csv")
    output_path = tmp_path / "features.csv"

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        input_path,
        output_path,
    )

    assert output_path.is_file()


def test_export_features_to_csv_extension_no_parent_dir(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with extension and no parent directory."""
    input_path = pathlib.Path("/fake/input/path.csv")
    output_path = tmp_path / "nonexistent" / "features.csv"

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        input_path,
        output_path,
    )

    assert output_path.is_file()
    assert "Creating parent directory that doesn't exist:" in caplog.text


def test_export_features_to_csv_no_extension_dir_exists(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with no extension and existing directory."""
    input_path = pathlib.Path("/fake/input/path.csv")
    output_path = tmp_path

    expected_filename = (
        f"{valid_spiral.metadata['id']}_{valid_spiral.metadata['task']}_{valid_spiral.metadata['hand']}_"
        f"features_{datetime.datetime.today().strftime('%Y%m%d')}.csv"
    )

    expected_output_path = output_path / expected_filename

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        input_path,
        output_path,
    )

    assert expected_output_path.is_file()


def test_export_features_to_csv_no_extension_no_dir(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with no extension and no directory."""
    input_path = pathlib.Path("/fake/input/path.csv")
    output_path = tmp_path / "output_dir"

    expected_filename = (
        f"{valid_spiral.metadata['id']}_{valid_spiral.metadata['task']}_{valid_spiral.metadata['hand']}_"
        f"features_{datetime.datetime.today().strftime('%Y%m%d')}.csv"
    )

    expected_output_path = output_path / expected_filename

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        input_path,
        output_path,
    )

    assert expected_output_path.is_file()
    assert "Creating directory that doesn't exist:" in caplog.text


def test_export_features_to_csv_overwrite(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with overwrite option."""
    input_path = pathlib.Path("/fake/input/path.csv")
    output_path = tmp_path / "features.csv"
    output_path.write_text("This should be overwritten\n")

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        input_path,
        output_path,
    )

    csv_content = output_path.read_text()

    assert output_path.is_file()
    assert "Overwriting existing file:" in caplog.text
    assert "This should be overwritten" not in csv_content


def test_export_features_to_csv_raise_exception(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv raises an exception with a read-only file."""
    input_path = pathlib.Path("/fake/input/path.csv")
    output_path = tmp_path / "features.csv"
    original_content = "So Long, and Thanks for All the Fish"
    output_path.write_text(original_content)
    output_path.chmod(0o444)

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        input_path,
        output_path,
    )

    assert output_path.read_text() == original_content
    assert "Failed to save features to" in caplog.text
    assert "Permission denied" in caplog.text


def test_extract_features_no_output_no_config(sample_data: pathlib.Path) -> None:
    """Test extract_features with no output path or custom config."""
    feature_categories: list[orchestrator.FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ]
    features = orchestrator.extract_features(
        sample_data, None, feature_categories, None
    )

    assert isinstance(features, dict)
    assert len(features) == 25
    assert all(isinstance(value, str) for value in features.values())
    assert all(len(value.split(".")[-1]) <= 15 for value in features.values())


def test_extract_features_with_output_path(
    sample_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test extract_features with an output path specified."""
    output_path = tmp_path / "features.csv"
    feature_categories: list[orchestrator.FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ]

    features = orchestrator.extract_features(
        sample_data, output_path, feature_categories, None
    )

    csv_content = output_path.read_text()
    lines = csv_content.strip().split("\n")
    header_lines = lines[:4]
    feature_lines = lines[4:]

    assert output_path.is_file()
    assert len(lines) == 29
    assert any(line.startswith("participant_id,") for line in header_lines)
    assert any(line.startswith("task,") for line in header_lines)
    assert any(line.startswith("hand,") for line in header_lines)
    assert any(line.startswith("source_file,") for line in header_lines)
    for line in feature_lines:
        name, value = line.split(",", 1)
        assert name in features
        assert features[name] == value
        if "." in value:
            assert len(value.split(".")[-1]) <= 15


def test_extract_features_with_custom_spiral_config(
    sample_data: pathlib.Path,
    valid_spiral: models.Spiral,
) -> None:
    """Test extract_features with a custom spiral configuration."""
    spiral_config = config.SpiralConfig.add_custom_params(
        {"center_x": 0, "center_y": 0, "growth_rate": 0}
    )
    feature_categories: list[orchestrator.FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ]
    features = orchestrator.extract_features(
        sample_data, None, feature_categories, spiral_config
    )

    expected_max_hausdorff_distance = max(
        np.sqrt(x**2 + y**2)
        for x, y in zip(valid_spiral.data["x"], valid_spiral.data["y"])
    )

    assert (
        features["hausdorff_distance_maximum"]
        == f"{expected_max_hausdorff_distance:.15f}"
    )
