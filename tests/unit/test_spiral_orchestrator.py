"""Tests for the orchestrator module."""

import datetime
import pathlib
import re
import typing

import numpy as np
import pandas as pd
import pytest

from graphomotor.core import config, models, spiral_orchestrator
from graphomotor.utils import center_spiral, generate_reference_spiral


def test_parse_filename_valid(sample_spiral_data: pathlib.Path) -> None:
    """Test that valid filenames are parsed correctly."""
    expected_metadata = {
        "id": "5000000",
        "hand": "Dom",
        "task": "spiral_trace1",
    }
    metadata = spiral_orchestrator._parse_spiral_metadata(sample_spiral_data.stem)
    assert metadata == expected_metadata, (
        f"Expected {expected_metadata}, but got {metadata}"
    )


@pytest.mark.parametrize(
    "invalid_filename",
    [
        "asdf123-spiral_trace1_Dom.csv",  # missing ID
        "[5123456]-spiral_trace1_Dom.csv",  # missing Curious ID
        "[5123456]asdf123-Dom.csv",  # missing task
        "[5123456]asdf123-spiral_trace1.csv",  # missing hand
    ],
)
def test_parse_filename_invalid(invalid_filename: str) -> None:
    """Test that invalid filenames raise a ValueError."""
    filename = re.escape(invalid_filename)
    with pytest.raises(
        ValueError,
        match=f"Filename does not match expected pattern: {filename}",
    ):
        spiral_orchestrator._parse_spiral_metadata(invalid_filename)


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
    feature_categories: list[spiral_orchestrator.FeatureCategories],
    expected_valid_count: int,
) -> None:
    """Test _validate_feature_categories with valid categories."""
    valid_categories = spiral_orchestrator._validate_feature_categories(
        feature_categories
    )

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
def test_validate_feature_categories_invalid(
    feature_categories: list[spiral_orchestrator.FeatureCategories],
) -> None:
    """Test _validate_feature_categories with only invalid categories."""
    with pytest.raises(ValueError, match="No valid feature categories provided"):
        spiral_orchestrator._validate_feature_categories(feature_categories)


def test_validate_feature_categories_mixed(caplog: pytest.LogCaptureFixture) -> None:
    """Test _validate_feature_categories with mix of valid and invalid categories."""
    feature_categories = typing.cast(
        list[spiral_orchestrator.FeatureCategories],
        [
            "duration",
            "meaning_of_life",
        ],
    )

    valid_categories = spiral_orchestrator._validate_feature_categories(
        feature_categories
    )

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
def test_extract_features_categories(
    feature_categories: list[str],
    expected_feature_number: int,
    valid_spiral: models.Drawing,
    ref_spiral: np.ndarray,
) -> None:
    """Test extract_features with various feature categories."""
    centered_spiral = center_spiral.center_spiral(valid_spiral)
    centered_reference_spiral = center_spiral.center_spiral(ref_spiral)

    features = spiral_orchestrator.extract_features(
        centered_spiral, feature_categories, centered_reference_spiral
    )

    assert len(features) == expected_feature_number + 5
    assert all(isinstance(value, str) for value in features.values())
    assert "participant_id" in features
    assert "task" in features
    assert "hand" in features
    assert "source_file" in features
    assert "start_time" in features


def test_extract_features_with_custom_spiral_config(
    valid_spiral: models.Drawing,
) -> None:
    """Test extract_features with a custom spiral configuration."""
    centered_spiral = center_spiral.center_spiral(valid_spiral)
    spiral_config = config.SpiralConfig.add_custom_params(
        {"center_x": 0, "center_y": 0, "growth_rate": 0}
    )
    feature_categories = ["duration", "velocity", "hausdorff", "AUC"]
    reference_spiral = generate_reference_spiral.generate_reference_spiral(
        spiral_config=spiral_config
    )
    centered_reference_spiral = center_spiral.center_spiral(reference_spiral)
    expected_max_hausdorff_distance = max(
        np.sqrt(x**2 + y**2)
        for x, y in zip(valid_spiral.data["x"], valid_spiral.data["y"])
    )

    features = spiral_orchestrator.extract_features(
        centered_spiral, feature_categories, centered_reference_spiral
    )

    assert (
        features["hausdorff_distance_maximum"]
        == f"{expected_max_hausdorff_distance:.15f}"
    )


def test_export_features_to_csv_basic(
    tmp_path: pathlib.Path,
    sample_features: pd.DataFrame,
) -> None:
    """Test basic export_features_to_csv functionality."""
    output_path = tmp_path / "features.csv"

    spiral_orchestrator.export_features_to_csv(sample_features, output_path)
    saved_df = pd.read_csv(output_path, index_col=0)

    assert output_path.is_file()
    assert len(saved_df) == 1
    assert list(saved_df.columns) == ["participant_id", "task", "hand", "test_feature"]


def test_export_features_to_csv_file_with_parent_creation(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    sample_features: pd.DataFrame,
) -> None:
    """Test export_features_to_csv creates parent directory when needed."""
    output_path = tmp_path / "nonexistent" / "features.csv"

    with caplog.at_level("DEBUG", logger="graphomotor"):
        spiral_orchestrator.export_features_to_csv(sample_features, output_path)

    assert output_path.is_file()
    assert "Creating parent directory that doesn't exist:" in caplog.text


def test_export_features_to_csv_directory_single_row(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    sample_features: pd.DataFrame,
) -> None:
    """Test export_features_to_csv with directory output for single participant."""
    output_path = tmp_path / "output_dir"

    with caplog.at_level("DEBUG", logger="graphomotor"):
        spiral_orchestrator.export_features_to_csv(sample_features, output_path)
    created_files = list(output_path.glob("5123456_spiral_trace1_Dom_features_*.csv"))

    assert len(created_files) == 1
    assert "Creating directory that doesn't exist:" in caplog.text


def test_export_features_to_csv_directory_batch(
    tmp_path: pathlib.Path,
    sample_features: pd.DataFrame,
) -> None:
    """Test export_features_to_csv with directory output for multiple participants."""
    output_path = tmp_path
    test_df = pd.concat([sample_features, sample_features])

    spiral_orchestrator.export_features_to_csv(test_df, output_path)
    created_files = list(output_path.glob("batch_features_*.csv"))

    assert len(created_files) == 1


def test_export_features_to_csv_overwrite(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    sample_features: pd.DataFrame,
) -> None:
    """Test export_features_to_csv overwrites existing files."""
    output_path = tmp_path / "features.csv"
    output_path.write_text("This should be overwritten\n")

    with caplog.at_level("DEBUG", logger="graphomotor"):
        spiral_orchestrator.export_features_to_csv(sample_features, output_path)

    assert output_path.is_file()
    assert "Overwriting existing file:" in caplog.text
    assert "This should be overwritten" not in output_path.read_text()


def test_export_features_to_csv_permission_error(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    sample_features: pd.DataFrame,
) -> None:
    """Test export_features_to_csv handles file permission errors."""
    readonly_path = tmp_path / "readonly.csv"
    original_content = "Original content"
    readonly_path.write_text(original_content)
    readonly_path.chmod(0o444)

    spiral_orchestrator.export_features_to_csv(sample_features, readonly_path)

    assert readonly_path.read_text() == original_content
    assert "Failed to save features to" in caplog.text
    assert "Permission denied" in caplog.text


def test_run_pipeline_directory(sample_spiral_data: pathlib.Path) -> None:
    """Test run_pipeline with a directory containing multiple files."""
    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]
    sample_dir = sample_spiral_data.parent
    expected_columns = ["participant_id", "task", "hand", "duration"]

    features = spiral_orchestrator.run_pipeline(sample_dir, None, feature_categories)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 2

    for col in expected_columns:
        assert col in features.columns

    assert all(isinstance(index, str) for index in features.index)
    assert all(isinstance(value, str) for value in features.values.flatten())


def test_run_pipeline_directory_with_failed_files(
    tmp_path: pathlib.Path,
    sample_spiral_data: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test run_pipeline with directory containing files that fail processing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    valid_file = input_dir / "[5000000]test-spiral_trace1_Dom.csv"
    valid_file.write_text(sample_spiral_data.read_text())

    invalid_file = input_dir / "[5000002]test-spiral_trace1_Dom.csv"
    invalid_file.write_text("invalid,csv,data\n1,2,3")

    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]

    with caplog.at_level("DEBUG", logger="graphomotor"):
        result = spiral_orchestrator.run_pipeline(input_dir, None, feature_categories)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert f"Failed to process {len([invalid_file.name])} files" in caplog.text
    assert f"Failed to process {invalid_file.name}:" in caplog.text
    assert "Batch processing complete, successfully processed 1 files" in caplog.text


def test_run_pipeline_directory_all_files_fail(tmp_path: pathlib.Path) -> None:
    """Test run_pipeline with directory where all files fail processing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    for i in range(2):
        invalid_file = input_dir / f"[512345{i}]test-spiral_trace1_Dom.csv"
        invalid_file.write_text("invalid,csv,data\n1,2,3")

    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Could not extract features from any file"):
        spiral_orchestrator.run_pipeline(input_dir, None, feature_categories)


def test_run_pipeline_invalid_path() -> None:
    """Test run_pipeline with invalid path."""
    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(
        ValueError, match="Input path does not exist or is not a file/directory"
    ):
        spiral_orchestrator.run_pipeline("/nonexistent/path", None, feature_categories)


def test_run_pipeline_empty_directory(tmp_path: pathlib.Path) -> None:
    """Test run_pipeline with empty directory should raise ValueError."""
    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No CSV files found in directory"):
        spiral_orchestrator.run_pipeline(empty_dir, None, feature_categories)


@pytest.mark.parametrize(
    "valid_extension",
    [".csv", ".CSV", ""],
)
def test_run_pipeline_output_path_valid(
    sample_spiral_data: pathlib.Path, tmp_path: pathlib.Path, valid_extension: str
) -> None:
    """Test run_pipeline validates output paths with valid extensions."""
    output_path = tmp_path / f"output{valid_extension}"
    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]

    result = spiral_orchestrator.run_pipeline(
        sample_spiral_data, output_path, feature_categories
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "duration" in result.columns
    assert "participant_id" in result.columns
    assert "task" in result.columns
    assert "hand" in result.columns
    assert "start_time" in result.columns
    assert "source_file" == result.index.name


@pytest.mark.parametrize(
    "invalid_extension",
    [".txt", ".json", ".xlsx"],
)
def test_run_pipeline_output_path_invalid(
    sample_spiral_data: pathlib.Path, tmp_path: pathlib.Path, invalid_extension: str
) -> None:
    """Test run_pipeline validates output paths with invalid extensions."""
    output_path = tmp_path / f"output{invalid_extension}"
    feature_categories: list[spiral_orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Output file must have a .csv extension"):
        spiral_orchestrator.run_pipeline(
            sample_spiral_data, output_path, feature_categories
        )


@pytest.mark.parametrize(
    "key,invalid_value,expected_error",
    [
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
    valid_spiral_metadata: dict[str, str | datetime.datetime],
    key: str,
    invalid_value: str | datetime.datetime,
    expected_error: str,
) -> None:
    """Test validation errors for various invalid metadata values."""
    invalid_metadata = valid_spiral_metadata.copy()
    invalid_metadata[key] = invalid_value

    with pytest.raises(ValueError, match=expected_error):
        spiral_orchestrator._validate_spiral_metadata(invalid_metadata)
