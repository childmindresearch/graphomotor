"""Tests for the orchestrator module."""

import pathlib
import typing

import numpy as np
import pandas as pd
import pytest

from graphomotor.core import config, models, orchestrator
from graphomotor.utils import center_spiral, generate_reference_spiral


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
def test_validate_feature_categories_invalid(
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


# Tests for extract_features()


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
    feature_categories: list[orchestrator.FeatureCategories],
    expected_feature_number: int,
    valid_spiral: models.Spiral,
    ref_spiral: np.ndarray,
) -> None:
    """Test extract_features with various feature categories."""
    centered_spiral = center_spiral.center_spiral(valid_spiral)
    centered_reference_spiral = center_spiral.center_spiral(ref_spiral)

    features = orchestrator.extract_features(
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
    valid_spiral: models.Spiral,
) -> None:
    """Test extract_features with a custom spiral configuration."""
    centered_spiral = center_spiral.center_spiral(valid_spiral)
    spiral_config = config.SpiralConfig.add_custom_params(
        {"center_x": 0, "center_y": 0, "growth_rate": 0}
    )
    feature_categories: list[orchestrator.FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ]
    reference_spiral = generate_reference_spiral.generate_reference_spiral(
        spiral_config=spiral_config
    )
    centered_reference_spiral = center_spiral.center_spiral(reference_spiral)
    expected_max_hausdorff_distance = max(
        np.sqrt(x**2 + y**2)
        for x, y in zip(valid_spiral.data["x"], valid_spiral.data["y"])
    )

    features = orchestrator.extract_features(
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

    orchestrator.export_features_to_csv(sample_features, output_path)
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
        orchestrator.export_features_to_csv(sample_features, output_path)

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
        orchestrator.export_features_to_csv(sample_features, output_path)
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

    orchestrator.export_features_to_csv(test_df, output_path)
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
        orchestrator.export_features_to_csv(sample_features, output_path)

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

    orchestrator.export_features_to_csv(sample_features, readonly_path)

    assert readonly_path.read_text() == original_content
    assert "Failed to save features to" in caplog.text
    assert "Permission denied" in caplog.text


def test_run_pipeline_single_file(sample_data: pathlib.Path) -> None:
    """Test run_pipeline with a single file."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    features = orchestrator.run_pipeline(sample_data, None, feature_categories)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert "duration" in features.columns
    assert "participant_id" in features.columns
    assert "task" in features.columns
    assert "hand" in features.columns
    assert "start_time" in features.columns
    assert "source_file" == features.index.name


def test_run_pipeline_directory(sample_data: pathlib.Path) -> None:
    """Test run_pipeline with a directory containing multiple files."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]
    sample_dir = sample_data.parent
    expected_columns = ["participant_id", "task", "hand", "duration"]

    features = orchestrator.run_pipeline(sample_dir, None, feature_categories)

    assert isinstance(features, pd.DataFrame)
    assert len(features) == 2

    for col in expected_columns:
        assert col in features.columns

    assert all(isinstance(index, str) for index in features.index)
    assert all(isinstance(value, str) for value in features.values.flatten())


def test_run_pipeline_directory_comprehensive(
    tmp_path: pathlib.Path, sample_data: pathlib.Path
) -> None:
    """Test run_pipeline with directory processing and export functionality."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    for i in range(1, 4):
        filename = f"[5123456]test-spiral_trace{i}_Dom.csv"
        csv_file = input_dir / filename
        csv_file.write_text(sample_data.read_text())

    output_csv = tmp_path / "all_features.csv"
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    result = orchestrator.run_pipeline(input_dir, output_csv, feature_categories)
    saved_df = pd.read_csv(output_csv, index_col=0)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert output_csv.exists()
    assert len(saved_df) == 3
    assert "participant_id" in saved_df.columns
    assert "task" in saved_df.columns
    assert "hand" in saved_df.columns


def test_run_pipeline_directory_with_failed_files(
    tmp_path: pathlib.Path,
    sample_data: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test run_pipeline with directory containing files that fail processing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    valid_file = input_dir / "[5123456]test-spiral_trace1_Dom.csv"
    valid_file.write_text(sample_data.read_text())

    invalid_file = input_dir / "[5123457]test-spiral_trace1_Dom.csv"
    invalid_file.write_text("invalid,csv,data\n1,2,3")

    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    result = orchestrator.run_pipeline(input_dir, None, feature_categories)

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

    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Could not extract features from any file"):
        orchestrator.run_pipeline(input_dir, None, feature_categories)


def test_run_pipeline_invalid_path() -> None:
    """Test run_pipeline with invalid path."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(
        ValueError, match="Input path does not exist or is not a file/directory"
    ):
        orchestrator.run_pipeline("/nonexistent/path", None, feature_categories)


def test_run_pipeline_empty_directory(tmp_path: pathlib.Path) -> None:
    """Test run_pipeline with empty directory should raise ValueError."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No CSV files found in directory"):
        orchestrator.run_pipeline(empty_dir, None, feature_categories)


@pytest.mark.parametrize(
    "valid_extension",
    [".csv", ".CSV", ""],
)
def test_run_pipeline_output_path_valid(
    sample_data: pathlib.Path, tmp_path: pathlib.Path, valid_extension: str
) -> None:
    """Test run_pipeline validates output paths with valid extensions."""
    output_path = tmp_path / f"output{valid_extension}"
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    result = orchestrator.run_pipeline(sample_data, output_path, feature_categories)

    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "invalid_extension",
    [".txt", ".json", ".xlsx", ".xml", ".dat", ".log"],
)
def test_run_pipeline_output_path_invalid(
    sample_data: pathlib.Path, tmp_path: pathlib.Path, invalid_extension: str
) -> None:
    """Test run_pipeline validates output paths with invalid extensions."""
    output_path = tmp_path / f"output{invalid_extension}"
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Output file must have a .csv extension"):
        orchestrator.run_pipeline(sample_data, output_path, feature_categories)


def test_run_pipeline_invalid_output_extension_single_file(
    tmp_path: pathlib.Path, sample_data: pathlib.Path
) -> None:
    """Test run_pipeline raises error for single file with invalid output extension."""
    output_path = tmp_path / "output.txt"
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Output file must have a .csv extension"):
        orchestrator.run_pipeline(sample_data, output_path, feature_categories)


def test_run_pipeline_invalid_output_extension_directory(
    tmp_path: pathlib.Path, sample_data: pathlib.Path
) -> None:
    """Test run_pipeline raises error for directory with invalid output extension."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    csv_file = input_dir / sample_data.name
    csv_file.write_text(sample_data.read_text())

    output_path = tmp_path / "output.txt"
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Output file must have a .csv extension"):
        orchestrator.run_pipeline(input_dir, output_path, feature_categories)
