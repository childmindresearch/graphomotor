"""Tests for the orchestrator module."""

import datetime
import pathlib
import typing

import numpy as np
import pandas as pd
import pytest

from graphomotor.core import config, models, orchestrator
from graphomotor.utils import center_spiral, generate_reference_spiral


@pytest.mark.parametrize(
    "valid_extension",
    [".csv", ".CSV", ""],
)
def test_validate_output_path_valid(valid_extension: str) -> None:
    """Test _validate_output_path with valid extensions."""
    path = pathlib.Path(f"/path/to/output{valid_extension}")
    orchestrator._validate_output_path(path)


@pytest.mark.parametrize(
    "invalid_extension",
    [".txt", ".json", ".xlsx", ".xml", ".dat", ".log"],
)
def test_validate_output_path_invalid(invalid_extension: str) -> None:
    """Test _validate_output_path with invalid extensions."""
    invalid_path = pathlib.Path(f"/path/to/output{invalid_extension}")

    with pytest.raises(ValueError, match="Output file must have a .csv extension"):
        orchestrator._validate_output_path(invalid_path)


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


def test_export_features_to_csv_extension_parent_dir(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
) -> None:
    """Test _export_features_to_csv with extension and parent directory."""
    output_path = tmp_path / "features.csv"

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        output_path,
    )

    assert output_path.is_file()


def test_export_features_to_csv_extension_no_parent_dir(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with extension and no parent directory."""
    output_path = tmp_path / "nonexistent" / "features.csv"

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        output_path,
    )

    assert output_path.is_file()
    assert "Creating parent directory that doesn't exist:" in caplog.text


def test_export_features_to_csv_no_extension_dir_exists(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
) -> None:
    """Test _export_features_to_csv with no extension and existing directory."""
    output_path = tmp_path

    expected_filename = (
        f"{valid_spiral.metadata['id']}_{valid_spiral.metadata['task']}_{valid_spiral.metadata['hand']}_"
        f"features_{datetime.datetime.today().strftime('%Y%m%d')}.csv"
    )

    expected_output_path = output_path / expected_filename

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        output_path,
    )

    assert expected_output_path.is_file()


def test_export_features_to_csv_no_extension_no_dir(
    valid_spiral: models.Spiral,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_features_to_csv with no extension and no directory."""
    output_path = tmp_path / "output_dir"

    expected_filename = (
        f"{valid_spiral.metadata['id']}_{valid_spiral.metadata['task']}_{valid_spiral.metadata['hand']}_"
        f"features_{datetime.datetime.today().strftime('%Y%m%d')}.csv"
    )

    expected_output_path = output_path / expected_filename

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
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
    output_path = tmp_path / "features.csv"
    output_path.write_text("This should be overwritten\n")

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
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
    output_path = tmp_path / "features.csv"
    original_content = "So Long, and Thanks for All the Fish"
    output_path.write_text(original_content)
    output_path.chmod(0o444)

    orchestrator._export_features_to_csv(
        valid_spiral,
        {"feature1": "0"},
        output_path,
    )

    assert output_path.read_text() == original_content
    assert "Failed to save features to" in caplog.text
    assert "Permission denied" in caplog.text


def test_export_directory_features_to_csv_creates_parent_dir(
    tmp_path: pathlib.Path,
) -> None:
    """Test _export_directory_features_to_csv creates parent directory if needed."""
    test_data = {
        "participant_id": ["5123456", "5123457"],
        "task": ["spiral_trace1", "spiral_trace1"],
        "hand": ["Dom", "NonDom"],
        "test_feature": [1.0, 2.0],
    }
    df = pd.DataFrame(test_data)
    df.index.name = "source_file"

    output_path = tmp_path / "nested" / "output.csv"

    orchestrator._export_directory_features_to_csv(df, output_path)
    saved_df = pd.read_csv(output_path, index_col=0)

    assert output_path.exists()
    assert len(saved_df) == 2
    assert list(saved_df.columns) == ["participant_id", "task", "hand", "test_feature"]


def test_export_directory_features_to_csv_overwrite(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_directory_features_to_csv overwrites existing file."""
    test_data = {
        "participant_id": ["5123456"],
        "task": ["spiral_trace1"],
        "hand": ["Dom"],
        "test_feature": [1.0],
    }
    df = pd.DataFrame(test_data)
    df.index.name = "source_file"

    output_path = tmp_path / "output.csv"
    output_path.write_text("This should be overwritten\n")

    orchestrator._export_directory_features_to_csv(df, output_path)

    assert output_path.exists()
    assert "Overwriting existing file:" in caplog.text
    assert "This should be overwritten" not in output_path.read_text()


def test_export_directory_features_to_csv_exception(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _export_directory_features_to_csv handles exception with read-only file."""
    test_data = {
        "participant_id": ["5123456"],
        "task": ["spiral_trace1"],
        "hand": ["Dom"],
        "test_feature": [1.0],
    }
    df = pd.DataFrame(test_data)
    df.index.name = "source_file"

    output_path = tmp_path / "output.csv"
    output_path.write_text("Original content")
    output_path.chmod(0o444)

    orchestrator._export_directory_features_to_csv(df, output_path)

    assert "Failed to save batch features to" in caplog.text
    assert "Permission denied" in caplog.text


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
        centered_spiral, None, feature_categories, centered_reference_spiral
    )

    assert len(features) == expected_feature_number
    assert all(isinstance(value, str) for value in features.values())
    assert all(len(value.split(".")[-1]) <= 15 for value in features.values())


def test_extract_features_no_output_no_config(
    valid_spiral: models.Spiral, ref_spiral: np.ndarray
) -> None:
    """Test extract_features with no output path or custom config."""
    centered_spiral = center_spiral.center_spiral(valid_spiral)
    centered_reference_spiral = center_spiral.center_spiral(ref_spiral)
    feature_categories: list[orchestrator.FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ]

    features = orchestrator.extract_features(
        centered_spiral, None, feature_categories, centered_reference_spiral
    )

    assert isinstance(features, dict)
    assert len(features) == 25
    assert all(isinstance(value, str) for value in features.values())
    assert all(len(value.split(".")[-1]) <= 15 for value in features.values())


def test_extract_features_with_output_path(
    valid_spiral: models.Spiral,
    ref_spiral: np.ndarray,
    tmp_path: pathlib.Path,
) -> None:
    """Test extract_features with an output path specified."""
    centered_spiral = center_spiral.center_spiral(valid_spiral)
    centered_reference_spiral = center_spiral.center_spiral(ref_spiral)
    output_path = tmp_path / "features.csv"
    feature_categories: list[orchestrator.FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ]

    features = orchestrator.extract_features(
        centered_spiral, output_path, feature_categories, centered_reference_spiral
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

    features = orchestrator.extract_features(
        centered_spiral, None, feature_categories, centered_reference_spiral
    )

    expected_max_hausdorff_distance = max(
        np.sqrt(x**2 + y**2)
        for x, y in zip(valid_spiral.data["x"], valid_spiral.data["y"])
    )

    assert (
        features["hausdorff_distance_maximum"]
        == f"{expected_max_hausdorff_distance:.15f}"
    )


def test_run_pipeline_single_file(sample_data: pathlib.Path) -> None:
    """Test run_pipeline with a single file."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    features = orchestrator.run_pipeline(sample_data, None, feature_categories)

    assert isinstance(features, dict)
    assert "duration" in features
    assert all(isinstance(key, str) for key in features.keys())
    assert all(isinstance(value, str) for value in features.values())


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
    assert all(isinstance(value, str) for value in features["duration"].values)

    assert all(isinstance(value, str) for value in features["participant_id"].values)
    assert all(isinstance(value, str) for value in features["task"].values)
    assert all(isinstance(value, str) for value in features["hand"].values)


def test_run_pipeline_invalid_path() -> None:
    """Test run_pipeline with invalid path."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Input path does not exist"):
        orchestrator.run_pipeline("/nonexistent/path", None, feature_categories)


def test_run_pipeline_empty_directory(tmp_path: pathlib.Path) -> None:
    """Test run_pipeline with empty directory should raise ValueError."""
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No CSV files found in directory"):
        orchestrator.run_pipeline(empty_dir, None, feature_categories)


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
    csv_file = input_dir / sample_data.name  # Use the original filename
    csv_file.write_text(sample_data.read_text())

    output_path = tmp_path / "output.txt"
    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    with pytest.raises(ValueError, match="Output file must have a .csv extension"):
        orchestrator.run_pipeline(input_dir, output_path, feature_categories)


def test_run_pipeline_directory_to_single_csv(
    tmp_path: pathlib.Path, sample_data: pathlib.Path
) -> None:
    """Test run_pipeline saves directory processing results to single CSV file."""
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
    assert f"Failed to process {len([invalid_file.name])} files:" in caplog.text
    assert invalid_file.name in caplog.text
    assert "Batch processing complete. Successfully processed 1 files" in caplog.text


def test_run_pipeline_directory_all_files_fail(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test run_pipeline with directory where all files fail processing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    for i in range(2):
        invalid_file = input_dir / f"[512345{i}]test-spiral_trace1_Dom.csv"
        invalid_file.write_text("invalid,csv,data\n1,2,3")

    feature_categories: list[orchestrator.FeatureCategories] = ["duration"]

    result = orchestrator.run_pipeline(input_dir, None, feature_categories)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "Failed to process 2 files:" in caplog.text
    assert "Batch processing complete. Successfully processed 0 files" in caplog.text
