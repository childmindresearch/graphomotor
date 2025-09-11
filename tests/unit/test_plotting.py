"""Test cases for plotting.py functions."""

import pathlib

import pandas as pd
import pytest

from graphomotor.utils import plotting


@pytest.mark.parametrize(
    "valid_input, expected_output",
    [
        ("sample_batch_features", "Successfully loaded"),
        ("sample_batch_features_df", "Using provided DataFrame with"),
    ],
)
def test_load_data_valid(
    valid_input: str,
    expected_output: str,
    caplog: pytest.LogCaptureFixture,
    request: pytest.FixtureRequest,
) -> None:
    """Test loading data using a string path."""
    with caplog.at_level("DEBUG", logger="graphomotor"):
        result = plotting._load_features_dataframe(request.getfixturevalue(valid_input))

    assert (
        f"{expected_output} {len(result)} rows and {len(result.columns)} columns"
        in caplog.text
    )


def test_load_data_file_not_found_string_path() -> None:
    """Test FileNotFoundError when string path doesn't exist."""
    nonexistent_path = "/nonexistent/path/file.csv"

    with pytest.raises(FileNotFoundError, match="Features file not found"):
        plotting._load_features_dataframe(nonexistent_path)


def test_load_data_invalid_csv_content(tmp_path: pathlib.Path) -> None:
    """Test ValueError when CSV file is corrupted or unreadable."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text('invalid,csv,content\nwith,malformed\n"unclosed quote')

    with pytest.raises(ValueError, match="Failed to read CSV file"):
        plotting._load_features_dataframe(bad_csv)


def test_validate_features_dataframe_invalid_spiral_dataframe() -> None:
    """Test ValueError when DataFrame doesn't pass Spiral validation."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="DataFrame is empty"):
        plotting._validate_features_dataframe(empty_df)


@pytest.mark.parametrize(
    "missing_column",
    [["participant_id"], ["task"], ["hand"]],
)
def test_validate_features_dataframe_missing_metadata_columns(
    missing_column: list[str], sample_features: pd.DataFrame
) -> None:
    """Test ValueError when required metadata columns are missing."""
    df = sample_features.copy()
    df = df.drop(columns=missing_column)

    with pytest.raises(
        ValueError, match=f"Required metadata columns missing: \\{missing_column}"
    ):
        plotting._validate_features_dataframe(df)


def test_validate_features_dataframe_invalid_metadata_values(
    sample_features: pd.DataFrame,
) -> None:
    """Test ValueError when metadata values don't pass validation."""
    df = sample_features.copy()
    df["participant_id"] = "invalid_id"

    with pytest.raises(ValueError):
        plotting._validate_features_dataframe(df)


def test_validate_features_dataframe_no_features_found(
    sample_features: pd.DataFrame,
) -> None:
    """Test ValueError when no features can be found."""
    df = sample_features.copy()
    df = df.drop(columns=["test_feature"])

    with pytest.raises(ValueError, match="No feature columns found to plot"):
        plotting._validate_features_dataframe(df)


def test_validate_features_dataframe_missing_requested_features(
    sample_features: pd.DataFrame,
) -> None:
    """Test ValueError when requested features don't exist in DataFrame."""
    with pytest.raises(ValueError, match="Features not found in data"):
        plotting._validate_features_dataframe(
            sample_features, requested_features=["nonexistent_feature"]
        )


def test_validate_features_dataframe_logging_behavior(
    sample_batch_features_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that appropriate logging messages are generated."""
    with caplog.at_level("DEBUG", logger="graphomotor"):
        features = plotting._validate_features_dataframe(
            sample_batch_features_df, ["hausdorff_distance_maximum"]
        )

    assert "Validating features and metadata" in caplog.text
    assert "Metadata columns validation passed" in caplog.text
    assert "All metadata rows validation passed" in caplog.text
    assert f"Using {len(features)} user-specified features" in caplog.text
    assert "Feature validation completed successfully" in caplog.text


def test_load_spirals_from_directory_no_csv_files(tmp_path: pathlib.Path) -> None:
    """Test ValueError when directory contains no CSV files."""
    empty_dir = tmp_path / "empty_directory"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No CSV files found in directory"):
        plotting.load_spirals_from_directory(empty_dir)


def test_load_spirals_from_directory_failed_file_warning(
    sample_data: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test logger warning when individual CSV files fail to load."""
    test_dir = sample_data.parent

    spirals = plotting.load_spirals_from_directory(test_dir)

    assert "Failed to load 1 files" in caplog.text
    assert len(spirals) == 2
