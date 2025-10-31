"""Test cases for spiral_plots.py functions."""

import pathlib
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from graphomotor.core import models
from graphomotor.plot import spiral_plots


def test_single_spiral_saved(
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting of a single spiral and saving the figure."""
    output_dir = tmp_path / "single_spiral"
    output_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = spiral_plots.plot_single_spiral(
            data=sample_spiral_data,
            output_path=output_dir,
            include_reference=True,
            color_segments=True,
        )
    output_files = list(output_dir.glob("*.png"))

    assert "Starting single spiral plot generation" in caplog.text
    assert "Saving figure to" in caplog.text
    assert "Figure saved successfully" in caplog.text

    assert len(output_files) == 1
    assert output_files[0].is_file()

    assert isinstance(fig, plt.Figure)

    assert len(fig.axes) == 1
    assert fig.axes[0].has_data()


def test_plot_batch_spirals_saved(
    sample_batch_spirals: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting of batch spirals and saving the figure."""
    output_dir = tmp_path / "batch_spirals"
    output_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = spiral_plots.plot_batch_spirals(
            data=sample_batch_spirals,
            output_path=output_dir,
            include_reference=True,
            color_segments=True,
        )
    output_files = list(output_dir.glob("*.png"))

    assert "Starting batch spiral plot generation" in caplog.text
    assert "Found 30 CSV files to process" in caplog.text
    assert "Saving figure to" in caplog.text
    assert "Figure saved successfully" in caplog.text

    assert len(output_files) == 1
    assert output_files[0].is_file()

    assert isinstance(fig, plt.Figure)

    assert len(fig.axes) == 32
    assert sum(1 for ax in fig.axes if ax.has_data()) == 30

    plt.close(fig)


def test_plot_single_spiral_invalid_data_type_error(
    sample_batch_features: pathlib.Path,
) -> None:
    """Test ValueError when invalid data type is passed to plot_single_spiral."""
    with pytest.raises(ValueError, match="Failed to load spiral data"):
        spiral_plots.plot_single_spiral(data=sample_batch_features)


def test_plot_single_spiral_with_valid_spiral_object(
    valid_spiral: models.Drawing,
) -> None:
    """Test plotting with a pre-loaded valid Spiral object."""
    fig = spiral_plots.plot_single_spiral(
        data=valid_spiral,
        include_reference=True,
        color_segments=False,
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].has_data()

    plt.close(fig)


def test_plot_single_spiral_invalid_dataframe_error(
    valid_spiral_data: pd.DataFrame,
) -> None:
    """Test ValueError when DataFrame (not Spiral object) is passed."""
    with pytest.raises(
        ValueError, match="Invalid data type.*Expected str, Path, or Spiral"
    ):
        spiral_plots.plot_single_spiral(data=valid_spiral_data)


def test_plot_batch_spirals_invalid_input_path_error() -> None:
    """Test ValueError when input path doesn't exist or is not a directory."""
    nonexistent_path = pathlib.Path("/nonexistent/path")

    with pytest.raises(
        ValueError, match="Input path does not exist or is not a directory"
    ):
        spiral_plots.plot_batch_spirals(data=nonexistent_path)


def test_plot_batch_spirals_empty_directory_error(
    sample_batch_features: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Test ValueError when directory contains no valid spiral files."""
    single_spiral_dir = tmp_path / "single_spiral_dir"
    single_spiral_dir.mkdir()
    copied_file = single_spiral_dir / sample_batch_features.name
    shutil.copy2(sample_batch_features, copied_file)

    with pytest.raises(ValueError, match="Could not load any valid spiral files"):
        spiral_plots.plot_batch_spirals(data=single_spiral_dir)


def test_plot_batch_spirals_failed_files_warning(
    sample_batch_features: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test logger warning when some files fail to load."""
    test_dir = sample_batch_features.parent

    spiral_plots.plot_batch_spirals(data=test_dir)

    assert "Failed to load 1 files" in caplog.text


def test_plot_batch_spirals_single_spiral_fallback(
    sample_spiral_data: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Test that plot_single_spiral is called when only one spiral is loaded."""
    single_spiral_dir = tmp_path / "single_spiral_dir"
    single_spiral_dir.mkdir()
    copied_file = single_spiral_dir / sample_spiral_data.name
    shutil.copy2(sample_spiral_data, copied_file)

    fig = spiral_plots.plot_batch_spirals(data=single_spiral_dir)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    assert fig.axes[0].has_data()

    plt.close(fig)
