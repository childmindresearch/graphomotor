"""Test cases for feature_plots.py functions."""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from graphomotor.plot import feature_plots


def test_plot_feature_distributions_all_saved(
    sample_batch_features_df: pd.DataFrame,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting of feature distributions with all features and saving."""
    output_dir = tmp_path / "feature_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_distributions(
            data=sample_batch_features_df, output_path=output_dir
        )
    output_files = list(output_dir.glob("*.png"))
    legend_labels = []
    legend = fig.axes[0].legend_
    if legend is not None:
        legend_labels = [t.get_text() for t in legend.get_texts()]

    assert "Starting feature distributions plot generation" in caplog.text
    assert "Created subplot grid for 25 features" in caplog.text
    assert "Saving figure to" in caplog.text
    assert "Figure saved successfully" in caplog.text

    assert len(output_files) == 1
    assert output_files[0].is_file()

    assert isinstance(fig, plt.Figure)
    assert fig.get_suptitle() == "Feature Distributions across Task Types and Hands"
    assert len(fig.axes) == 25
    for ax in fig.axes:
        assert ax.has_data()
        assert ax.get_ylabel() == "Density"
        legend = ax.get_legend()
        assert (
            legend is not None and legend.get_title().get_text() == "Hand - Task Type"
        )
        assert ax.get_title().find("\n") < 21
        assert len(ax.get_xgridlines()) > 0 or len(ax.get_ygridlines()) > 0
    assert len(legend_labels) == 4

    plt.close(fig)


def test_plot_feature_distributions_specific_features_no_save(
    sample_batch_features_df: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting specific feature distributions without saving."""
    features_to_plot = ["duration", "area_under_curve", "hausdorff_distance_maximum"]

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_distributions(
            data=sample_batch_features_df,
            features=features_to_plot,
        )

    assert "Created subplot grid for 3 features" in caplog.text
    assert "Feature distributions plot generated but not saved" in caplog.text

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    assert not fig.axes[-1].get_visible()
    assert not fig.axes[-1].has_data()

    for ax in fig.axes[:-1]:
        assert ax.has_data()
        assert ax.get_title().replace("\n", "_") in features_to_plot

    plt.close(fig)


def test_plot_feature_trends_all_saved(
    sample_batch_features_df: pd.DataFrame,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting of feature trends with all features and saving."""
    output_dir = tmp_path / "feature_trends"
    output_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_trends(
            data=sample_batch_features_df, output_path=output_dir
        )
    output_files = list(output_dir.glob("*.png"))

    assert "Starting feature trends plot generation" in caplog.text

    assert len(output_files) == 1
    assert output_files[0].is_file()

    assert isinstance(fig, plt.Figure)
    assert fig.get_suptitle() == "Feature Trends across Tasks and Hands"
    assert len(fig.axes) == 25
    for ax in fig.axes:
        assert ax.has_data()
        assert ax.get_xlabel() == "Task"
        legend = ax.get_legend()
        assert legend is not None and legend.get_title().get_text() == "Hand"
        assert ax.get_title().find("\n") < 21
        assert len(ax.get_xgridlines()) > 0 or len(ax.get_ygridlines()) > 0

    plt.close(fig)


def test_plot_feature_trends_specific_features_no_save(
    sample_batch_features_df: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting specific feature trends without saving."""
    features_to_plot = ["duration", "area_under_curve", "hausdorff_distance_maximum"]

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_trends(
            data=sample_batch_features_df,
            features=features_to_plot,
        )

    assert "Feature trends plot generated but not saved" in caplog.text

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    assert not fig.axes[-1].get_visible()
    assert not fig.axes[-1].has_data()

    for ax in fig.axes[:-1]:
        assert ax.has_data()
        assert ax.get_title().replace("\n", "_") in features_to_plot

    plt.close(fig)


def test_plot_feature_boxplots_all_saved(
    sample_batch_features_df: pd.DataFrame,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting of feature boxplots with all features and saving."""
    output_dir = tmp_path / "feature_boxplots"
    output_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_boxplots(
            data=sample_batch_features_df, output_path=output_dir
        )
    output_files = list(output_dir.glob("*.png"))

    assert "Starting feature boxplots generation" in caplog.text

    assert len(output_files) == 1
    assert output_files[0].is_file()

    assert isinstance(fig, plt.Figure)
    assert fig.get_suptitle() == "Feature Boxplots across Tasks and Hands"
    assert len(fig.axes) == 25
    for ax in fig.axes:
        assert ax.has_data()
        assert ax.get_xlabel() == "Task"
        legend = ax.get_legend()
        assert legend is not None and legend.get_title().get_text() == "Hand"
        assert ax.get_title().find("\n") < 21
        assert len(ax.get_xgridlines()) > 0 or len(ax.get_ygridlines()) > 0

    plt.close(fig)


def test_plot_feature_boxplots_specific_features_no_save(
    sample_batch_features_df: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting specific feature boxplots without saving."""
    features_to_plot = ["duration", "area_under_curve", "hausdorff_distance_maximum"]

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_boxplots(
            data=sample_batch_features_df,
            features=features_to_plot,
        )

    assert "Feature boxplots generated but not saved" in caplog.text

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    assert not fig.axes[-1].get_visible()
    assert not fig.axes[-1].has_data()

    for ax in fig.axes[:-1]:
        assert ax.has_data()
        assert ax.get_title().replace("\n", "_") in features_to_plot

    plt.close(fig)


def test_plot_feature_clusters_all_saved(
    sample_batch_features_df: pd.DataFrame,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting of feature clusters with all features and saving."""
    output_dir = tmp_path / "feature_clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_clusters(
            data=sample_batch_features_df, output_path=output_dir
        )
    output_files = list(output_dir.glob("*.png"))

    assert "Starting feature clusters heatmap generation" in caplog.text
    assert "Heatmap data shape: (25, 16) for (features, conditions)" in caplog.text
    assert len(output_files) == 1
    assert output_files[0].is_file()

    assert isinstance(fig, plt.Figure)
    assert fig.get_suptitle() == "Feature Clusters Across Conditions"
    assert len(fig.axes) == 4  # dendograms + heatmap + colorbar
    for ax in fig.axes:
        assert ax.has_data()
    assert fig.axes[2].get_xlabel() == "Condition"
    assert fig.axes[2].get_ylabel() == "Feature"
    assert fig.axes[3].get_xlabel() == "z-score"

    plt.close(fig)


def test_plot_feature_clusters_specific_features_no_save(
    sample_batch_features_df: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test plotting specific feature clusters without saving."""
    features_to_plot = ["duration", "area_under_curve", "hausdorff_distance_maximum"]

    with caplog.at_level("DEBUG", logger="graphomotor"):
        fig = feature_plots.plot_feature_clusters(
            data=sample_batch_features_df,
            features=features_to_plot,
        )
    yticklabels = [label.get_text() for label in fig.axes[2].get_yticklabels()]

    assert "Heatmap data shape: (3, 16) for (features, conditions)" in caplog.text
    assert "Feature clusters heatmap generated but not saved" in caplog.text

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # dendograms + heatmap + colorbar
    for ax in fig.axes:
        assert ax.has_data()
    for label in yticklabels:
        assert label in features_to_plot

    plt.close(fig)


def test_plot_feature_clusters_insufficient_features_raises_error(
    sample_batch_features_df: pd.DataFrame,
) -> None:
    """Test that plotting feature clusters with <2 features raises ValueError."""
    features_to_plot = ["duration"]

    with pytest.raises(
        ValueError,
        match="At least 2 features required for clustered heatmap, got 1",
    ):
        feature_plots.plot_feature_clusters(
            data=sample_batch_features_df,
            features=features_to_plot,
        )
