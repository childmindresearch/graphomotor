"""Feature visualization functions for Graphomotor.

This module provides plotting functions for visualizing extracted features from spiral
drawing data. The plotting functions expect CSV files with the first 5 columns reserved
for metadata (`source_file`, `participant_id`, `task`, `hand`, `start_time`), and treat
all subsequent columns as numerical features.

Available Features
------------------
The graphomotor toolkit extracts 25 features from spiral drawing data.
For a complete list of all available features, see the
[features module documentation](https://childmindresearch.github.io/graphomotor/graphomotor/features.html).

Custom Features
---------------
Users can add custom feature columns to their CSV files alongside the standard
graphomotor features. Any additional columns after the first 5 metadata columns
will be automatically detected and available for plotting.

Plot Types
----------
- **Distribution plots**: Kernel density estimation plots showing feature distributions
  grouped by task type and hand.
- **Trend plots**: Line plots displaying feature progression across task sequences.
- **Box plots**: Box-and-whisker plots comparing distributions across conditions.
- **Cluster heatmaps**: Hierarchically clustered heatmaps of standardized features.
"""

import pathlib
import warnings

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from graphomotor.core import config
from graphomotor.utils import plotting

matplotlib.use("agg")  # prevent interactive matplotlib
logger = config.get_logger()


def plot_feature_distributions(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> plt.Figure:
    """Plot histograms for each feature grouped by task type and hand.

    This function creates kernel density estimation plots showing feature distributions
    grouped by task type (trace/recall) and hand (Dom/NonDom). The input data should
    be a CSV file with the first 5 columns reserved for metadata (`source_file`,
    `participant_id`, `task`, `hand`, `start_time`), with all subsequent columns treated
    as numerical features.

    Both standard graphomotor features and custom feature columns added by users
    are supported. For a complete list of the 25 standard features available from
    the graphomotor extraction pipeline, see the
    [features module documentation](https://childmindresearch.github.io/graphomotor/graphomotor/features.html).

    Args:
        data: Path to CSV file containing features or pandas DataFrame. Input data
            should have the first 5 columns as metadata (`source_file`,
            `participant_id`, `task`, `hand`, `start_time`) followed by numerical
            feature columns.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.
            Can include any of the 25 standard graphomotor features (see module
            docstring) or custom feature columns added to the CSV file.

    Returns:
        The matplotlib Figure.
    """
    logger.debug("Starting feature distributions plot generation")

    plot_data, features, _ = plotting.prepare_feature_plot_data(data, features)

    hands = plot_data["hand"].unique()
    task_types = plot_data["task_type"].unique()

    colors = {
        (hand, task_type): plt.get_cmap("tab20")(i)
        for i, (hand, task_type) in enumerate(
            [(h, t) for h in hands for t in task_types]
        )
    }

    fig, axes = plotting.init_feature_subplots(len(features))
    for i, feature in enumerate(features):
        ax = axes[i]

        for hand in hands:
            for task_type in task_types:
                subset = plot_data[
                    (plot_data["hand"] == hand) & (plot_data["task_type"] == task_type)
                ]
                sns.kdeplot(
                    data=subset,
                    x=feature,
                    fill=True,
                    cut=0,
                    alpha=0.6,
                    color=colors[(hand, task_type)],
                    label=f"{hand} - {task_type.capitalize()}",
                    ax=ax,
                )

        display_name = plotting.format_feature_name(feature)
        ax.set_title(display_name)
        ax.set_xlabel(display_name)
        ax.set_ylabel("Density")
        ax.legend(title="Hand - Task Type")
        ax.grid(alpha=0.3)

    plotting.hide_extra_axes(axes=axes, num_subplots=len(features))

    plt.tight_layout()
    plt.suptitle(
        "Feature Distributions across Task Types and Hands",
        y=1.01,
        fontsize=10 + len(axes) // 2,
    )

    if output_path:
        plotting.save_figure(
            figure=fig, output_path=output_path, filename="feature_distributions"
        )
    else:
        logger.debug("Feature distributions plot generated but not saved")

    return fig


def plot_feature_trends(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> plt.Figure:
    """Plot lineplots to compare feature values across conditions per participant.

    This function creates line plots displaying feature progression across task
    sequences with individual participant trajectories and group means. The input
    data should be a CSV file with the first 5 columns reserved for metadata
    (`source_file`, `participant_id`, `task`, `hand`, `start_time`), with all subsequent
    columns treated as numerical features.

    Both standard graphomotor features and custom feature columns added by users
    are supported. For a complete list of the 25 standard features available from
    the graphomotor extraction pipeline, see the
    [features module documentation](https://childmindresearch.github.io/graphomotor/graphomotor/features.html).

    Args:
        data: Path to CSV file containing features or pandas DataFrame. Input data
            should have the first 5 columns as metadata (`source_file`,
            `participant_id`, `task`, `hand`, `start_time`) followed by numerical
            feature columns.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.
            Can include any of the 25 standard graphomotor features (see module
            docstring) or custom feature columns added to the CSV file.

    Returns:
        The matplotlib Figure.
    """
    logger.debug("Starting feature trends plot generation")

    plot_data, features, tasks = plotting.prepare_feature_plot_data(data, features)
    logger.debug(f"Plotting trends across {len(tasks)} tasks")

    fig, axes = plotting.init_feature_subplots(len(features))
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.lineplot(
            data=plot_data,
            x="task_order",
            y=feature,
            hue="hand",
            units="participant_id",
            estimator=None,
            alpha=0.2,
            linewidth=0.5,
            legend=False,
            ax=ax,
        )
        sns.lineplot(
            data=plot_data,
            x="task_order",
            y=feature,
            hue="hand",
            estimator="mean",
            errorbar=None,
            linewidth=2,
            marker="o",
            markersize=4,
            ax=ax,
        )
        display_name = plotting.format_feature_name(feature)
        ax.set_title(display_name)
        ax.set_ylabel(display_name)
        ax.set_xlabel("Task")
        ax.set_xticks(list(range(1, len(tasks) + 1)))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend(title="Hand")
        ax.grid(alpha=0.3)

    plotting.hide_extra_axes(axes=axes, num_subplots=len(features))

    plt.tight_layout()
    plt.suptitle(
        "Feature Trends across Tasks and Hands", y=1.01, fontsize=10 + len(axes) // 2
    )

    if output_path:
        plotting.save_figure(
            figure=fig, output_path=output_path, filename="feature_trends"
        )
    else:
        logger.debug("Feature trends plot generated but not saved")

    return fig


def plot_feature_boxplots(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> plt.Figure:
    """Plot boxplots to compare feature distributions across conditions.

    This function creates box-and-whisker plots comparing feature distributions
    across different tasks and hand conditions. The input data should be a CSV
    file with the first 5 columns reserved for metadata (`source_file`,
    `participant_id`, `task`, `hand`, `start_time`), with all subsequent columns treated
    as numerical features.

    Both standard graphomotor features and custom feature columns added by users
    are supported. For a complete list of the 25 standard features available from
    the graphomotor extraction pipeline, see the
    [features module documentation](https://childmindresearch.github.io/graphomotor/graphomotor/features.html).

    Args:
        data: Path to CSV file containing features or pandas DataFrame. Input data
            should have the first 5 columns as metadata (`source_file`,
            `participant_id`, `task`, `hand`, `start_time`) followed by numerical
            feature columns.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.
            Can include any of the 25 standard graphomotor features (see module
            docstring) or custom feature columns added to the CSV file.

    Returns:
        The matplotlib Figure.
    """
    logger.debug("Starting feature boxplots generation")

    plot_data, features, tasks = plotting.prepare_feature_plot_data(data, features)
    logger.debug(f"Creating boxplots across {len(tasks)} tasks")

    fig, axes = plotting.init_feature_subplots(len(features))
    for i, feature in enumerate(features):
        ax = axes[i]

        # Suppress seaborn's internal deprecation warning about 'vert' parameter
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                message="vert: bool will be deprecated.*",
            )
            sns.boxplot(
                data=plot_data,
                x="task",
                y=feature,
                hue="hand",
                order=tasks,
                ax=ax,
            )

        display_name = plotting.format_feature_name(feature)
        ax.set_title(display_name)
        ax.set_ylabel(display_name)
        ax.set_xlabel("Task")
        ax.set_xticks(list(range(len(tasks))))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend(title="Hand")
        ax.grid(alpha=0.3)

    plotting.hide_extra_axes(axes=axes, num_subplots=len(features))

    plt.tight_layout()
    plt.suptitle(
        "Feature Boxplots across Tasks and Hands", y=1.01, fontsize=10 + len(axes) // 2
    )

    if output_path:
        plotting.save_figure(
            figure=fig, output_path=output_path, filename="feature_boxplots"
        )
    else:
        logger.debug("Feature boxplots generated but not saved")

    return fig


def plot_feature_clusters(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> plt.Figure:
    """Plot clustered heatmap of standardized feature values across conditions.

    This function creates a hierarchically clustered heatmap that visualizes the median
    feature values across conditions. Values are z-score standardized across features to
    allow comparison when features are on different scales. Both features and
    conditions are hierarchically clustered to highlight groups of similar feature
    response patterns and conditions that elicit similar profiles.

    The input data should be a CSV file with the first 5 columns reserved for metadata
    (`source_file`, `participant_id`, `task`, `hand`, `start_time`), with all subsequent
    columns treated as numerical features. Both standard graphomotor features and custom
    feature columns added by users are supported. For a complete list of the 25
    standard features available from the graphomotor extraction pipeline, see the
    [features module documentation](https://childmindresearch.github.io/graphomotor/graphomotor/features.html).

    Args:
        data: Path to CSV file containing features or pandas DataFrame. Input data
            should have the first 5 columns as metadata (`source_file`,
            `participant_id`, `task`, `hand`, `start_time`) followed by numerical
            feature columns.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.
            Can include any of the 25 standard graphomotor features (see module
            docstring) or custom feature columns added to the CSV file.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If less than 2 features are provided.
    """
    logger.debug("Starting feature clusters heatmap generation")

    plot_data, features, _ = plotting.prepare_feature_plot_data(data, features)

    if len(features) < 2:
        error_msg = (
            f"At least 2 features required for clustered heatmap, got {len(features)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    plot_data["condition"] = plot_data["task"] + "_" + plot_data["hand"]

    condition_medians = plot_data.groupby("condition")[features].median()

    heatmap_data = condition_medians.T
    logger.debug(f"Heatmap data shape: {heatmap_data.shape} for (features, conditions)")

    width = max(10, len(heatmap_data.columns) * 0.8)
    height = max(6, len(heatmap_data.index) * 0.3)

    grid = sns.clustermap(
        heatmap_data,
        z_score=0,
        figsize=(width, height),
        cbar_kws={
            "label": "z-score",
            "location": "bottom",
            "orientation": "horizontal",
        },
        cbar_pos=(0.025, 0.93, 0.1 + 0.001 * width, 0.02 + 0.001 * height),
        center=0,
        cmap="coolwarm",
        linewidths=0.1,
        linecolor="black",
    )

    grid.figure.suptitle(
        "Feature Clusters Across Conditions",
        fontsize=14,
        y=1.01,
    )
    grid.ax_heatmap.set_xlabel("Condition")
    grid.ax_heatmap.set_ylabel("Feature")
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0)
    grid.ax_heatmap.set_xticklabels(
        grid.ax_heatmap.get_xticklabels(), rotation=45, ha="right"
    )

    if output_path:
        plotting.save_figure(
            figure=grid.figure, output_path=output_path, filename="feature_clusters"
        )
    else:
        logger.debug("Feature clusters heatmap generated but not saved")

    return grid.figure
