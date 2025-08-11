"""Feature visualization functions for Graphomotor."""

import datetime
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, figure
from matplotlib import pyplot as plt

from graphomotor.core import config, models

logger = config.get_logger()

TASK_ORDER = {
    "spiral_trace1": 1,
    "spiral_trace2": 2,
    "spiral_trace3": 3,
    "spiral_trace4": 4,
    "spiral_trace5": 5,
    "spiral_recall1": 6,
    "spiral_recall2": 7,
    "spiral_recall3": 8,
}


def _load_data(data: str | pathlib.Path | pd.DataFrame) -> pd.DataFrame:
    """Load and validate feature data from CSV file or DataFrame.

    Args:
        data: Path to CSV file containing features or pandas DataFrame

    Returns:
        DataFrame with features data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If data format is invalid
    """
    if isinstance(data, (str, pathlib.Path)):
        data_path = pathlib.Path(data)
        logger.debug(f"Loading feature data from CSV file: {data_path}")
        if not data_path.exists():
            error_msg = f"Features file not found: {data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        try:
            df = pd.read_csv(data_path)
            logger.debug(
                f"Successfully loaded {len(df)} rows and {len(df.columns)} "
                "columns from CSV"
            )
            return df
        except Exception as e:
            error_msg = f"Failed to read CSV file {data_path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    elif isinstance(data, pd.DataFrame):
        logger.debug(
            f"Using provided DataFrame with {len(data)} rows and "
            f"{len(data.columns)} columns"
        )
        return data.copy()
    else:
        error_msg = (
            f"Invalid data type: {type(data)}. "
            "Expected str, pathlib.Path, or pd.DataFrame"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _validate_features_dataframe(
    df: pd.DataFrame, requested_features: list[str] | None = None
) -> list[str]:
    """Validate feature DataFrame structure and return validated features.

    Leverages models.Spiral validation logic for metadata consistency. Uses the same
    validation patterns as the Spiral model.

    Args:
        df: DataFrame to validate
        requested_features: User-specified feature names

    Returns:
        List of validated feature names

    Raises:
        ValueError: If DataFrame structure or metadata is invalid
    """
    logger.debug("Validating features and metadata")

    try:
        models.Spiral.validate_dataframe(df)
    except ValueError as e:
        logger.error(str(e))
        raise

    required_metadata_cols = [
        "participant_id",
        "task",
        "hand",
    ]
    missing_metadata = [col for col in required_metadata_cols if col not in df.columns]
    if missing_metadata:
        error_msg = f"Required metadata columns missing: {missing_metadata}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("Metadata columns validation passed")

    for _, row in df.iterrows():
        metadata = {
            "id": str(row["participant_id"]),
            "hand": row["hand"],
            "task": row["task"],
        }
        try:
            models.Spiral.validate_metadata(metadata)
        except ValueError as e:
            logger.error(str(e))
            raise

    logger.debug("All metadata rows validation passed")

    if requested_features:
        features = requested_features
        logger.debug(f"Using {len(features)} user-specified features")
    else:
        features = df.iloc[:, 5:].columns.tolist()
        logger.debug(f"Auto-detected {len(features)} feature columns")

    if not features:
        error_msg = "No feature columns found to plot"
        logger.error(error_msg)
        raise ValueError(error_msg)

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        error_msg = f"Features not found in data: {missing_features}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(
        f"Feature validation completed successfully for {len(features)} features"
    )
    return features


def _add_task_metadata(features_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for plotting by adding task metadata.

    Args:
        features_df: The DataFrame containing feature data.

    Returns:
        DataFrame with task metadata added.
    """
    plot_data = features_df.copy()
    plot_data["task_order"] = plot_data["task"].map(TASK_ORDER)
    plot_data["task_type"] = plot_data["task"].apply(
        lambda x: "trace" if "trace" in x else "recall"
    )
    return plot_data


def _init_feature_subplots(n_features: int) -> tuple[figure.Figure, list[axes.Axes]]:
    """Create a grid of subplots sized for the number of features.

    Args:
        n_features: The number of features to plot.

    Returns:
        A tuple containing the figure and a list of axes.
    """
    n_rows = int(np.ceil(np.sqrt(n_features)))
    n_cols = int(np.ceil(n_features / n_rows))

    base_size = 6.0

    width = max(12, n_cols * base_size)
    height = max(8, n_rows * base_size * 0.75)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))

    flat_axes = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]

    return fig, flat_axes


def _format_feature_name(feature: str) -> str:
    """Format feature name for better display in plots.

    Args:
        feature: The feature name to format.

    Returns:
        The formatted feature name.
    """
    parts = feature.split("_")
    lines = []
    current_line = ""

    for part in parts:
        if current_line and len(current_line + "_" + part) > 21:
            lines.append(current_line)
            current_line = part
        else:
            current_line = current_line + "_" + part if current_line else part

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def _hide_extra_axes(axes: list[axes.Axes], n_features: int) -> None:
    """Hide any extra axes that are not used for plotting.

    Args:
        axes: List of matplotlib Axes objects.
        n_features: Number of features being plotted.
    """
    for extra_ax in axes[n_features:]:
        extra_ax.set_visible(False)


def _save_figure(output_path: str | pathlib.Path, filename: str) -> None:
    """Save the current matplotlib figure to file.

    Args:
        output_path: Path to the directory where the figure will be saved.
        filename: Base filename for the saved figure.
    """
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = (
        output_path
        / f"{filename}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    )
    logger.debug(f"Saving figure to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.debug(f"Figure saved successfully: {save_path}")


def plot_feature_distributions(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> figure.Figure:
    """Plot histograms for each feature grouped by task type and hand.

    Args:
        data: Path to CSV file containing features or pandas DataFrame.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.

    Returns:
        The matplotlib Figure.
    """
    logger.info("Starting feature distributions plot generation")

    features_df = _load_data(data)
    features = _validate_features_dataframe(features_df, features)

    plot_data = _add_task_metadata(features_df)
    hands = plot_data["hand"].unique()
    task_types = plot_data["task_type"].unique()

    logger.debug(f"Plot data contains {len(hands)} hand types: {hands}")
    logger.debug(f"Plot data contains {len(task_types)} task types: {task_types}")

    colors = {
        (hand, task_type): plt.get_cmap("tab20")(i)
        for i, (hand, task_type) in enumerate(
            [(h, t) for h in hands for t in task_types]
        )
    }

    fig, axes = _init_feature_subplots(len(features))
    logger.debug(f"Created subplot grid for {len(features)} features")

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

        display_name = _format_feature_name(feature)
        ax.set_title(display_name)
        ax.set_xlabel(display_name)
        ax.set_ylabel("Density")
        ax.legend(title="Hand - Task Type")
        ax.grid(alpha=0.3)

    _hide_extra_axes(axes, len(features))

    plt.tight_layout()
    plt.suptitle(
        "Feature Distributions across Task Types and Hands",
        y=1.01,
        fontsize=10 + len(axes) // 2,
    )

    if output_path:
        _save_figure(output_path, "feature_distributions")
        logger.info("Feature distributions plot saved successfully")
    else:
        logger.debug("Feature distributions plot generated but not saved")

    return fig


def plot_feature_trends(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> figure.Figure:
    """Plot lineplots to compare feature values across conditions per participant.

    Args:
        data: Path to CSV file containing features or pandas DataFrame.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.

    Returns:
        The matplotlib Figure.
    """
    logger.info("Starting feature trends plot generation")

    features_df = _load_data(data)
    features = _validate_features_dataframe(features_df, features)

    plot_data = _add_task_metadata(features_df)
    ordered_tasks = sorted(TASK_ORDER.keys(), key=lambda x: TASK_ORDER[x])
    logger.debug(f"Plotting trends across {len(ordered_tasks)} tasks")

    fig, axes = _init_feature_subplots(len(features))
    logger.debug(f"Created subplot grid for {len(features)} features")

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
        display_name = _format_feature_name(feature)
        ax.set_title(display_name)
        ax.set_ylabel(display_name)
        ax.set_xlabel("Task")
        ax.set_xticks(list(range(1, len(ordered_tasks) + 1)))
        ax.set_xticklabels(ordered_tasks, rotation=45, ha="right")
        ax.legend(title="Hand")
        ax.grid(alpha=0.3)

    _hide_extra_axes(axes, len(features))

    plt.tight_layout()
    plt.suptitle(
        "Feature Trends across Tasks and Hands", y=1.01, fontsize=10 + len(axes) // 2
    )

    if output_path:
        _save_figure(output_path, "feature_trends")
        logger.info("Feature trends plot saved successfully")
    else:
        logger.debug("Feature trends plot generated but not saved")

    return fig


def plot_feature_boxplots(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> figure.Figure:
    """Plot boxplots to compare feature distributions across conditions.

    Args:
        data: Path to CSV file containing features or pandas DataFrame.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.

    Returns:
        The matplotlib Figure.
    """
    logger.info("Starting feature boxplots generation")

    features_df = _load_data(data)
    features = _validate_features_dataframe(features_df, features)

    plot_data = _add_task_metadata(features_df)
    ordered_tasks = sorted(TASK_ORDER.keys(), key=lambda x: TASK_ORDER[x])

    logger.debug(f"Creating boxplots across {len(ordered_tasks)} tasks")

    fig, axes = _init_feature_subplots(len(features))
    logger.debug(f"Created subplot grid for {len(features)} features")

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            data=plot_data,
            x="task",
            y=feature,
            hue="hand",
            order=ordered_tasks,
            ax=ax,
        )
        display_name = _format_feature_name(feature)
        ax.set_title(display_name)
        ax.set_ylabel(display_name)
        ax.set_xlabel("Task")
        ax.set_xticks(list(range(len(ordered_tasks))))
        ax.set_xticklabels(ordered_tasks, rotation=45, ha="right")
        ax.legend(title="Hand")
        ax.grid(alpha=0.3)

    _hide_extra_axes(axes, len(features))

    plt.tight_layout()
    plt.suptitle(
        "Feature Boxplots across Tasks and Hands", y=1.01, fontsize=10 + len(axes) // 2
    )

    if output_path:
        _save_figure(output_path, "feature_boxplots")
        logger.info("Feature boxplots saved successfully")
    else:
        logger.debug("Feature boxplots generated but not saved")

    return fig


def plot_feature_clusters(
    data: str | pathlib.Path | pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    features: list[str] | None = None,
) -> figure.Figure:
    """Plot clustered heatmap of standardized feature values across conditions.

    This function creates a hierarchically clustered heatmap that visualizes the median
    feature values across conditions. Values are z-score standardized across features to
    allow comparison when features are on different scales. Both features and
    conditions are hierarchically clustered to highlight groups of similar feature
    response patterns and conditions that elicit similar profiles.

    Args:
        data: Path to CSV file containing features or pandas DataFrame.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        features: List of specific features to plot, if None plots all features.

    Returns:
        The matplotlib Figure.
    """
    logger.info("Starting feature clusters heatmap generation")

    features_df = _load_data(data)
    features = _validate_features_dataframe(features_df, features)

    plot_data = _add_task_metadata(features_df)

    plot_data["condition"] = plot_data["task"] + "_" + plot_data["hand"]

    condition_medians = plot_data.groupby("condition")[features].median()

    heatmap_data = condition_medians.T

    logger.debug(f"Heatmap data shape: {heatmap_data.shape} (features x conditions)")

    width = max(10, len(heatmap_data.columns) * 0.8)
    height = max(6, len(heatmap_data.index) * 0.3)

    g = sns.clustermap(
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
        cmap="RdBu_r",
        linewidths=0.1,
        linecolor="black",
    )

    g.figure.suptitle(
        "Feature Clusters Across Conditions",
        fontsize=14,
        y=1.01,
    )
    g.ax_heatmap.set_xlabel("Condition")
    g.ax_heatmap.set_ylabel("Feature")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha="right"
    )

    if output_path:
        _save_figure(output_path, "feature_clusters")
        logger.info("Feature clusters heatmap saved successfully")
    else:
        logger.debug("Feature clusters heatmap generated but not saved")

    return g.figure
