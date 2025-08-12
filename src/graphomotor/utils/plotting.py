"""Utility functions for plot module."""

import datetime
import pathlib

import numpy as np
import pandas as pd
from matplotlib import axes, figure
from matplotlib import pyplot as plt

from graphomotor.core import config, models

logger = config.get_logger()


def _load_features_dataframe(data: str | pathlib.Path | pd.DataFrame) -> pd.DataFrame:
    """Load and validate feature data from CSV file or DataFrame.

    Args:
        data: Path to CSV file containing features or pandas DataFrame.

    Returns:
        DataFrame with features data.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If data format is invalid.
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
    else:
        logger.debug(
            f"Using provided DataFrame with {len(data)} rows and "
            f"{len(data.columns)} columns"
        )
        return data.copy()


def _validate_features_dataframe(
    df: pd.DataFrame, requested_features: list[str] | None = None
) -> list[str]:
    """Validate feature DataFrame structure and return validated features.

    Leverages models.Spiral validation logic for metadata consistency. Uses the same
    validation patterns as the Spiral model.

    Args:
        df: DataFrame to validate.
        requested_features: User-specified feature names.

    Returns:
        List of validated feature names.

    Raises:
        ValueError: If DataFrame structure or metadata is invalid.
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
        logger.debug(f"Found {len(features)} feature columns")

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


def _add_task_metadata(features_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare DataFrame for plotting by adding task metadata.

    Args:
        features_df: The DataFrame containing feature data.

    Returns:
        A tuple containing the modified DataFrame and a list of task names.
    """
    # This is the standard order of tasks in the graphomotor protocol
    task_order = {
        "spiral_trace1": 1,
        "spiral_trace2": 2,
        "spiral_trace3": 3,
        "spiral_trace4": 4,
        "spiral_trace5": 5,
        "spiral_recall1": 6,
        "spiral_recall2": 7,
        "spiral_recall3": 8,
    }
    plot_data = features_df.copy()
    plot_data["task_order"] = plot_data["task"].map(task_order)
    plot_data["task_type"] = plot_data["task"].apply(
        lambda x: "trace" if "trace" in x else "recall"
    )
    return plot_data, list(task_order.keys())


def prepare_feature_plot_data(
    data: str | pathlib.Path | pd.DataFrame, features: list[str] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare data for plotting by loading, validating, and adding task metadata.

    Args:
        data: The input data (CSV file path or DataFrame).
        features: List of features to plot.

    Returns:
        A tuple containing the plot data, validated features, and task names.
    """
    features_df = _load_features_dataframe(data)
    features = _validate_features_dataframe(features_df, features)
    plot_data, tasks = _add_task_metadata(features_df)
    return plot_data, features, tasks


def init_feature_subplots(n_features: int) -> tuple[figure.Figure, list[axes.Axes]]:
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
    logger.debug(f"Created subplot grid for {n_features} features")
    return fig, flat_axes


def format_feature_name(feature: str) -> str:
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


def hide_extra_axes(axes: list[axes.Axes], n_features: int) -> None:
    """Hide any extra axes that are not used for plotting.

    Args:
        axes: List of matplotlib Axes objects.
        n_features: Number of features being plotted.
    """
    for extra_ax in axes[n_features:]:
        extra_ax.set_visible(False)


def save_figure(output_path: str | pathlib.Path, filename: str) -> None:
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
    logger.debug("Figure saved successfully")
