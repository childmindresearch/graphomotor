"""Utility functions for plot module."""

import datetime
import pathlib

import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import center_spiral, generate_reference_spiral

logger = config.get_logger()

# This is the standard order of tasks in the Graphomotor protocol
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

# Grid layout configuration for batch spiral plotting
GRID_CONFIG = {
    "subplot_size": 3,
    "min_figure_size": 12,
    "row_extra_spacing": 0.7,
    "row_normal_spacing": 0.35,
    "column_extra_spacing": 0.3,
    "column_normal_spacing": 0.1,
}


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
    plot_data = features_df.copy()
    plot_data["task_order"] = plot_data["task"].map(TASK_ORDER)
    plot_data["task_type"] = plot_data["task"].apply(
        lambda x: "trace" if "trace" in x else "recall"
    )
    return plot_data, list(TASK_ORDER.keys())


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


def extract_spiral_metadata(spiral: models.Spiral) -> tuple[str, str, str, str]:
    """Extract common metadata from a spiral object.

    Args:
        spiral: Spiral object containing metadata.

    Returns:
        Tuple of (participant_id, task, hand, start_time).
    """
    participant_id = str(spiral.metadata.get("id"))
    task = str(spiral.metadata.get("task"))
    hand = str(spiral.metadata.get("hand"))
    start_time = str(spiral.metadata.get("start_time")).split(".", 1)[0]
    return participant_id, task, hand, start_time


def get_reference_spiral(
    spiral_config: config.SpiralConfig | None = None,
) -> np.ndarray:
    """Get the reference spiral for a given spiral config.

    Args:
        spiral_config: Configuration for reference spiral generation.

    Returns:
        The centered reference spiral as a NumPy array.
    """
    if spiral_config is None:
        spiral_config = config.SpiralConfig()
    ref_spiral = generate_reference_spiral.generate_reference_spiral(spiral_config)
    return center_spiral.center_spiral(ref_spiral)


def load_spirals_from_directory(
    input_dir: pathlib.Path,
) -> list[models.Spiral]:
    """Load spiral CSV files from a directory.

    Args:
        input_dir: Directory path to search for CSV files.

    Returns:
        List of loaded Spiral objects.

    Raises:
        ValueError: If no CSV files are found or if no valid spirals could be loaded.
    """
    csv_files = list(input_dir.rglob("*.csv"))
    if not csv_files:
        error_msg = f"No CSV files found in directory: {input_dir}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Found {len(csv_files)} CSV files to process")

    spirals: list[models.Spiral] = []
    failed_files: list[str] = []
    for csv_file in csv_files:
        try:
            spiral = reader.load_spiral(csv_file)
            spirals.append(spiral)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
            failed_files.append(str(csv_file))

    if not spirals:
        error_msg = "Could not load any valid spiral files"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files")

    return spirals


def index_spirals_by_metadata(
    spirals: list[models.Spiral],
) -> tuple[
    dict[tuple[str, str, str], models.Spiral],
    list[tuple[str, str]],
    list[str],
]:
    """Index spirals by (participant, hand, task) for grid plotting.

    Constructs:
      - spiral_grid: dict mapping (participant_id, hand, task) to Spiral.
      - participant_hand_combos: ordered list of (participant_id, hand) tuples.
      - sorted_tasks: list of task names sorted according to TASK_ORDER.

    Args:
        spirals: Iterable of Spiral objects to index.

    Returns:
        A tuple of (spiral_grid, participant_hand_combos, sorted_tasks).
    """
    spiral_grid: dict[tuple[str, str, str], models.Spiral] = {}
    participants: set[str] = set()
    hands: set[str] = set()
    tasks: set[str] = set()

    for spiral in spirals:
        participant_id, task, hand, _ = extract_spiral_metadata(spiral)
        participants.add(participant_id)
        hands.add(hand)
        tasks.add(task)
        key = (participant_id, hand, task)
        spiral_grid[key] = spiral

    sorted_participants = sorted(participants)
    sorted_hands = sorted(hands)
    sorted_tasks = sorted(tasks, key=lambda t: TASK_ORDER.get(t, 9))

    participant_hand_combos: list[tuple[str, str]] = []
    for participant in sorted_participants:
        for hand in sorted_hands:
            participant_hand_combos.append((participant, hand))

    return spiral_grid, participant_hand_combos, sorted_tasks


def _calculate_spacing_ratios(num_subplots: int, is_width: bool = False) -> list[float]:
    """Calculate spacing ratios for grid layout with optional extra spacing.

    This function constructs a list of ratios used by GridSpec to size subplots and the
    spacing between them. For each subplot a ratio of 1.0 is appended. Spacing entries
    are inserted only between subplots (not after the last one) and differ depending on
    whether widths or heights are being computed.

    - For widths (is_width=True):
        - After the 5th column (index 4) an extra column spacing is inserted using
          GRID_CONFIG['column_extra_spacing'].
        - Otherwise, normal column spacing is inserted using
          GRID_CONFIG['column_normal_spacing'].

    - For heights (is_width=False):
        - After every even-numbered row (i.e., rows 2, 4, ...) an extra row spacing is
          inserted using GRID_CONFIG['row_extra_spacing'].
        - Otherwise, normal row spacing is inserted using
          GRID_CONFIG['row_normal_spacing'].

    The resulting list alternates subplot ratio and spacing ratio where applicable,
    e.g. [1.0, spacing, 1.0, spacing, 1.0, ...].

    Args:
        num_subplots: Number of subplots (rows or columns).
        is_width: If True, use column spacing constants; otherwise, use row spacing.

    Returns:
        List of ratios for gridspec (subplots + spacing).
    """
    ratios = []
    for i in range(num_subplots):
        ratios.append(1.0)
        if i < num_subplots - 1:
            if is_width:
                if i == 4:
                    ratios.append(GRID_CONFIG["column_extra_spacing"])
                else:
                    ratios.append(GRID_CONFIG["column_normal_spacing"])
            else:
                if (i + 1) % 2 == 0:
                    ratios.append(GRID_CONFIG["row_extra_spacing"])
                else:
                    ratios.append(GRID_CONFIG["row_normal_spacing"])
    return ratios


def create_grid_layout(n_rows: int, n_cols: int) -> tuple[plt.Figure, np.ndarray]:
    """Create figure and grid layout with proper spacing.

    This function creates a matplotlib figure with a `GridSpec` layout that includes
    proper spacing between subplots. The `axes` array maps subplot positions to
    gridspec positions, where `grid_col` and `grid_row` are incremented by 2 to skip
    the spacing columns and rows in the gridspec.

    Args:
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.

    Returns:
        Tuple of (figure, axes_array).
    """
    fig_width = max(
        GRID_CONFIG["min_figure_size"], n_cols * GRID_CONFIG["subplot_size"]
    )
    fig_height = max(
        GRID_CONFIG["min_figure_size"], n_rows * GRID_CONFIG["subplot_size"]
    )

    height_ratios = _calculate_spacing_ratios(n_rows, is_width=False)
    width_ratios = _calculate_spacing_ratios(n_cols, is_width=True)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        len(height_ratios),
        len(width_ratios),
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=0,
        wspace=0,
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    grid_row = 0
    for i in range(n_rows):
        grid_col = 0
        for j in range(n_cols):
            axes[i, j] = fig.add_subplot(gs[grid_row, grid_col])
            grid_col += 2
        grid_row += 2

    return fig, axes


def add_grid_labels(
    ax: plt.Axes, key: tuple[str, str, str], row_idx: int, col_idx: int
) -> None:
    """Add column and row labels to grid subplots.

    Args:
        ax: Matplotlib axes to label.
        key: Tuple containing (participant, hand, task) for subplot labeling.
        row_idx: Row index (0-based).
        col_idx: Column index (0-based).
    """
    participant, hand, task = key
    if row_idx == 0:
        ax.text(
            0.5,
            1.3,
            f"Task: {task}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    if col_idx == 0:
        ax.set_ylabel(
            f"ID: {participant}\nHand: {hand}",
            fontsize=9,
            fontweight="bold",
        )


def init_feature_subplots(n_features: int) -> tuple[plt.Figure, list[plt.Axes]]:
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


def hide_extra_axes(axes: list[plt.Axes], num_subplots: int) -> None:
    """Hide any extra axes that are not used for plotting.

    Args:
        axes: List of matplotlib Axes objects.
        num_subplots: Number of subplots being created.
    """
    for extra_ax in axes[num_subplots:]:
        extra_ax.set_visible(False)


def save_figure(
    figure: plt.Figure, output_path: str | pathlib.Path, filename: str
) -> None:
    """Save the current matplotlib figure to file.

    Args:
        figure: Matplotlib figure to save.
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
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.debug("Figure saved successfully")
    plt.close()
