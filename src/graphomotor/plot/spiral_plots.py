"""Spiral visualization functions for quality control and data inspection.

This module provides plotting functions for visualizing spiral drawing trajectories from
CSV files. The functions support both single spiral plotting and batch processing for
quality control purposes.

Plot Types
----------
- **Single spiral plots**: Visualize individual spiral drawings with optional
  reference spiral overlay and color-coded line segments.
- **Batch spiral plots**: Visualize multiple spiral drawings organized by metadata
  with configurable subplot density and arrangement.
"""

import pathlib

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import center_spiral, plotting

matplotlib.use("agg")  # prevent interactive matplotlib
logger = config.get_logger()


def _plot_spiral(
    ax: plt.Axes,
    spiral: models.Spiral,
    centered_ref: np.ndarray,
    include_reference: bool = False,
    color_segments: bool = False,
    is_batch: bool = False,
) -> None:
    """Plot a spiral on a given matplotlib Axes.

    Args:
        ax: Matplotlib Axes to plot on.
        spiral: Spiral object containing drawing data and metadata.
        centered_ref: Pre-computed centered reference spiral coordinates.
        include_reference: If True, overlays the reference spiral for comparison.
        color_segments: If True, colors each line segment with distinct colors.
        is_batch: If True, formats for batch mode (no legend, no axis labels,
            smaller font).
    """
    centered_spiral = center_spiral.center_spiral(spiral)
    x_coords = centered_spiral.data["x"].values
    y_coords = centered_spiral.data["y"].values
    line_numbers = centered_spiral.data["line_number"].values

    if color_segments:
        unique_line_numbers = np.unique(line_numbers)
        color_indices = np.linspace(0, 1, len(unique_line_numbers))

        if len(unique_line_numbers) <= 10:
            colors = plt.get_cmap("tab10")(color_indices)
        elif len(unique_line_numbers) <= 20:
            colors = plt.get_cmap("tab20")(color_indices)
        else:
            colors = plt.get_cmap("viridis")(color_indices)

        color_map = dict(zip(unique_line_numbers, colors))

        for i in range(len(x_coords) - 1):
            current_line = line_numbers[i]
            label = (
                f"Line {int(current_line)}"
                if i == 0 or line_numbers[i - 1] != current_line
                else None
            )
            ax.plot(
                [x_coords[i], x_coords[i + 1]],
                [y_coords[i], y_coords[i + 1]],
                color=color_map[current_line],
                linewidth=2 if not is_batch else 1,
                alpha=0.8,
                label=label,
            )
    else:
        ax.plot(
            x_coords,
            y_coords,
            "tab:blue",
            linewidth=1.5,
            alpha=0.8,
            label="Drawn spiral",
        )

    ax.plot(
        centered_ref[0, 0],
        centered_ref[0, 1],
        "go",
        markersize=12 if not is_batch else 6,
        label="Start" if not is_batch else None,
        zorder=5,
        alpha=0.3,
    )
    ax.plot(
        centered_ref[-1, 0],
        centered_ref[-1, 1],
        "ro",
        markersize=12 if not is_batch else 6,
        label="End" if not is_batch else None,
        zorder=5,
        alpha=0.3,
    )

    if include_reference:
        ax.plot(
            centered_ref[:, 0],
            centered_ref[:, 1],
            "k--",
            linewidth=6 if not is_batch else 3,
            alpha=0.15,
            label="Reference spiral" if not is_batch else None,
        )

    participant_id, task, hand, start_time = plotting.extract_spiral_metadata(spiral)

    ax.set_title(
        label=f"ID: {participant_id}\n{task} - {hand}\n{start_time}",
        fontsize=8 if is_batch else 14,
    )

    ax.set_aspect("equal")

    if not is_batch:
        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_single_spiral(
    data: str | pathlib.Path | models.Spiral,
    output_path: str | pathlib.Path | None = None,
    include_reference: bool = False,
    color_segments: bool = False,
    spiral_config: config.SpiralConfig | None = None,
) -> plt.Figure:
    """Plot a single spiral drawing with optional reference spiral and color coding.

    This function creates a visualization of an individual spiral drawing trajectory.
    The spiral can be colored with distinct segments for better visualization of
    drawing progression, and optionally overlaid with a reference spiral for
    comparison.

    Args:
        data: Path to CSV file containing spiral data, or a loaded Spiral object.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        include_reference: If True, overlays the reference spiral for comparison.
        color_segments: If True, colors each line segment with distinct colors.
        spiral_config: Configuration for reference spiral generation. If None,
            uses default configuration.

    Returns:
        The matplotlib Figure object.

    Raises:
        ValueError: If the input data is invalid or cannot be loaded.
    """
    logger.debug("Starting single spiral plot generation")

    if isinstance(data, (str, pathlib.Path)):
        try:
            spiral = reader.load_spiral(data)
        except Exception as e:
            error_msg = f"Failed to load spiral data from {data}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    elif isinstance(data, models.Spiral):
        spiral = data
    else:
        error_msg = f"Invalid data type: {type(data)}. Expected str, Path, or Spiral."
        logger.error(error_msg)
        raise ValueError(error_msg)

    centered_ref = plotting.get_reference_spiral(spiral_config)

    fig, ax = plt.subplots(figsize=(10, 10))

    _plot_spiral(
        ax=ax,
        spiral=spiral,
        centered_ref=centered_ref,
        include_reference=include_reference,
        color_segments=color_segments,
        is_batch=False,
    )

    plt.tight_layout()

    if output_path:
        participant_id, task, hand, _ = plotting.extract_spiral_metadata(spiral)
        filename = f"spiral_{participant_id}_{task}_{hand}"
        plotting.save_figure(figure=fig, output_path=output_path, filename=filename)

    return fig


def plot_batch_spirals(
    data: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    include_reference: bool = False,
    color_segments: bool = False,
    spiral_config: config.SpiralConfig | None = None,
) -> plt.Figure:
    """Plot multiple spirals in a batch using a structured grid layout.

    This function processes multiple spiral CSV files from a directory and creates
    a structured grid of spiral visualizations with rows for participant/hand
    combinations and columns for tasks.

    Args:
        data: Path to directory containing spiral CSV files.
        output_path: Optional directory where the figure will be saved. If None,
            the function only returns the figure without saving.
        include_reference: If True, overlays reference spirals for comparison.
        color_segments: If True, colors each line segment with distinct colors.
        spiral_config: Configuration for reference spiral generation. If None,
            uses default configuration.

    Returns:
        The matplotlib Figure object containing all spiral plots.

    Raises:
        ValueError: If the input directory doesn't exist.
    """
    logger.debug("Starting batch spiral plot generation")

    data = pathlib.Path(data)
    if not data.exists() or not data.is_dir():
        error_msg = f"Input path does not exist or is not a directory: {data}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        spirals = plotting.load_spirals_from_directory(data)
    except ValueError as e:
        logger.error(f"Failed to load any spirals from directory: {e}")
        raise

    if len(spirals) == 1:
        return plot_single_spiral(
            data=spirals[0],
            output_path=output_path,
            include_reference=include_reference,
            color_segments=color_segments,
            spiral_config=spiral_config,
        )

    centered_ref = plotting.get_reference_spiral(spiral_config)

    spiral_grid, participant_hand_combos, sorted_tasks = (
        plotting.index_spirals_by_metadata(spirals)
    )

    n_rows = len(participant_hand_combos)
    n_cols = len(sorted_tasks)

    fig, axes = plotting.create_grid_layout(n_rows, n_cols)

    for row_idx, (participant, hand) in enumerate(participant_hand_combos):
        for col_idx, task in enumerate(sorted_tasks):
            ax = axes[row_idx, col_idx]
            key = (participant, hand, task)

            if key in spiral_grid:
                spiral = spiral_grid[key]
                _plot_spiral(
                    ax=ax,
                    spiral=spiral,
                    centered_ref=centered_ref,
                    include_reference=include_reference,
                    color_segments=color_segments,
                    is_batch=True,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    alpha=0.5,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                ax.set_title("No Data", fontsize=8)

            plotting.add_grid_labels(ax, key, row_idx, col_idx)

    if output_path:
        filename = "batch_spirals"
        plotting.save_figure(figure=fig, output_path=output_path, filename=filename)

    return fig
