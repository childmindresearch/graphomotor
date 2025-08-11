"""Spiral plotting module for quality control visualization."""

import pathlib
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import generate_reference_spiral


def plot_single_spiral(
    input_path: Union[str, pathlib.Path],
    output_path: Optional[Union[str, pathlib.Path]] = None,
    include_reference: bool = False,
    color_by_segment: bool = False,
    spiral_config: Optional[config.SpiralConfig] = None,
    **kwargs: Any,
) -> None:
    """Plot a single spiral with optional reference overlay and segment coloring.

    Args:
        input_path: Path to the spiral CSV file.
        output_path: Path to save the plot. If None, auto-generates based on input.
        include_reference: Whether to overlay the reference spiral.
        color_by_segment: Whether to use different colors for line segments.
        spiral_config: Custom spiral configuration. Uses defaults if None.
        **kwargs: Additional arguments passed to matplotlib.

    Raises:
        IOError: If the input file cannot be read.
        ValueError: If the spiral data is invalid.
    """
    # Load spiral data
    spiral = reader.load_spiral(input_path)
    
    # Generate output path if not provided
    if output_path is None:
        input_path_obj = pathlib.Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_plot.png"
    
    output_path = pathlib.Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract x, y coordinates
    x_coords = spiral.data["x"].values
    y_coords = spiral.data["y"].values
    
    # Plot spiral data
    if color_by_segment:
        # Create color map for segments
        n_points = len(x_coords)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points - 1))
        
        # Plot each segment with different color
        for i in range(n_points - 1):
            ax.plot(
                x_coords[i:i+2], 
                y_coords[i:i+2], 
                color=colors[i], 
                linewidth=1.5
            )
    else:
        # Plot as single continuous line
        ax.plot(x_coords, y_coords, 'b-', linewidth=1.5, label='Drawn spiral')
    
    # Plot reference spiral if requested
    if include_reference:
        if spiral_config is None:
            spiral_config = config.SpiralConfig()
        
        reference_points = generate_reference_spiral.generate_reference_spiral(spiral_config)
        ax.plot(
            reference_points[:, 0], 
            reference_points[:, 1], 
            'r--', 
            linewidth=2, 
            alpha=0.7, 
            label='Reference spiral'
        )
    
    # Customize plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Spiral Plot - {spiral.metadata["id"]} - {spiral.metadata["task"]} - {spiral.metadata["hand"]}')
    ax.grid(True, alpha=0.3)
    
    # Add legend if reference is included
    if include_reference or not color_by_segment:
        ax.legend()
    
    # Apply any additional kwargs to the plot
    for key, value in kwargs.items():
        if hasattr(ax, f'set_{key}'):
            getattr(ax, f'set_{key}')(value)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_spiral_batch(
    input_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    group_by: Optional[str] = None,
    max_per_row: int = 4,
    include_reference: bool = False,
    color_by_segment: bool = False,
    spiral_config: Optional[config.SpiralConfig] = None,
    **kwargs: Any,
) -> None:
    """Plot multiple spirals with optional grouping and batch processing.

    Args:
        input_path: Path to directory containing spiral CSV files.
        output_path: Directory path to save plots.
        group_by: Group spirals by 'participant', 'task', 'hand', or None for individual plots.
        max_per_row: Maximum number of subplots per row when group_by is used.
        include_reference: Whether to overlay reference spirals.
        color_by_segment: Whether to use different colors for line segments.
        spiral_config: Custom spiral configuration. Uses defaults if None.
        **kwargs: Additional arguments passed to matplotlib.

    Raises:
        IOError: If input directory doesn't exist or no CSV files found.
        ValueError: If group_by parameter is invalid.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    
    # Validate inputs
    if not input_path.is_dir():
        raise IOError(f"Input path is not a directory: {input_path}")
    
    valid_group_by = {'participant', 'task', 'hand', None}
    if group_by not in valid_group_by:
        raise ValueError(f"group_by must be one of {valid_group_by}, got: {group_by}")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        raise IOError(f"No CSV files found in directory: {input_path}")
    
    # Load all spirals with progress bar
    spirals = []
    for csv_file in tqdm(csv_files, desc="Loading spiral files"):
        try:
            spiral = reader.load_spiral(csv_file)
            spirals.append(spiral)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue
    
    if not spirals:
        raise IOError("No valid spiral files could be loaded")
    
    # Handle different grouping options
    if group_by is None:
        # Individual plots for each spiral
        _plot_individual_spirals(
            spirals, output_path, include_reference, color_by_segment, spiral_config, **kwargs
        )
    else:
        # Grouped plots
        _plot_grouped_spirals(
            spirals, output_path, group_by, max_per_row, include_reference, 
            color_by_segment, spiral_config, **kwargs
        )


def _plot_individual_spirals(
    spirals: List[models.Spiral],
    output_path: pathlib.Path,
    include_reference: bool,
    color_by_segment: bool,
    spiral_config: Optional[config.SpiralConfig],
    **kwargs: Any,
) -> None:
    """Plot each spiral individually."""
    for spiral in tqdm(spirals, desc="Creating individual plots"):
        # Generate output filename
        filename = f"{spiral.metadata['id']}_{spiral.metadata['task']}_{spiral.metadata['hand']}_plot.png"
        plot_path = output_path / filename
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot spiral
        _plot_spiral_on_axis(
            ax, spiral, include_reference, color_by_segment, spiral_config, **kwargs
        )
        
        # Save plot
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def _plot_grouped_spirals(
    spirals: List[models.Spiral],
    output_path: pathlib.Path,
    group_by: str,
    max_per_row: int,
    include_reference: bool,
    color_by_segment: bool,
    spiral_config: Optional[config.SpiralConfig],
    **kwargs: Any,
) -> None:
    """Plot spirals grouped by the specified attribute."""
    # Group spirals by the specified attribute
    groups: Dict[str, List[models.Spiral]] = {}
    for spiral in spirals:
        # Map 'participant' to 'id' for consistency with metadata
        metadata_key = 'id' if group_by == 'participant' else group_by
        key = str(spiral.metadata[metadata_key])  # Convert to string for consistency
        if key not in groups:
            groups[key] = []
        groups[key].append(spiral)
    
    # Create plots for each group
    for group_key, group_spirals in tqdm(groups.items(), desc=f"Creating grouped plots by {group_by}"):
        n_spirals = len(group_spirals)
        
        # Calculate subplot layout
        n_cols = min(max_per_row, n_spirals)
        n_rows = (n_spirals + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        
        # Handle single subplot case
        if n_spirals == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each spiral in the group
        for i, spiral in enumerate(group_spirals):
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            _plot_spiral_on_axis(
                ax, spiral, include_reference, color_by_segment, spiral_config, 
                show_legend=(i == 0), **kwargs
            )
        
        # Hide empty subplots
        if isinstance(axes, (list, np.ndarray)):
            for i in range(n_spirals, len(axes)):
                axes[i].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'Spirals grouped by {group_by}: {group_key}', fontsize=16)
        
        # Generate output filename
        safe_key = "".join(c for c in group_key if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"spirals_{group_by}_{safe_key}_plot.png"
        plot_path = output_path / filename
        
        # Save plot
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def _plot_spiral_on_axis(
    ax: plt.Axes,
    spiral: models.Spiral,
    include_reference: bool,
    color_by_segment: bool,
    spiral_config: Optional[config.SpiralConfig],
    show_legend: bool = True,
    **kwargs: Any,
) -> None:
    """Helper function to plot a spiral on a given axis."""
    # Extract coordinates
    x_coords = spiral.data["x"].values
    y_coords = spiral.data["y"].values
    
    # Plot spiral data
    if color_by_segment:
        # Create color map for segments
        n_points = len(x_coords)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points - 1))
        
        # Plot each segment with different color
        for i in range(n_points - 1):
            ax.plot(
                x_coords[i:i+2], 
                y_coords[i:i+2], 
                color=colors[i], 
                linewidth=1.5
            )
    else:
        # Plot as single continuous line
        ax.plot(x_coords, y_coords, 'b-', linewidth=1.5, label='Drawn spiral')
    
    # Plot reference spiral if requested
    if include_reference:
        if spiral_config is None:
            spiral_config = config.SpiralConfig()
        
        reference_points = generate_reference_spiral.generate_reference_spiral(spiral_config)
        ax.plot(
            reference_points[:, 0], 
            reference_points[:, 1], 
            'r--', 
            linewidth=2, 
            alpha=0.7, 
            label='Reference spiral'
        )
    
    # Customize plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'{spiral.metadata["id"]} - {spiral.metadata["task"]} - {spiral.metadata["hand"]}')
    ax.grid(True, alpha=0.3)
    
    # Add legend if requested and appropriate
    if show_legend and (include_reference or not color_by_segment):
        ax.legend()
    
    # Apply any additional kwargs to the plot
    for key, value in kwargs.items():
        if hasattr(ax, f'set_{key}'):
            getattr(ax, f'set_{key}')(value)