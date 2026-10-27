"""Utility functions for Alphabet and DSYM stroke segmentation."""

from typing import List, Optional

import pandas as pd

from graphomotor.core import models


def segment_strokes(
    data: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_rows: int,
    n_cols: int,
    labels: Optional[List[str]] = None,
) -> models.Grid:
    """Segment drawing data into strokes and assign them to grid cells.

    Groups the data by line_number to create individual Stroke objects. Each
    stroke is assigned to a GridCell based on its centroid (mean x, mean y).
    The grid is constructed via Grid.from_bbox with default padding so that
    centroids on the outermost edge are captured.

    Args:
        data: DataFrame containing drawing data with at least line_number,
            x, y, and seconds columns.
        x_min: Left boundary of the grid bounding box.
        x_max: Right boundary of the grid bounding box.
        y_min: Bottom boundary of the grid bounding box.
        y_max: Top boundary of the grid bounding box.
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        labels: Optional list of labels for each cell in row-major order.
            Must have length n_rows * n_cols if provided.

    Returns:
        A Grid populated with strokes assigned to their matching cells.
    """
    grid = models.Grid.from_bbox(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_rows=n_rows,
        n_cols=n_cols,
        labels=labels,
    )

    for line_number, group in data.groupby("line_number"):
        stroke = models.Stroke(
            points=group.reset_index(drop=True),
            line_number=int(line_number),
        )

        centroid_x = group["x"].mean()
        centroid_y = group["y"].mean()
        cell_index = grid.get_cell_for_point(centroid_x, centroid_y)

        if cell_index != -1:
            grid.cells[cell_index].strokes.append(stroke)

    return grid
