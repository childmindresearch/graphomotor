"""Test cases for the Grid model."""

from typing import Dict

import pytest

from graphomotor.core import models


@pytest.fixture
def bbox_kwargs() -> Dict[str, float]:
    """Bounding box keyword arguments for a 10x10 grid area."""
    return {"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0}


@pytest.fixture
def grid_2x2(bbox_kwargs: Dict[str, float]) -> models.Grid:
    """Create a labeled 2x2 grid over a 10x10 bounding box."""
    return models.Grid.from_bbox(
        n_rows=2, n_cols=2, labels=["TL", "TR", "BL", "BR"], **bbox_kwargs
    )


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(1, 1), (2, 2), (3, 3), (2, 3)],
    ids=["1x1", "2x2", "3x3", "2x3"],
)
def test_grid_structure(
    bbox_kwargs: Dict[str, float], n_rows: int, n_cols: int
) -> None:
    """Grids have n_rows * n_cols cells in row-major order, tiling without gaps."""
    grid = models.Grid.from_bbox(n_rows=n_rows, n_cols=n_cols, **bbox_kwargs)

    assert len(grid.cells) == n_rows * n_cols
    assert [cell.index for cell in grid.cells] == list(range(n_rows * n_cols))
    for row in range(n_rows):
        for col in range(n_cols):
            cell = grid.cells[row * n_cols + col]
            if col > 0:
                left_neighbor = grid.cells[row * n_cols + col - 1]
                assert cell.x_min == pytest.approx(left_neighbor.x_max)
            if row > 0:
                upper_neighbor = grid.cells[(row - 1) * n_cols + col]
                assert cell.y_max == pytest.approx(upper_neighbor.y_min)


def test_padding_only_extends_outer_boundaries() -> None:
    """Interior boundaries are exact subdivisions; only outer edges get padding.

    For a 3x3 grid over [0, 30], interior boundaries must fall exactly on 10
    and 20, while the outermost edges extend by the default padding of 0.1.
    """
    grid = models.Grid.from_bbox(
        x_min=0.0, x_max=30.0, y_min=0.0, y_max=30.0, n_rows=3, n_cols=3
    )

    center = grid.cells[4]  # row 1, col 1: fully interior, no padding anywhere
    assert center.x_min == pytest.approx(10.0)
    assert center.x_max == pytest.approx(20.0)
    assert center.y_min == pytest.approx(10.0)
    assert center.y_max == pytest.approx(20.0)

    middle_right = grid.cells[5]  # row 1, col 2: padded only on the right edge
    assert middle_right.x_min == pytest.approx(20.0)
    assert middle_right.x_max == pytest.approx(30.1)
    assert middle_right.y_min == pytest.approx(10.0)
    assert middle_right.y_max == pytest.approx(20.0)


@pytest.mark.parametrize("padding", [0.1, 1.0], ids=["default", "custom"])
def test_outer_boundaries_extended_by_padding(
    bbox_kwargs: Dict[str, float], padding: float
) -> None:
    """The outermost grid edges extend beyond the bounding box by the padding."""
    grid = models.Grid.from_bbox(n_rows=1, n_cols=1, padding=padding, **bbox_kwargs)

    cell = grid.cells[0]
    assert cell.x_min == pytest.approx(-padding)
    assert cell.x_max == pytest.approx(10.0 + padding)
    assert cell.y_min == pytest.approx(-padding)
    assert cell.y_max == pytest.approx(10.0 + padding)


def test_labels_assigned(grid_2x2: models.Grid) -> None:
    """Labels are assigned in row-major order when provided."""
    assert [cell.label for cell in grid_2x2.cells] == ["TL", "TR", "BL", "BR"]


def test_labels_default_empty(bbox_kwargs: Dict[str, float]) -> None:
    """Cells have empty labels when none are provided."""
    grid = models.Grid.from_bbox(n_rows=1, n_cols=2, **bbox_kwargs)

    assert all(cell.label == "" for cell in grid.cells)


def test_labels_length_mismatch_raises(bbox_kwargs: Dict[str, float]) -> None:
    """Providing the wrong number of labels raises ValueError."""
    with pytest.raises(ValueError, match="labels length"):
        models.Grid.from_bbox(n_rows=2, n_cols=2, labels=["A", "B"], **bbox_kwargs)


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(0, 1), (1, 0), (0, 0), (-1, 1)],
    ids=["zero_rows", "zero_cols", "both_zero", "negative_rows"],
)
def test_invalid_dimensions_raise(
    bbox_kwargs: Dict[str, float], n_rows: int, n_cols: int
) -> None:
    """Non-positive row or column counts raise ValueError."""
    with pytest.raises(ValueError, match="n_rows and n_cols must be at least 1"):
        models.Grid.from_bbox(n_rows=n_rows, n_cols=n_cols, **bbox_kwargs)


def test_default_strokes_are_empty(bbox_kwargs: Dict[str, float]) -> None:
    """All cells start with empty strokes lists."""
    grid = models.Grid.from_bbox(n_rows=2, n_cols=3, **bbox_kwargs)

    assert all(cell.strokes == [] for cell in grid.cells)


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(2, 2), (3, 3), (2, 4)],
    ids=["2x2", "3x3", "2x4"],
)
def test_get_cell_for_point_maps_cell_centers(
    bbox_kwargs: Dict[str, float], n_rows: int, n_cols: int
) -> None:
    """The center of every cell maps back to that cell's index."""
    grid = models.Grid.from_bbox(n_rows=n_rows, n_cols=n_cols, **bbox_kwargs)
    col_width = (bbox_kwargs["x_max"] - bbox_kwargs["x_min"]) / n_cols
    row_height = (bbox_kwargs["y_max"] - bbox_kwargs["y_min"]) / n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            center_x = bbox_kwargs["x_min"] + (col + 0.5) * col_width
            center_y = bbox_kwargs["y_max"] - (row + 0.5) * row_height
            assert grid.get_cell_for_point(center_x, center_y) == row * n_cols + col


def test_point_outside_grid_returns_negative_one(grid_2x2: models.Grid) -> None:
    """A point far outside the grid returns -1."""
    assert grid_2x2.get_cell_for_point(100.0, 100.0) == -1


def test_point_on_inner_boundary_ownership(grid_2x2: models.Grid) -> None:
    """Half-open intervals assign interior boundary points deterministically.

    A point on a vertical interior boundary belongs to the cell on its right;
    a point on a horizontal interior boundary belongs to the cell above it.
    """
    assert grid_2x2.get_cell_for_point(5.0, 8.0) == 1
    assert grid_2x2.get_cell_for_point(2.0, 5.0) == 0


def test_point_on_outer_edge_captured_by_padding(
    bbox_kwargs: Dict[str, float],
) -> None:
    """Points on the exact bounding box corners are captured thanks to padding."""
    grid = models.Grid.from_bbox(n_rows=1, n_cols=1, **bbox_kwargs)

    assert grid.get_cell_for_point(0.0, 0.0) == 0
    assert grid.get_cell_for_point(10.0, 10.0) == 0


def test_empty_grid_returns_negative_one() -> None:
    """A grid with no cells always returns -1."""
    grid = models.Grid(cells=[])

    assert grid.get_cell_for_point(5.0, 5.0) == -1
