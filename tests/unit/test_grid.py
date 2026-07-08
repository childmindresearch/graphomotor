"""Test cases for the Grid model."""

import pytest

from graphomotor.core import models


@pytest.fixture
def bbox_bounds() -> tuple[float, float, float, float]:
    """Bounding box coordinates as (x_min, x_max, y_min, y_max)."""
    return (0.0, 10.0, 0.0, 10.0)


@pytest.fixture
def grid_2x2(bbox_bounds: tuple[float, float, float, float]) -> models.Grid:
    """Create a labeled 2x2 grid over a 10x10 bounding box."""
    x_min, x_max, y_min, y_max = bbox_bounds
    return models.Grid.from_bbox(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_rows=2,
        n_cols=2,
        labels=["TL", "TR", "BL", "BR"],
    )


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(1, 1), (2, 2), (3, 3), (2, 3)],
    ids=["1x1", "2x2", "3x3", "2x3"],
)
def test_grid_structure(
    bbox_bounds: tuple[float, float, float, float], n_rows: int, n_cols: int
) -> None:
    """Grids have n_rows * n_cols cells in row-major order, tiling without gaps."""
    x_min, x_max, y_min, y_max = bbox_bounds
    grid = models.Grid.from_bbox(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    assert len(grid.cells) == n_rows * n_cols
    assert [cell.index for cell in grid.cells] == list(range(n_rows * n_cols))
    for row in range(n_rows):
        for col in range(n_cols):
            cell = grid.cells[row * n_cols + col]
            # No horizontal gaps/overlaps: a cell's left edge meets the right
            # edge of its left neighbor.
            if col > 0:
                left_neighbor = grid.cells[row * n_cols + col - 1]
                assert cell.x_min == pytest.approx(left_neighbor.x_max)
            # No vertical gaps/overlaps: a cell's top edge meets the bottom edge
            # of the neighbor above it (rows run top-to-bottom).
            if row > 0:
                upper_neighbor = grid.cells[(row - 1) * n_cols + col]
                assert cell.y_max == pytest.approx(upper_neighbor.y_min)


@pytest.mark.parametrize(
    "padding", [0.1, 1.0], ids=["default_padding", "custom_padding"]
)
def test_cell_boundaries_interior_exact_outer_padded(padding: float) -> None:
    """Interior edges are exact subdivisions; only outer edges extend by padding.

    Builds a 3x3 grid over [0, 30] (each cell spans 10 units) and checks every
    cell's four boundaries in one pass, covering all three boundary kinds:
    - interior edges shared between neighbors fall exactly on multiples of 10,
    - the four outermost edges extend beyond the bounding box by ``padding``,
    - corner cells combine both (two exact interior edges, two padded outer).
    Parameterizing padding exercises both the default and a custom value.
    """
    n_rows = n_cols = 3
    cell_size = 10.0
    grid = models.Grid.from_bbox(
        x_min=0.0,
        x_max=30.0,
        y_min=0.0,
        y_max=30.0,
        n_rows=n_rows,
        n_cols=n_cols,
        padding=padding,
    )

    for row in range(n_rows):
        for col in range(n_cols):
            cell = grid.cells[row * n_cols + col]

            expected_x_min = col * cell_size - (padding if col == 0 else 0.0)
            expected_x_max = (col + 1) * cell_size + (
                padding if col == n_cols - 1 else 0.0
            )
            # Rows run top-to-bottom, so row 0 is the top of the [0, 30] range.
            expected_y_max = (n_rows - row) * cell_size + (padding if row == 0 else 0.0)
            expected_y_min = (n_rows - row - 1) * cell_size - (
                padding if row == n_rows - 1 else 0.0
            )

            assert cell.x_min == pytest.approx(expected_x_min)
            assert cell.x_max == pytest.approx(expected_x_max)
            assert cell.y_min == pytest.approx(expected_y_min)
            assert cell.y_max == pytest.approx(expected_y_max)


def test_labels_assigned(grid_2x2: models.Grid) -> None:
    """Labels are assigned in row-major order when provided."""
    assert [cell.label for cell in grid_2x2.cells] == ["TL", "TR", "BL", "BR"]


def test_labels_default_empty(bbox_bounds: tuple[float, float, float, float]) -> None:
    """Cells have empty labels when none are provided."""
    x_min, x_max, y_min, y_max = bbox_bounds
    grid = models.Grid.from_bbox(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_rows=1, n_cols=2
    )

    assert all(cell.label == "" for cell in grid.cells)


def test_labels_length_mismatch_raises(
    bbox_bounds: tuple[float, float, float, float],
) -> None:
    """Providing the wrong number of labels raises ValueError."""
    x_min, x_max, y_min, y_max = bbox_bounds
    with pytest.raises(ValueError, match="labels length"):
        models.Grid.from_bbox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_rows=2,
            n_cols=2,
            labels=["A", "B"],
        )


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(0, 1), (1, 0), (0, 0), (-1, 1)],
    ids=["zero_rows", "zero_cols", "both_zero", "negative_rows"],
)
def test_invalid_dimensions_raise(
    bbox_bounds: tuple[float, float, float, float], n_rows: int, n_cols: int
) -> None:
    """Non-positive row or column counts raise ValueError."""
    x_min, x_max, y_min, y_max = bbox_bounds
    with pytest.raises(ValueError, match="n_rows and n_cols must be at least 1"):
        models.Grid.from_bbox(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            n_rows=n_rows,
            n_cols=n_cols,
        )


def test_default_strokes_are_empty(
    bbox_bounds: tuple[float, float, float, float],
) -> None:
    """All cells start with empty strokes lists."""
    x_min, x_max, y_min, y_max = bbox_bounds
    grid = models.Grid.from_bbox(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_rows=2, n_cols=3
    )

    assert all(cell.strokes == [] for cell in grid.cells)


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(2, 2), (3, 3), (2, 4)],
    ids=["2x2", "3x3", "2x4"],
)
def test_get_cell_for_point_maps_cell_centers(
    bbox_bounds: tuple[float, float, float, float], n_rows: int, n_cols: int
) -> None:
    """The center of every cell maps back to that cell's index."""
    x_min, x_max, y_min, y_max = bbox_bounds
    grid = models.Grid.from_bbox(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    col_width = (x_max - x_min) / n_cols
    row_height = (y_max - y_min) / n_rows

    for row in range(n_rows):
        for col in range(n_cols):
            center_x = x_min + (col + 0.5) * col_width
            center_y = y_max - (row + 0.5) * row_height
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
    bbox_bounds: tuple[float, float, float, float],
) -> None:
    """Points on the exact bounding box corners are captured thanks to padding."""
    x_min, x_max, y_min, y_max = bbox_bounds
    grid = models.Grid.from_bbox(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_rows=1, n_cols=1
    )

    assert grid.get_cell_for_point(0.0, 0.0) == 0
    assert grid.get_cell_for_point(10.0, 10.0) == 0


def test_empty_grid_returns_negative_one() -> None:
    """A grid with no cells always returns -1."""
    grid = models.Grid(cells=[])

    assert grid.get_cell_for_point(5.0, 5.0) == -1
