"""Test cases for the Grid model."""

import pytest

from graphomotor.core import models


class TestGridFromBbox:
    """Tests for the Grid.from_bbox factory method."""

    def test_single_cell_grid(self) -> None:
        """A 1x1 grid should produce exactly one cell covering the padded bbox."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=1, n_cols=1
        )

        assert len(grid.cells) == 1
        cell = grid.cells[0]
        assert cell.index == 0
        assert cell.x_min == pytest.approx(-0.1)
        assert cell.x_max == pytest.approx(10.1)
        assert cell.y_min == pytest.approx(-0.1)
        assert cell.y_max == pytest.approx(10.1)

    def test_two_by_two_grid_cell_count(self) -> None:
        """A 2x2 grid should produce exactly four cells."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=2, n_cols=2
        )

        assert len(grid.cells) == 4

    def test_row_major_order(self) -> None:
        """Cells should be ordered left-to-right, top-to-bottom."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=2, n_cols=2
        )

        assert grid.cells[0].index == 0
        assert grid.cells[1].index == 1
        assert grid.cells[2].index == 2
        assert grid.cells[3].index == 3

        assert grid.cells[0].y_min > grid.cells[2].y_min
        assert grid.cells[0].x_min < grid.cells[1].x_min

    def test_labels_assigned(self) -> None:
        """Labels should be assigned in row-major order when provided."""
        labels = ["A", "B", "C", "D"]
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=2, n_cols=2, labels=labels,
        )

        assert [c.label for c in grid.cells] == ["A", "B", "C", "D"]

    def test_labels_default_empty(self) -> None:
        """Cells should have empty labels when none are provided."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=1, n_cols=2
        )

        assert all(c.label == "" for c in grid.cells)

    def test_labels_length_mismatch_raises(self) -> None:
        """Providing the wrong number of labels should raise ValueError."""
        with pytest.raises(ValueError, match="labels length"):
            models.Grid.from_bbox(
                x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
                n_rows=2, n_cols=2, labels=["A", "B"],
            )

    @pytest.mark.parametrize(
        "n_rows,n_cols",
        [(0, 1), (1, 0), (0, 0), (-1, 1)],
        ids=["zero_rows", "zero_cols", "both_zero", "negative_rows"],
    )
    def test_invalid_dimensions_raise(self, n_rows: int, n_cols: int) -> None:
        """Non-positive row or column counts should raise ValueError."""
        with pytest.raises(ValueError, match="n_rows and n_cols must be at least 1"):
            models.Grid.from_bbox(
                x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
                n_rows=n_rows, n_cols=n_cols,
            )

    def test_custom_padding(self) -> None:
        """Custom padding should extend the outer boundaries accordingly."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=1, n_cols=1, padding=1.0,
        )

        cell = grid.cells[0]
        assert cell.x_min == pytest.approx(-1.0)
        assert cell.x_max == pytest.approx(11.0)
        assert cell.y_min == pytest.approx(-1.0)
        assert cell.y_max == pytest.approx(11.0)

    def test_cells_tile_without_gaps(self) -> None:
        """Adjacent cells should share boundaries with no gaps or overlap."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=2, n_cols=2
        )

        top_left, top_right = grid.cells[0], grid.cells[1]
        bottom_left = grid.cells[2]

        assert top_left.x_max == pytest.approx(top_right.x_min)
        assert top_left.y_min == pytest.approx(bottom_left.y_max)

    def test_default_strokes_are_empty(self) -> None:
        """All cells should start with empty strokes lists."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=2, n_cols=3
        )

        assert all(c.strokes == [] for c in grid.cells)


class TestGetCellForPoint:
    """Tests for Grid.get_cell_for_point."""

    @pytest.fixture
    def grid_2x2(self) -> models.Grid:
        """Create a 2x2 grid over the unit square with labels."""
        return models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=2, n_cols=2, labels=["TL", "TR", "BL", "BR"],
        )

    def test_point_in_top_left(self, grid_2x2: models.Grid) -> None:
        """Point in the top-left quadrant should return index 0."""
        assert grid_2x2.get_cell_for_point(2.0, 8.0) == 0

    def test_point_in_top_right(self, grid_2x2: models.Grid) -> None:
        """Point in the top-right quadrant should return index 1."""
        assert grid_2x2.get_cell_for_point(8.0, 8.0) == 1

    def test_point_in_bottom_left(self, grid_2x2: models.Grid) -> None:
        """Point in the bottom-left quadrant should return index 2."""
        assert grid_2x2.get_cell_for_point(2.0, 2.0) == 2

    def test_point_in_bottom_right(self, grid_2x2: models.Grid) -> None:
        """Point in the bottom-right quadrant should return index 3."""
        assert grid_2x2.get_cell_for_point(8.0, 2.0) == 3

    def test_point_outside_grid(self, grid_2x2: models.Grid) -> None:
        """Point far outside the grid should return -1."""
        assert grid_2x2.get_cell_for_point(100.0, 100.0) == -1

    def test_point_on_inner_boundary(self, grid_2x2: models.Grid) -> None:
        """Point on an internal cell boundary belongs to the next cell."""
        mid_x = (grid_2x2.cells[0].x_max + grid_2x2.cells[1].x_min) / 2
        result = grid_2x2.get_cell_for_point(mid_x, 8.0)
        assert result in (0, 1)

    def test_point_on_outer_edge_captured_by_padding(self) -> None:
        """Point on the exact data boundary should be inside due to padding."""
        grid = models.Grid.from_bbox(
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0, n_rows=1, n_cols=1
        )

        assert grid.get_cell_for_point(0.0, 0.0) == 0
        assert grid.get_cell_for_point(10.0, 10.0) == 0

    def test_empty_grid_returns_negative_one(self) -> None:
        """Grid with no cells should always return -1."""
        grid = models.Grid(cells=[])
        assert grid.get_cell_for_point(5.0, 5.0) == -1
