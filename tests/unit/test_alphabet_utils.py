"""Test cases for the segment_strokes utility function."""

import pandas as pd

from graphomotor.utils import alphabet_utils


def _make_drawing_data() -> pd.DataFrame:
    """Create drawing data with three strokes in distinct spatial regions.

    Stroke 0 has centroid near (5, 85) - top-left of a 2x2 grid.
    Stroke 1 has centroid near (55, 85) - top-right.
    Stroke 2 has centroid near (5, 15) - bottom-left.
    """
    return pd.DataFrame(
        {
            "line_number": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [3.0, 5.0, 7.0, 53.0, 55.0, 57.0, 3.0, 5.0, 7.0],
            "y": [83.0, 85.0, 87.0, 83.0, 85.0, 87.0, 13.0, 15.0, 17.0],
            "seconds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


class TestSegmentStrokes:
    """Tests for the segment_strokes function."""

    def test_strokes_assigned_to_correct_cells(self) -> None:
        """Each stroke should be placed in the cell containing its centroid."""
        data = _make_drawing_data()
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0,
            n_rows=2, n_cols=2, labels=["TL", "TR", "BL", "BR"],
        )

        assert len(grid.cells[0].strokes) == 1
        assert grid.cells[0].strokes[0].line_number == 0
        assert len(grid.cells[1].strokes) == 1
        assert grid.cells[1].strokes[0].line_number == 1
        assert len(grid.cells[2].strokes) == 1
        assert grid.cells[2].strokes[0].line_number == 2
        assert len(grid.cells[3].strokes) == 0

    def test_total_stroke_count_matches_line_numbers(self) -> None:
        """Total strokes across all cells should equal the number of line groups."""
        data = _make_drawing_data()
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0,
            n_rows=2, n_cols=2,
        )

        total_strokes = sum(len(c.strokes) for c in grid.cells)
        assert total_strokes == 3

    def test_stroke_points_are_correct(self) -> None:
        """Each Stroke should contain the correct subset of points."""
        data = _make_drawing_data()
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0,
            n_rows=2, n_cols=2,
        )

        stroke_0 = grid.cells[0].strokes[0]
        assert len(stroke_0.points) == 3
        assert list(stroke_0.points["x"]) == [3.0, 5.0, 7.0]

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should produce a grid with no strokes."""
        data = pd.DataFrame(columns=["line_number", "x", "y", "seconds"])
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=1, n_cols=1,
        )

        assert all(len(c.strokes) == 0 for c in grid.cells)

    def test_stroke_outside_grid_is_not_assigned(self) -> None:
        """Strokes whose centroids fall outside the grid should be dropped."""
        data = pd.DataFrame(
            {
                "line_number": [0, 0],
                "x": [500.0, 600.0],
                "y": [500.0, 600.0],
                "seconds": [0.0, 0.1],
            }
        )
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=1, n_cols=1,
        )

        assert len(grid.cells[0].strokes) == 0

    def test_multiple_strokes_in_same_cell(self) -> None:
        """Multiple strokes in the same spatial region should all land in one cell."""
        data = pd.DataFrame(
            {
                "line_number": [0, 0, 1, 1, 2, 2],
                "x": [5.0, 6.0, 5.5, 6.5, 4.5, 5.5],
                "y": [5.0, 6.0, 5.5, 6.5, 4.5, 5.5],
                "seconds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=1, n_cols=1,
        )

        assert len(grid.cells[0].strokes) == 3

    def test_grid_structure_matches_parameters(self) -> None:
        """Returned grid should have the correct number of labeled cells."""
        data = _make_drawing_data()
        labels = ["A", "B", "C", "D", "E", "F"]
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0,
            n_rows=2, n_cols=3, labels=labels,
        )

        assert len(grid.cells) == 6
        assert [c.label for c in grid.cells] == labels

    def test_stroke_index_is_reset(self) -> None:
        """Stroke points should have a reset index starting from 0."""
        data = pd.DataFrame(
            {
                "line_number": [5, 5, 5],
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
                "seconds": [0.0, 0.1, 0.2],
            },
            index=[10, 11, 12],
        )
        grid = alphabet_utils.segment_strokes(
            data=data,
            x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0,
            n_rows=1, n_cols=1,
        )

        stroke = grid.cells[0].strokes[0]
        assert list(stroke.points.index) == [0, 1, 2]
