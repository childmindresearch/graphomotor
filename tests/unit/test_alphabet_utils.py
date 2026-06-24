"""Test cases for the segment_strokes utility function."""

import pandas as pd
import pytest

from graphomotor.utils import alphabet_utils


@pytest.fixture
def drawing_data() -> pd.DataFrame:
    """Drawing data with three strokes in distinct spatial regions.

    On a 2x2 grid over [0, 100]:
    - Stroke 0 has centroid near (5, 85): top-left cell.
    - Stroke 1 has centroid near (55, 85): top-right cell.
    - Stroke 2 has centroid near (5, 15): bottom-left cell.
    """
    return pd.DataFrame(
        {
            "line_number": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x": [3.0, 5.0, 7.0, 53.0, 55.0, 57.0, 3.0, 5.0, 7.0],
            "y": [83.0, 85.0, 87.0, 83.0, 85.0, 87.0, 13.0, 15.0, 17.0],
            "seconds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


def test_strokes_assigned_to_correct_cells(drawing_data: pd.DataFrame) -> None:
    """Each stroke is placed in the labeled cell containing its centroid.

    Also checks that the total number of assigned strokes equals the number of
    line_number groups, so no stroke is dropped or duplicated.
    """
    grid = alphabet_utils.segment_strokes(
        data=drawing_data,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
        n_rows=2,
        n_cols=2,
        labels=["TL", "TR", "BL", "BR"],
    )
    total_strokes = sum(len(cell.strokes) for cell in grid.cells)

    assert [cell.label for cell in grid.cells] == ["TL", "TR", "BL", "BR"]
    assert len(grid.cells[0].strokes) == 1
    assert grid.cells[0].strokes[0].line_number == 0
    assert len(grid.cells[1].strokes) == 1
    assert grid.cells[1].strokes[0].line_number == 1
    assert len(grid.cells[2].strokes) == 1
    assert grid.cells[2].strokes[0].line_number == 2
    assert len(grid.cells[3].strokes) == 0
    assert total_strokes == drawing_data["line_number"].nunique()


def test_stroke_points_are_correct(drawing_data: pd.DataFrame) -> None:
    """Every stroke contains exactly the points of its line_number group."""
    grid = alphabet_utils.segment_strokes(
        data=drawing_data,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
        n_rows=2,
        n_cols=2,
    )

    strokes_by_line = {
        stroke.line_number: stroke for cell in grid.cells for stroke in cell.strokes
    }
    assert sorted(strokes_by_line) == [0, 1, 2]
    for line_number, stroke in strokes_by_line.items():
        expected = drawing_data[drawing_data["line_number"] == line_number]
        assert list(stroke.points["x"]) == list(expected["x"])
        assert list(stroke.points["y"]) == list(expected["y"])
        assert list(stroke.points["seconds"]) == list(expected["seconds"])


def test_empty_dataframe() -> None:
    """An empty DataFrame produces a grid with no strokes."""
    data = pd.DataFrame(columns=["line_number", "x", "y", "seconds"])
    grid = alphabet_utils.segment_strokes(
        data=data,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        n_rows=1,
        n_cols=1,
    )

    assert all(len(cell.strokes) == 0 for cell in grid.cells)


def test_stroke_outside_grid_is_not_assigned() -> None:
    """Strokes whose centroids fall outside the grid are dropped."""
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
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        n_rows=1,
        n_cols=1,
    )

    assert len(grid.cells[0].strokes) == 0


def test_multiple_strokes_in_same_cell() -> None:
    """Multiple strokes in the same spatial region all land in one cell."""
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
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        n_rows=1,
        n_cols=1,
    )

    assert len(grid.cells[0].strokes) == 3


def test_grid_structure_matches_parameters(drawing_data: pd.DataFrame) -> None:
    """Returned grid has the correct number of labeled cells."""
    labels = ["A", "B", "C", "D", "E", "F"]
    grid = alphabet_utils.segment_strokes(
        data=drawing_data,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
        n_rows=2,
        n_cols=3,
        labels=labels,
    )

    assert len(grid.cells) == 6
    assert [cell.label for cell in grid.cells] == labels


def test_stroke_index_is_reset() -> None:
    """Stroke points have a fresh 0-based index.

    pandas groupby chunks keep the row indices of the parent DataFrame
    (here 10-12); segment_strokes resets them so each stroke's points are
    independently indexed from 0.
    """
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
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        n_rows=1,
        n_cols=1,
    )

    stroke = grid.cells[0].strokes[0]
    assert list(stroke.points.index) == [0, 1, 2]
