"""Test cases for the Stroke model."""

import pandas as pd

from graphomotor.core import models


def _make_points(line_number: int = 0) -> pd.DataFrame:
    """Create a stroke DataFrame matching real Alphabet CSV structure.

    In the real data each stroke (line_number) contains many rows with the same
    line_number value, plus x, y, seconds, and timestamp columns.
    """
    return pd.DataFrame(
        {
            "line_number": [line_number] * 5,
            "x": [18.35, 18.24, 18.15, 18.12, 17.99],
            "y": [92.84, 92.85, 92.88, 92.92, 93.01],
            "seconds": [0.0, 0.02, 0.037, 0.046, 0.062],
        }
    )


def test_stroke_initialization() -> None:
    """Stroke should store points and line_number with default feature values."""
    points = _make_points(line_number=0)
    stroke = models.Stroke(points=points, line_number=0)

    assert stroke.line_number == 0
    assert stroke.points.equals(points)
    assert list(stroke.points.columns) == ["line_number", "x", "y", "seconds"]
    assert len(stroke.points) == 5
    assert stroke.duration == 0.0
    assert stroke.distance == 0.0
    assert stroke.mean_speed == 0.0
    assert stroke.speed_variance == 0.0
    assert stroke.smoothness == 0.0
    assert stroke.hesitation_count == 0
    assert stroke.hesitation_duration == 0.0
    assert stroke.velocities == []
    assert stroke.accelerations == []


def test_stroke_features_are_mutable() -> None:
    """Computed features should be writable after initialization."""
    stroke = models.Stroke(points=_make_points(line_number=0), line_number=0)

    stroke.duration = 0.5
    stroke.distance = 25.0
    stroke.mean_speed = 50.0
    stroke.velocities = [40.0, 50.0, 60.0]

    assert stroke.duration == 0.5
    assert stroke.distance == 25.0
    assert stroke.mean_speed == 50.0
    assert stroke.velocities == [40.0, 50.0, 60.0]


def test_grid_cell_stores_strokes() -> None:
    """GridCell should hold Stroke objects in its strokes list."""
    cell = models.GridCell(x_min=0.0, x_max=30.0, y_min=70.0, y_max=100.0)
    stroke_a = models.Stroke(points=_make_points(line_number=0), line_number=0)
    stroke_b = models.Stroke(points=_make_points(line_number=1), line_number=1)

    cell.strokes.append(stroke_a)
    cell.strokes.append(stroke_b)

    assert len(cell.strokes) == 2
    assert cell.strokes[0].line_number == 0
    assert cell.strokes[1].line_number == 1


def test_grid_cell_strokes_default_empty() -> None:
    """GridCell should default to an empty strokes list."""
    cell = models.GridCell(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    assert cell.strokes == []
