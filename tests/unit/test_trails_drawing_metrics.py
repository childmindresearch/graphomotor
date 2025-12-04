"""Test cases for drawing_metrics.py functions."""

import pandas as pd

from graphomotor.core import models
from graphomotor.features.trails.drawing_metrics import detect_pen_lifts


def test_no_pen_lifts() -> None:
    """Test case with no pen lifts."""
    df = pd.DataFrame({"line_number": [1, 1, 1, 1]})
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    assert detect_pen_lifts(drawing) == 0


def test_valid_pen_lifts() -> None:
    """Test case with valid pen lifts."""
    df = pd.DataFrame({"line_number": [1, 2, 3, 4]})
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    assert detect_pen_lifts(drawing) == 3
