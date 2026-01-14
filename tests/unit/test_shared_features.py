"""Unit tests for shared features module."""

import pandas as pd

from graphomotor.core import models
from graphomotor.features import shared_features


def test_no_pen_lifts() -> None:
    """Test case with no pen lifts."""
    df = pd.DataFrame({"line_number": [1, 1, 1, 1]})
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = shared_features.detect_pen_lifts(drawing)
    assert result == {"pen_lifts": 0}


def test_valid_pen_lifts() -> None:
    """Test case with valid pen lifts."""
    df = pd.DataFrame({"line_number": [1, 2, 3, 4]})
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})
    result = shared_features.detect_pen_lifts(drawing)
    assert result == {"pen_lifts": 3}
