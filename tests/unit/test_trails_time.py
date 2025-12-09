"""Tests for trails time.py."""

import pandas as pd

from graphomotor.core import models
from graphomotor.features.trails import time


def test_total_error_time_no_errors() -> None:
    """Test case with no errors."""
    df = pd.DataFrame({
        "line_number": [1, 1, 1, 1],
        "error": ["E0", "E0", "E0", "E0"],
        "seconds": [0, 1, 2, 3],
    })
    drawing = models.Drawing(data=df, task_name="trails", metadata={"id": "5555555"})

    result = time.calculate_total_error_time(drawing)
    assert result == {"total_error_time": 0.0}
