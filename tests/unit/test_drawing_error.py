"""Test cases for drawing_error.py functions."""

import numpy as np
import pandas as pd

from graphomotor.core import models
from graphomotor.features import drawing_error


def test_calculate_area_under_curve(valid_spiral: models.Spiral) -> None:
    """Test that the area under the curve is calculated correctly."""
    x = np.linspace(-np.pi / 2, 3 * np.pi / 2, 100)
    y1 = np.sin(x)
    y2 = np.sin(x + np.pi)

    expected_area = 8.0

    valid_spiral.data = pd.DataFrame({"x": x, "y": y1})
    calculated_area = drawing_error.calculate_area_under_curve(
        valid_spiral, np.column_stack((x, y2))
    )["area_under_curve"]

    assert np.isclose(calculated_area, expected_area, atol=0, rtol=1e-3)
