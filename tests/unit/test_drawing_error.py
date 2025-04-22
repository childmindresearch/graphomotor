"""Test cases for drawing_error.py functions."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from graphomotor.core import models
from graphomotor.features import drawing_error


def test_calculate_area_under_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the area under the curve is calculated correctly."""
    x = np.linspace(-np.pi / 2, 6 * np.pi / 4, 100)
    y1 = np.sin(x)
    y2 = np.sin(x + np.pi)

    expected_area, _ = integrate.quad(
        lambda x: np.abs(np.sin(x) - np.sin(x + np.pi)), -np.pi / 2, 6 * np.pi / 4
    )

    class MockSpiral:
        def __init__(self, data: pd.DataFrame) -> None:
            self.data = data

    monkeypatch.setattr(models, "Spiral", MockSpiral)

    calculated_area = drawing_error.calculate_area_under_curve(
        models.Spiral(data=pd.DataFrame({"x": x, "y": y1})), np.array([x, y2]).T
    )["area_under_curve"]

    assert np.isclose(calculated_area, expected_area, rtol=1e-3)
