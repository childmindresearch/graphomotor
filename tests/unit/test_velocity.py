"""Test cases for velocity.py functions."""

import numpy as np
import pandas as pd

from graphomotor.core import models
from graphomotor.features import velocity


def test_calculate_velocity_metrics(valid_spiral: models.Spiral) -> None:
    """Test that velocity metrics are calculated correctly."""
    num_points = 1000
    time_total = 100
    theta_end = 10

    t = np.linspace(0, time_total, num_points, endpoint=False)
    theta = np.linspace(0, theta_end, num_points, endpoint=False)
    x = 50 + theta * np.cos(theta)
    y = 50 + theta * np.sin(theta)

    data = pd.DataFrame({"x": x, "y": y, "seconds": t})
    valid_spiral.data = data

    expected_angular_velocity_sum = (theta_end / time_total) * (num_points - 1)

    metrics = velocity.calculate_velocity_metrics(valid_spiral)

    assert metrics["angular_velocity_sum"] == expected_angular_velocity_sum
