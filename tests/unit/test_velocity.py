"""Test cases for velocity.py functions."""

import numpy as np
import pandas as pd

from graphomotor.core import config, models
from graphomotor.features import velocity


def test_calculate_velocity_metrics(valid_spiral: models.Spiral) -> None:
    """Test that velocity metrics are calculated correctly."""
    num_points = 1000
    time_total = 100
    theta_end = 10

    t = np.linspace(0, time_total, num_points, endpoint=False)
    theta = np.linspace(0, theta_end, num_points, endpoint=False)
    r = config._SpiralConfig.SPIRAL_GROWTH_RATE * theta
    x = config._SpiralConfig.SPIRAL_CENTER_X + r * np.cos(theta)
    y = config._SpiralConfig.SPIRAL_CENTER_Y + r * np.sin(theta)

    data = pd.DataFrame({"x": x, "y": y, "seconds": t})
    valid_spiral.data = data

    expected_angular_velocity_median = theta_end / time_total
    expected_radial_velocity_median = (
        config._SpiralConfig.SPIRAL_GROWTH_RATE * theta_end / time_total
    )
    expected_linear_velocity_median = np.median(
        np.sqrt(np.gradient(x, t) ** 2 + np.gradient(y, t) ** 2)
    )

    metrics = velocity.calculate_velocity_metrics(valid_spiral)

    assert np.isclose(
        metrics["angular_velocity_median"],
        expected_angular_velocity_median,
        atol=0,
        rtol=1e-14,
    )
    assert np.isclose(
        metrics["radial_velocity_median"],
        expected_radial_velocity_median,
        atol=0,
        rtol=1e-13,
    )
    assert np.isclose(
        metrics["linear_velocity_median"],
        expected_linear_velocity_median,
        atol=0,
        rtol=1e-4,
    )
