"""Feature extraction module for velocity-based metrics in spiral drawing data."""

import numpy as np
from scipy import stats

from graphomotor.core import models


def calculate_velocity_metrics(spiral: models.Spiral) -> dict[str, float]:
    """Calculate velocity-based metrics from spiral drawing data.

    Args:
        spiral: Spiral object containing drawing data

    Returns:
        Dictionary containing calculated velocity metrics
    """
    x = spiral.data["x"].values - 50
    y = spiral.data["y"].values - 50
    t = spiral.data["seconds"].values
    r = np.sqrt(x**2 + y**2)
    theta = np.unwrap(np.arctan2(y, x))

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dr = np.diff(r)
    dtheta = np.diff(theta)

    vx = dx / dt
    vy = dy / dt
    velocity = np.sqrt(vx**2 + vy**2)
    
    radial_velocity = dr / dt
    angular_velocity = dtheta / dt

    return {
        "velocity_sum": np.sum(np.abs(velocity)),
        "velocity_cv": stats.variation(velocity),
        "velocity_skewness": stats.skew(velocity),
        "velocity_kurtosis": stats.kurtosis(velocity),
        "radial_velocity_sum": np.sum(np.abs(radial_velocity)),
        "radial_velocity_cv": stats.variation(radial_velocity),
        "radial_velocity_skewness": stats.skew(radial_velocity),
        "radial_velocity_kurtosis": stats.kurtosis(radial_velocity),
        "angular_velocity_sum": np.sum(np.abs(angular_velocity)),
        "angular_velocity_cv": stats.variation(angular_velocity),
        "angular_velocity_skewness": stats.skew(angular_velocity),
        "angular_velocity_kurtosis": stats.kurtosis(angular_velocity),
    }
