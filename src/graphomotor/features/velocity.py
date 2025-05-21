"""Feature extraction module for velocity-based metrics in spiral drawing data."""

import numpy as np
from scipy import stats

from graphomotor.core import models


def _calculate_statistics(values: np.ndarray, name: str) -> dict[str, float]:
    """Helper function to calculate statistics for a given array.

    Args:
        values: 1-D Numpy array of numerical values.
        name: Name prefix for the statistics (e.g., "linear_velocity").

    Returns:
        Dictionary containing calculated metrics (sum, median, variation, skewness,
        kurtosis) with keys prefixed by the provided name.
    """
    return {
        f"{name}_sum": np.sum(np.abs(values)),
        f"{name}_median": np.median(np.abs(values)),
        f"{name}_variation": stats.variation(values),
        f"{name}_skewness": stats.skew(values),
        f"{name}_kurtosis": stats.kurtosis(values),
    }


def calculate_velocity_metrics(spiral: models.Spiral) -> dict[str, float]:
    """Calculate velocity-based metrics from spiral drawing data.

    This function computes three types of velocity metrics by calculating the difference
    between consecutive points in the spiral drawing data. The three types of velocity
    are:
        1. Linear velocity: The magnitude of change of Euclidean distance in pixels
           per second. This is calculated as the square root of the sum of squares of
           the differences in x and y coordinates divided by the difference in time.
        2. Radial velocity: The magnitude of change of distance from center (radius) in
           pixels per second. Radius is calculated as the square root of the sum of
           squares of x and y coordinates.
        3. Angular velocity: The magnitude of change of angle in radians per second.
           Angle is calculated using the arctangent of y coordinates divided by x
           coordinates, and then unwrapped to maintain continuity across the -π to π
           boundary.

    For each velocity type, the following metrics are calculated:
        - Sum: Sum of absolute velocity values
        - Median: Median of absolute velocity values
        - Variation: Coefficient of variation
        - Skewness: Asymmetry of the velocity distribution
        - Kurtosis: Tailedness of the velocity distribution

    Args:
        spiral: Spiral object containing drawing data.

    Returns:
        Dictionary containing calculated velocity metrics.
    """
    x = spiral.data["x"].values
    y = spiral.data["y"].values
    time = spiral.data["seconds"].values
    radius = np.sqrt(x**2 + y**2)
    theta = np.unwrap(np.arctan2(y, x))

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(time)
    dr = np.diff(radius)
    dtheta = np.diff(theta)

    linear_velocity = np.sqrt(dx**2 + dy**2) / dt
    radial_velocity = dr / dt
    angular_velocity = dtheta / dt

    return {
        **_calculate_statistics(linear_velocity, "linear_velocity"),
        **_calculate_statistics(radial_velocity, "radial_velocity"),
        **_calculate_statistics(angular_velocity, "angular_velocity"),
    }
