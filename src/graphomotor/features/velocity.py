"""Feature extraction module for velocity-based metrics in spiral drawing data."""

import numpy as np
from scipy import stats

from graphomotor.core import models


def _get_velocity_metrics(velocity: np.ndarray, type_: str) -> dict[str, float]:
    """Calculate velocity metrics for a given type of velocity.

    Args:
        velocity: Numpy array of velocity values.
        type_: Type of velocity (e.g., "linear_velocity", "radial_velocity",
        "angular_velocity").

    Returns:
        Dictionary containing calculated metrics for the specified type of velocity.
    """
    return {
        f"{type_}_sum": np.sum(np.abs(velocity)),
        f"{type_}_variation": stats.variation(velocity),
        f"{type_}_skewness": stats.skew(velocity),
        f"{type_}_kurtosis": stats.kurtosis(velocity),
    }


def calculate_velocity_metrics(spiral: models.Spiral) -> dict[str, float]:
    """Calculate velocity-based metrics from spiral drawing data.

    This function computes three types of velocity metrics by calculating the difference
    between consecutive points in the spiral drawing data. The three types of velocity
    are:
        1. Linear velocity: The magnitude of change of Euclidean distance in pixels
           per second. This is calculated as the square root of the sum of squares of
           the differences in x and y coordinates divided by the difference in time.
        2. Radial velocity: The magnitude of change of distance from center in pixels
           per second. Radius is calculated as the square root of the sum of squares of
           the x and y coordinates. The radial velocity is the change in radius divided
           by the change in time.
        3. Angular velocity: The magnitude of change of angle in radians per second.
           Angle is calculated using the arctangent of y coordinates divided by x
           coordinates. It is assumed that the drawing starts in the first quadrant
           The angle is unwrapped to ensure continuity. The angular
           velocity is the change in angle divided by the change in time.

    For each velocity type, the following metrics are calculated:
        - Sum: Sum of absolute velocity values
        - Variation: Coefficient of variation
        - Skewness: Asymmetry of the velocity distribution
        - Kurtosis: Tailedness of the velocity distribution

    Args:
        spiral: Spiral object containing drawing data.

    Returns:
        Dictionary containing calculated velocity metrics.
    """
    x_coord = spiral.data["x"].values - 50
    y_coord = spiral.data["y"].values - 50
    time = spiral.data["seconds"].values
    radius = np.sqrt(x_coord**2 + y_coord**2)
    theta = np.unwrap(np.arctan2(y_coord, x_coord))

    dx = np.diff(x_coord)
    dy = np.diff(y_coord)
    dt = np.diff(time)
    dr = np.diff(radius)
    dtheta = np.diff(theta)

    vx = dx / dt
    vy = dy / dt
    linear_velocity = np.sqrt(vx**2 + vy**2)

    radial_velocity = dr / dt
    angular_velocity = dtheta / dt

    linear_velocity_metrics = _get_velocity_metrics(linear_velocity, "linear_velocity")
    radial_velocity_metrics = _get_velocity_metrics(radial_velocity, "radial_velocity")
    angular_velocity_metrics = _get_velocity_metrics(
        angular_velocity, "angular_velocity"
    )
    return {
        **linear_velocity_metrics,
        **radial_velocity_metrics,
        **angular_velocity_metrics,
    }
