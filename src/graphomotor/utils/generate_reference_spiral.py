"""Utility functions for generating an equidistant reference spiral."""

import functools

import numpy as np
from scipy import integrate, optimize

from graphomotor.core import config


def _arc_length_integrand(t: float, spiral_config: config.SpiralConfig) -> float:
    """Calculate the differential arc length at angle t for an Archimedean spiral.

    Args:
        t: Angle parameter.
        spiral_config: Spiral configuration.

    Returns:
        Differential arc length value.
    """
    r_t = spiral_config.start_radius + spiral_config.growth_rate * t
    return np.sqrt(r_t**2 + spiral_config.growth_rate**2)


def _calculate_arc_length_between(
    theta_start: float, theta_end: float, spiral_config: config.SpiralConfig
) -> float:
    """Calculate the arc length of the spiral between two theta values.

    Args:
        theta_start: Starting angle in radians.
        theta_end: Ending angle in radians.
        spiral_config: Spiral configuration.

    Returns:
        The arc length of the spiral from theta_start to theta_end.
    """
    return integrate.quad(
        lambda t: _arc_length_integrand(t, spiral_config),
        theta_start,
        theta_end,
    )[0]


def _find_theta_for_incremental_arc_length(
    target_increment: float,
    current_theta: float,
    spiral_config: config.SpiralConfig,
) -> float:
    """Find the theta value for a given incremental arc length from current position.

    Args:
        target_increment: Target arc length increment from current position.
        current_theta: Current theta position.
        spiral_config: Spiral configuration.

    Returns:
        Angle theta corresponding to the target cumulative arc length.
    """
    solution = optimize.root_scalar(
        lambda theta: _calculate_arc_length_between(current_theta, theta, spiral_config)
        - target_increment,
        bracket=(current_theta, spiral_config.end_angle),
    )
    return solution.root


@functools.lru_cache(maxsize=48)
def generate_reference_spiral(spiral_config: config.SpiralConfig) -> np.ndarray:
    """Generate a reference spiral with equidistant points along its arc length.

    This function creates an Archimedean spiral with points distributed at equal arc
    length intervals. The generated spiral serves as a standardized reference template
    for feature extraction algorithms that compare user-drawn spirals with an ideal
    form.

    The function first attempts to load a pre-computed spiral from cache. If not found,
    it calculates the spiral using numerical computation and automatically saves it
    to cache for future use.

    The algorithm works by:
        1. Computing the total arc length for the entire spiral,
        2. Creating equidistant target arc length values,
        3. For each target arc length, finding the corresponding theta value that
           produces that arc length using numerical root finding,
        4. Converting these theta values to Cartesian coordinates.

    Mathematical formulas used:
        - Spiral equation: r(θ) = a + b·θ
        - Arc length differential: ds = √(r(θ)² + b²) dθ
        - Arc length from 0 to θ: s(θ) = ∫₀ᶿ √(r(t)² + b²) dt
        - Cartesian coordinates: x = cx + r·cos(θ), y = cy + r·sin(θ)

    Parameters are defined in the SpiralConfig class:
        - Center coordinates: cx, cy = spiral_config.center_x, spiral_config.center_y
        - Start radius: a = spiral_config.start_radius
        - Growth rate: b = spiral_config.growth_rate
        - Total rotation: θ = spiral_config.end_angle - spiral_config.start_angle
        - Number of points: N = spiral_config.num_points

    Args:
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Array with shape (N, 2) containing Cartesian coordinates of the spiral points.
    """
    total_arc_length = _calculate_arc_length_between(
        spiral_config.start_angle, spiral_config.end_angle, spiral_config
    )

    arc_length_increment = total_arc_length / (spiral_config.num_points - 1)

    theta_values = np.zeros(spiral_config.num_points)
    theta_values[0] = spiral_config.start_angle

    for i in range(1, spiral_config.num_points):
        theta_values[i] = _find_theta_for_incremental_arc_length(
            arc_length_increment,
            theta_values[i - 1],
            spiral_config,
        )

    r_values = spiral_config.start_radius + spiral_config.growth_rate * theta_values
    x_values = spiral_config.center_x + r_values * np.cos(theta_values)
    y_values = spiral_config.center_y + r_values * np.sin(theta_values)

    return np.column_stack((x_values, y_values))
