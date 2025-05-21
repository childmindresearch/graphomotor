"""Utility functions for generating an equidistant reference spiral."""

import numpy as np
from scipy import integrate, optimize

from graphomotor.core.config import SpiralConfig


def _arc_length_integrand(t: float, config: SpiralConfig) -> float:
    """Calculate the differential arc length at angle t for an Archimedean spiral.

    Args:
        t: Angle parameter.
        config: Spiral configuration.

    Returns:
        Differential arc length value.
    """
    r_t = config.start_radius + config.growth_rate * t
    return np.sqrt(r_t**2 + config.growth_rate**2)


def _calculate_arc_length(theta: float, config: SpiralConfig) -> float:
    """Calculate the arc length of the spiral from start_angle to theta.

    Args:
        theta: The angle in radians.
        config: Spiral configuration.

    Returns:
        The arc length of the spiral from start_angle to theta.
    """
    return integrate.quad(
        lambda t: _arc_length_integrand(t, config), config.start_angle, theta
    )[0]


def _find_theta_for_arc_length(target_arc_length: float, config: SpiralConfig) -> float:
    """Find the theta value for a given arc length using numerical root finding.

    Args:
        target_arc_length: Target arc length.
        config: Spiral configuration.

    Returns:
        Angle theta corresponding to the arc length.
    """
    solution = optimize.root_scalar(
        lambda theta: _calculate_arc_length(theta, config) - target_arc_length,
        bracket=[config.start_angle, config.end_angle],
    )
    return solution.root


def generate_reference_spiral(config: SpiralConfig) -> np.ndarray:
    """Generate a reference spiral with equidistant points along its arc length.

    This function creates an Archimedean spiral with points distributed at equal arc
    length intervals. The generated spiral serves as a standardized reference template
    for feature extraction algorithms that compare user-drawn spirals with an ideal
    form.

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
        - Center coordinates: (cx, cy) = (config.center_x, config.center_y)
        - Start radius: a = config.start_radius
        - Growth rate: b = config.growth_rate
        - Total rotation: θ = config.end_angle - config.start_angle
        - Number of points: N = config.num_points

    Args:
        config: Configuration parameters for the spiral.

    Returns:
        Array with shape (N, 2) containing Cartesian coordinates of the spiral points.
    """
    total_arc_length = _calculate_arc_length(config.end_angle, config)

    arc_length_values = np.linspace(0, total_arc_length, config.num_points)

    theta_values = np.array(
        [_find_theta_for_arc_length(s, config) for s in arc_length_values]
    )

    r_values = config.start_radius + config.growth_rate * theta_values
    x_values = config.center_x + r_values * np.cos(theta_values)
    y_values = config.center_y + r_values * np.sin(theta_values)

    return np.column_stack((x_values, y_values))
