"""Utility functions for generating an equidistant reference spiral."""

import numpy as np
from scipy import integrate, optimize

from graphomotor.core.config import _SpiralConfig


def _arc_length_integrand(t: float) -> float:
    """Calculate the differential arc length at angle t for an Archimedean spiral.

    Args:
        t: Angle parameter.

    Returns:
        Differential arc length value.
    """
    r_t = _SpiralConfig.SPIRAL_START_RADIUS + _SpiralConfig.SPIRAL_GROWTH_RATE * t
    return np.sqrt(r_t**2 + _SpiralConfig.SPIRAL_GROWTH_RATE**2)


def _calculate_arc_length(theta: float) -> float:
    """Calculate the arc length of the spiral from _SPIRAL_START_ANGLE to theta.

    Args:
        theta: The angle in radians.

    Returns:
        The arc length of the spiral from _SPIRAL_START_ANGLE to theta.
    """
    return integrate.quad(
        lambda t: _arc_length_integrand(t), _SpiralConfig.SPIRAL_START_ANGLE, theta
    )[0]


def _find_theta_for_arc_length(target_arc_length: float) -> float:
    """Find the theta value for a given arc length using numerical root finding.

    Args:
        target_arc_length: Target arc length.

    Returns:
        Angle theta corresponding to the arc length.
    """
    solution = optimize.root_scalar(
        lambda theta: _calculate_arc_length(theta) - target_arc_length,
        bracket=[_SpiralConfig.SPIRAL_START_ANGLE, _SpiralConfig.SPIRAL_END_ANGLE],
    )
    return solution.root


def generate_reference_spiral() -> np.ndarray:
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

    Parameters used:
        - Center coordinates: (cx, cy) = (_SPIRAL_CENTER_X, _SPIRAL_CENTER_Y)
        - Start radius: a = _SPIRAL_START_RADIUS
        - Growth rate: b = _SPIRAL_GROWTH_RATE
        - Total rotation: θ = _SPIRAL_END_ANGLE - _SPIRAL_START_ANGLE
        - Number of points: N = _SPIRAL_NUM_POINTS

    Returns:
        Array with shape (N, 2) containing Cartesian coordinates of the spiral points.
    """
    total_arc_length = _calculate_arc_length(_SpiralConfig.SPIRAL_END_ANGLE)

    arc_length_values = np.linspace(
        0, total_arc_length, _SpiralConfig.SPIRAL_NUM_POINTS
    )

    theta_values = np.array([_find_theta_for_arc_length(s) for s in arc_length_values])

    r_values = (
        _SpiralConfig.SPIRAL_START_RADIUS
        + _SpiralConfig.SPIRAL_GROWTH_RATE * theta_values
    )
    x_values = _SpiralConfig.SPIRAL_CENTER_X + r_values * np.cos(theta_values)
    y_values = _SpiralConfig.SPIRAL_CENTER_Y + r_values * np.sin(theta_values)

    return np.column_stack((x_values, y_values))
