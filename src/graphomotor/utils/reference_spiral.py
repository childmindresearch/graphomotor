"""Generate a reference spiral with equidistant points along its arc length."""

import numpy as np
from scipy import integrate, optimize

_SPIRAL_CENTER_X = 50
_SPIRAL_CENTER_Y = 50
_SPIRAL_INITIAL_RADIUS = 0
_SPIRAL_GROWTH_RATE = 1.075
_SPIRAL_TOTAL_ROTATION = 8 * np.pi
_SPIRAL_NUM_POINTS = 10000


def _spiral_arc_length_integrand(t: float) -> float:
    """Calculate the differential arc length at angle t for an Archimedean spiral.

    Args:
        t: Angle parameter.

    Returns:
        Differential arc length value.
    """
    r_t = _SPIRAL_INITIAL_RADIUS + _SPIRAL_GROWTH_RATE * t
    return np.sqrt(r_t**2 + _SPIRAL_GROWTH_RATE**2)


def _calculate_arc_length(theta: float) -> float:
    """Calculate the arc length of the spiral from 0 to theta.

    Args:
        theta: The angle in radians.

    Returns:
        The arc length of the spiral from 0 to theta.
    """
    return integrate.quad(lambda t: _spiral_arc_length_integrand(t), 0, theta)[0]


def _arc_length_difference(theta: float, target_arc_length: float) -> float:
    """Function to find the root for a given arc length.

    Args:
        theta: Angle to evaluate.
        target_arc_length: Target arc length value.

    Returns:
        Difference between calculated and target arc length.
    """
    return _calculate_arc_length(theta) - target_arc_length


def _find_theta_for_arc_length(target_arc_length: float) -> float:
    """Find the theta value for a given arc length.

    Args:
        target_arc_length: Target arc length.

    Returns:
        Angle theta corresponding to the arc length.
    """
    solution = optimize.root_scalar(
        lambda theta: _arc_length_difference(theta, target_arc_length),
        bracket=[0, _SPIRAL_TOTAL_ROTATION],
    )
    return solution.root


def generate_reference_spiral() -> np.ndarray:
    """Generate a reference spiral with equidistant points along its arc length.

    This function creates an Archimedean spiral with points distributed at equal arc
    length intervals. The generated spiral serves as a standardized reference template
    for feature extraction algorithms that compare user-drawn spirals with an ideal
    form.

    The algorithm works by:
        1. Computing the total arc length for the entire spiral (0 to 8π),
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
        - Center coordinates: (50, 50)
        - Initial radius (a): 0
        - Growth rate (b): 1.075
        - Total rotation: 4 complete revolutions (θ from 0 to 8π)
        - Number of points: 10,000

    Returns:
        Array with shape (10000, 2) containing Cartesian coordinates of the spiral.
    """
    total_arc_length = _calculate_arc_length(_SPIRAL_TOTAL_ROTATION)

    arc_length_values = np.linspace(0, total_arc_length, _SPIRAL_NUM_POINTS)

    theta_values = np.array([_find_theta_for_arc_length(s) for s in arc_length_values])

    r_values = _SPIRAL_INITIAL_RADIUS + _SPIRAL_GROWTH_RATE * theta_values
    x_values = _SPIRAL_CENTER_X + r_values * np.cos(theta_values)
    y_values = _SPIRAL_CENTER_Y + r_values * np.sin(theta_values)

    return np.column_stack((x_values, y_values))
