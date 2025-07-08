"""Utility functions for generating an equidistant reference spiral."""
import functools
import hashlib
import pathlib

import numpy as np
from scipy import integrate, optimize

from graphomotor.core import config

logger = config.get_logger()


def _arc_length_integrand(t: float, spiral_config: config.SpiralConfig) -> float:
    """Calculate the differential arc length at angle t for an Archimedean spiral.

    Args:
        t: Angle parameter.
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Differential arc length value.
    """
    r_t = spiral_config.start_radius + spiral_config.growth_rate * t
    return np.sqrt(r_t**2 + spiral_config.growth_rate**2)


def _calculate_arc_length(theta: float, spiral_config: config.SpiralConfig) -> float:
    """Calculate the arc length of the spiral from start_angle to theta.

    Args:
        theta: The angle in radians.
        spiral_config: Configuration parameters for the spiral.

    Returns:
        The arc length of the spiral from start_angle to theta.
    """
    return integrate.quad(
        lambda t: _arc_length_integrand(t, spiral_config),
        spiral_config.start_angle,
        theta,
    )[0]


def _find_theta_for_arc_length(
    target_arc_length: float, spiral_config: config.SpiralConfig
) -> float:
    """Find the theta value for a given arc length using numerical root finding.

    Args:
        target_arc_length: Target arc length.
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Angle theta corresponding to the arc length.
    """
    solution = optimize.root_scalar(
        lambda theta: _calculate_arc_length(theta, spiral_config) - target_arc_length,
        bracket=[spiral_config.start_angle, spiral_config.end_angle],
    )
    return solution.root


def _get_spiral_cache_key(spiral_config: config.SpiralConfig) -> str:
    """Generate a cache key based on spiral configuration parameters.

    Args:
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Hash string representing the configuration.
    """
    config_str = (
        f"{spiral_config.center_x}_{spiral_config.center_y}_"
        f"{spiral_config.start_radius}_{spiral_config.growth_rate}_"
        f"{spiral_config.start_angle}_{spiral_config.end_angle}_"
        f"{spiral_config.num_points}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_cache_path(spiral_config: config.SpiralConfig) -> pathlib.Path:
    """Get the cache file path for a given spiral configuration.

    Args:
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Path to the cache file.
    """
    cache_key = _get_spiral_cache_key(spiral_config)
    package_cache_dir = pathlib.Path(__file__).parent.parent / "cache"

    try:
        package_cache_dir.mkdir(parents=True, exist_ok=True)
        test_file = package_cache_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError):
        logger.warning(
            "Package cache directory is not writable. "
            "Cannot save reference spiral to cache."
        )

    return package_cache_dir / f"reference_spiral_{cache_key}.npy"


def _load_reference_spiral(spiral_config: config.SpiralConfig) -> np.ndarray | None:
    """Load a pre-computed reference spiral from disk.

    Args:
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Reference spiral array if found, None otherwise.
    """
    cache_path = _get_cache_path(spiral_config)

    if cache_path.exists():
        try:
            spiral = np.load(cache_path)
            logger.info(f"Loaded pre-computed reference spiral from {cache_path}")
            return spiral
        except Exception as e:
            logger.warning(f"Error loading cached spiral from {cache_path}: {e}")
            return None

    return None


def _compute_reference_spiral(
    spiral_config: config.SpiralConfig,
) -> np.ndarray:
    """Generate a reference spiral using numerical computation.

    This is the computation-heavy implementation that performs numerical integration and
    root finding to create equidistant points along the spiral.

    Args:
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Array with shape (N, 2) containing Cartesian coordinates of the spiral points.
    """
    total_arc_length = _calculate_arc_length(spiral_config.end_angle, spiral_config)

    arc_length_values = np.linspace(0, total_arc_length, spiral_config.num_points)

    theta_values = np.array(
        [_find_theta_for_arc_length(s, spiral_config) for s in arc_length_values]
    )

    r_values = spiral_config.start_radius + spiral_config.growth_rate * theta_values
    x_values = spiral_config.center_x + r_values * np.cos(theta_values)
    y_values = spiral_config.center_y + r_values * np.sin(theta_values)

    return np.column_stack((x_values, y_values))

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
        - Center coordinates: (cx, cy) = (spiral_config.center_x,
          spiral_config.center_y)
        - Start radius: a = spiral_config.start_radius
        - Growth rate: b = spiral_config.growth_rate
        - Total rotation: θ = spiral_config.end_angle - spiral_config.start_angle
        - Number of points: N = spiral_config.num_points

    Args:
        spiral_config: Configuration parameters for the spiral.

    Returns:
        Array with shape (N, 2) containing Cartesian coordinates of the spiral points.
    """
    cached_spiral = _load_reference_spiral(spiral_config)
    if cached_spiral is not None:
        return cached_spiral

    logger.info("No cached reference spiral found, generating new reference spiral...")
    spiral = _compute_reference_spiral(spiral_config)

    cache_path = _get_cache_path(spiral_config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving generated reference spiral to cache: {cache_path}")
    np.save(cache_path, spiral)

    return spiral
