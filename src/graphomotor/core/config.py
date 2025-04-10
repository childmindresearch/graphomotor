"""Configuration module for the graphomotor repository."""

import numpy as np


def generate_reference_spiral() -> np.ndarray:
    """Generates a reference spiral for feature extraction purposes."""
    cx, cy = (50, 50)  # center of the spiral
    a = 0  # starting radius
    b = 1.075  # growth rate
    num_points = 10000
    spiral_length = 8 * np.pi  # spiral makes 4 full rotations
    theta = np.linspace(0, spiral_length, num_points)
    r = a + b * theta
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return np.column_stack((x, y))
