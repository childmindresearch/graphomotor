"""Utility functions for centering a spiral."""

from graphomotor.core import config, models


def center_spiral(spiral: models.Spiral) -> models.Spiral:
    """Center a spiral by translating it to the origin.

    Args:
        spiral: Spiral object containing spiral data.

    Returns:
        Spiral object with centered spiral data.
    """
    spiral_config = config.SpiralConfig()
    spiral.data["x"] -= spiral_config.center_x
    spiral.data["y"] -= spiral_config.center_y

    return spiral
