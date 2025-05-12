"""Utility functions for centering a spiral."""

from graphomotor.core import config, models


def center_spiral(spiral: models.Spiral) -> models.Spiral:
    """Center a spiral by translating it to the origin.

    Args:
        spiral: Spiral object containing spiral data.

    Returns:
        Spiral object with centered spiral data.
    """
    spiral.data["x"] -= config._SpiralConfig.SPIRAL_CENTER_X
    spiral.data["y"] -= config._SpiralConfig.SPIRAL_CENTER_Y

    return spiral
