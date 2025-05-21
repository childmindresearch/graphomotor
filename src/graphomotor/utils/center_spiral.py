"""Utility functions for centering a spiral."""

import typing

import numpy as np

from graphomotor.core import config, models


@typing.overload
def center_spiral(spiral: models.Spiral) -> models.Spiral: ...
@typing.overload
def center_spiral(spiral: np.ndarray) -> np.ndarray: ...
def center_spiral(spiral):
    """Center a spiral by translating it to the origin.

    Args:
        spiral: Either a Spiral object containing spiral data or a NumPy array
               with shape (N, 2) containing (x, y) coordinates.

    Returns:
        The centered spiral of the same type as the input.

    Raises:
        TypeError: If the input is neither a Spiral object nor a NumPy array.
    """
    spiral_config = config.SpiralConfig()

    if isinstance(spiral, models.Spiral):
        spiral.data["x"] -= spiral_config.center_x
        spiral.data["y"] -= spiral_config.center_y
        return spiral
    elif isinstance(spiral, np.ndarray):
        spiral[:, 0] -= spiral_config.center_x
        spiral[:, 1] -= spiral_config.center_y
        return spiral
    else:
        raise TypeError(
            f"Expected models.Spiral or np.ndarray, got {type(spiral).__name__}"
        )
