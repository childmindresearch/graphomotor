"""Utility functions for centering a spiral."""

import typing

import numpy as np

from graphomotor.core import config, models


@typing.overload
def center_spiral(spiral: models.Drawing) -> models.Drawing: ...
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

    if isinstance(spiral, models.Drawing):
        centered_spiral = models.Drawing(
            data=spiral.data.copy(), metadata=spiral.metadata.copy()
        )
        centered_spiral.data["x"] -= spiral_config.center_x
        centered_spiral.data["y"] -= spiral_config.center_y
        return centered_spiral
    elif isinstance(spiral, np.ndarray):
        centered_spiral = spiral.copy()
        centered_spiral[:, 0] -= spiral_config.center_x
        centered_spiral[:, 1] -= spiral_config.center_y
        return centered_spiral
    else:
        raise TypeError(
            f"Expected models.Spiral or np.ndarray, got {type(spiral).__name__}"
        )
