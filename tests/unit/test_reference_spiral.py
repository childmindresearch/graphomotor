"""Test cases for reference_spiral.py functions."""

import numpy as np

from graphomotor.core import config
from graphomotor.utils import generate_reference_spiral


def test_generate_reference_spiral() -> None:
    """Test the generation of a reference spiral."""
    expected_mean_arc_length = generate_reference_spiral._calculate_arc_length(
        config._SpiralConfig.SPIRAL_END_ANGLE
    ) / (config._SpiralConfig.SPIRAL_NUM_POINTS - 1)

    spiral = generate_reference_spiral.generate_reference_spiral()
    arc_lengths = np.linalg.norm(spiral[1:] - spiral[:-1], axis=1)
    mean_arc_length = np.mean(arc_lengths)

    assert isinstance(spiral, np.ndarray)
    assert spiral.shape == (config._SpiralConfig.SPIRAL_NUM_POINTS, 2)
    assert np.array_equal(
        spiral[0],
        [config._SpiralConfig.SPIRAL_CENTER_X, config._SpiralConfig.SPIRAL_CENTER_Y],
    )
    assert np.allclose(
        spiral[-1],
        [
            config._SpiralConfig.SPIRAL_CENTER_X
            + config._SpiralConfig.SPIRAL_GROWTH_RATE
            * config._SpiralConfig.SPIRAL_END_ANGLE,
            config._SpiralConfig.SPIRAL_CENTER_Y,
        ],
        atol=0,
        rtol=1e-8,
    )
    assert np.allclose(arc_lengths, mean_arc_length, atol=0, rtol=1e-3)
    assert np.isclose(mean_arc_length, expected_mean_arc_length, atol=0, rtol=1e-6)
