"""Test cases for generate_reference_spiral.py functions."""

import numpy as np

from graphomotor.core import config
from graphomotor.utils import generate_reference_spiral


def test_generate_reference_spiral() -> None:
    """Test the generation of a reference spiral."""
    spiral_config = config.SpiralConfig()
    expected_mean_arc_length = generate_reference_spiral._calculate_arc_length(
        spiral_config.end_angle, spiral_config
    ) / (spiral_config.num_points - 1)

    spiral = generate_reference_spiral.generate_reference_spiral(spiral_config)
    arc_lengths = np.linalg.norm(spiral[1:] - spiral[:-1], axis=1)
    mean_arc_length = np.mean(arc_lengths)

    assert isinstance(spiral, np.ndarray)
    assert spiral.shape == (spiral_config.num_points, 2)
    assert np.array_equal(
        spiral[0],
        [spiral_config.center_x, spiral_config.center_y],
    )
    assert np.allclose(
        spiral[-1],
        [
            spiral_config.center_x
            + spiral_config.growth_rate * spiral_config.end_angle,
            spiral_config.center_y,
        ],
        atol=0,
        rtol=1e-8,
    )
    assert np.allclose(arc_lengths, mean_arc_length, atol=0, rtol=1e-3)
    assert np.isclose(mean_arc_length, expected_mean_arc_length, atol=0, rtol=1e-6)
