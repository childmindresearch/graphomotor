"""Test cases for reference_spiral.py functions."""

import numpy as np

from graphomotor.utils import reference_spiral


def test_generate_reference_spiral() -> None:
    """Test the generation of a reference spiral."""
    spiral = reference_spiral.generate_reference_spiral()
    arc_lengths = np.linalg.norm(spiral[1:] - spiral[:-1], axis=1)
    mean_arc_length = np.mean(arc_lengths)

    expected_mean_arc_length = reference_spiral._calculate_arc_length(
        reference_spiral._SPIRAL_END_ANGLE
    ) / (reference_spiral._SPIRAL_NUM_POINTS - 1)

    assert isinstance(spiral, np.ndarray)
    assert spiral.shape == (reference_spiral._SPIRAL_NUM_POINTS, 2)
    assert np.array_equal(
        spiral[0],
        [reference_spiral._SPIRAL_CENTER_X, reference_spiral._SPIRAL_CENTER_Y],
    )
    assert np.allclose(
        spiral[-1],
        [
            reference_spiral._SPIRAL_CENTER_X
            + reference_spiral._SPIRAL_GROWTH_RATE * reference_spiral._SPIRAL_END_ANGLE,
            reference_spiral._SPIRAL_CENTER_Y,
        ],
        atol=1e-8,
    )
    assert np.allclose(arc_lengths, mean_arc_length, rtol=1e-3)
    assert np.isclose(mean_arc_length, expected_mean_arc_length, rtol=1e-6)
