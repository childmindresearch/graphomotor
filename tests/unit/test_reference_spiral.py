"""Test cases for reference_spiral.py functions."""

import numpy as np

from graphomotor.utils import reference_spiral


def test_generate_reference_spiral() -> None:
    """Test the generation of a reference spiral."""
    spiral = reference_spiral.generate_reference_spiral()
    assert isinstance(spiral, np.ndarray)
    assert spiral.shape == (10000, 2)
    assert np.array_equal(spiral[0], [50, 50])
    assert np.allclose(spiral[-1], [50 + 1.075 * 8 * np.pi, 50], atol=1e-8)

    distances = np.linalg.norm(np.diff(spiral, axis=0), axis=1)
    assert np.allclose(distances, distances[0], atol=1e-4)
