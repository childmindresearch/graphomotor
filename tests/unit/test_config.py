"""Test cases for config.py functions."""

import numpy as np
import pytest

from graphomotor.core import config


def test_generate_reference_spiral() -> None:
    """Test the generation of a reference spiral."""
    spiral = config.generate_reference_spiral()
    assert isinstance(spiral, np.ndarray)
    assert spiral.shape == (10000, 2)
    assert spiral[0] == pytest.approx([50, 50])
    assert spiral[-1] == pytest.approx([50 + 1.075 * 8 * np.pi, 50])
