"""Test cases for center_spiral.py functions."""

import typing

import numpy as np
import pytest

from graphomotor.core import models
from graphomotor.utils import center_spiral


def test_center_spiral_valid_type(
    perfect_spiral: models.Spiral, ref_spiral: np.ndarray
) -> None:
    """Test that center_spiral works with valid Spiral and reference spiral."""
    centered_perfect_spiral = center_spiral.center_spiral(perfect_spiral)
    centered_ref_spiral = center_spiral.center_spiral(ref_spiral)

    assert isinstance(centered_perfect_spiral, models.Spiral)
    assert centered_perfect_spiral.data["x"].iloc[0] == 0
    assert centered_perfect_spiral.data["y"].iloc[0] == 0

    assert isinstance(centered_ref_spiral, np.ndarray)
    assert np.array_equal(centered_ref_spiral[0], [0, 0])


@pytest.mark.parametrize(
    "invalid_input, expected_type_name",
    [
        ("invalid_type", "str"),
        (42, "int"),
        (None, "NoneType"),
        ([1, 2, 3], "list"),
        ({}, "dict"),
    ],
)
def test_center_spiral_invalid_type(
    invalid_input: object, expected_type_name: str
) -> None:
    """Test that center_spiral raises TypeError for invalid input types."""
    with pytest.raises(
        TypeError,
        match=f"Expected models.Spiral or np.ndarray, got {expected_type_name}",
    ):
        center_spiral.center_spiral(typing.cast(models.Spiral, invalid_input))
