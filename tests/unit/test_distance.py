"""Test cases for distance.py functions."""

import numpy as np
import pytest

from graphomotor.core import models
from graphomotor.features import distance


def test_segment_data_valid() -> None:
    """Test that the data is segmented correctly."""
    data = np.array([[i, i] for i in range(100)])

    segment = distance._segment_data(data, 0.1, 0.3)
    assert len(segment) == 20
    assert segment[0][0] == 10
    assert segment[-1][0] == 29


def test_segment_data_invalid() -> None:
    """Test that invalid percentages raise a ValueError."""
    data = np.array([[i, i] for i in range(100)])

    with pytest.raises(
        ValueError,
        match=(
            "Proportions must be between 0 and 1, "
            "and start_prop must be less than end_prop"
        ),
    ):
        distance._segment_data(data, 0.6, 0.5)


def test_calculate_hausdorff_metrics(
    valid_spiral: models.Spiral, ref_spiral: np.ndarray
) -> None:
    """Test that each Hausdorff metric is calculated."""
    metrics = distance.calculate_hausdorff_metrics(valid_spiral, ref_spiral)

    expected_metrics = {
        "hausdorff_distance_maximum",
        "hausdorff_distance_sum",
        "hausdorff_distance_sum_per_second",
        "hausdorff_distance_interquartile_range",
        "hausdorff_distance_start_segment_maximum_normalized",
        "hausdorff_distance_end_segment_maximum_normalized",
        "hausdorff_distance_middle_segment_maximum",
        "hausdorff_distance_middle_segment_maximum_per_second",
    }

    assert set(metrics.keys()) == expected_metrics
    assert all(isinstance(value, float) for value in metrics.values())
