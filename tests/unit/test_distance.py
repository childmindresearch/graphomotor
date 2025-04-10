"""Test cases for distance.py functions."""

import numpy as np
import pandas as pd
import pytest
import scipy.spatial.distance as dist
from scipy import stats

from graphomotor.core import models
from graphomotor.features import distance


def test_segment_data_valid() -> None:
    """Test that the data is segmented correctly."""
    data = np.array([[i, i] for i in range(100)])

    segment = distance._segment_data(data, 0.1, 0.3)
    assert len(segment) == 20
    assert segment[0][0] == 10
    assert segment[-1][0] == 29


@pytest.mark.parametrize(
    "start_pct,end_pct",
    [
        (-0.1, 0.5),
        (0.1, 1.1),
        (0.6, 0.5),
        (0.5, 0.5),
    ],
)
def test_segment_data_invalid(start_pct: float, end_pct: float) -> None:
    """Test that invalid percentages raise a ValueError."""
    data = np.array([[i, i] for i in range(100)])

    with pytest.raises(
        ValueError,
        match=(
            "Percentages must be between 0 and 1, "
            "and start_pct must be less than end_pct"
        ),
    ):
        distance._segment_data(data, start_pct, end_pct)


def test_calculate_hausdorff_metrics(
    valid_spiral: models.Spiral, reference_spiral: np.ndarray
) -> None:
    """Test that each Hausdorff metric is calculated."""
    metrics = distance.calculate_hausdorff_metrics(valid_spiral, reference_spiral)

    expected_metrics = [
        "max_haus_dist",
        "sum_haus_dist",
        "sum_haus_dist_time",
        "iqr_haus_dist",
        "max_haus_dist_start",
        "max_haus_dist_end",
        "max_haus_dist_mid",
        "max_haus_dist_mid_time",
    ]

    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)


def test_calculate_hausdorff_metrics_empty_segments(
    valid_spiral_data: pd.DataFrame,
    valid_spiral_metadata: dict,
    reference_spiral: np.ndarray,
) -> None:
    """Test that empty segments raise a ValueError."""
    small_spiral_data = valid_spiral_data.iloc[:3]
    small_spiral = models.Spiral(
        data=small_spiral_data,
        metadata=valid_spiral_metadata,
    )
    with pytest.raises(
        ValueError,
        match="Segmented data is empty, check spiral data or segment percentages",
    ):
        distance.calculate_hausdorff_metrics(small_spiral, reference_spiral)


def test_hausdorff_metrics_values(
    valid_spiral: models.Spiral, reference_spiral: np.ndarray
) -> None:
    """Test that Hausdorff metrics are calculated correctly."""
    metrics = distance.calculate_hausdorff_metrics(valid_spiral, reference_spiral)

    data = valid_spiral.data[["x", "y"]].values
    ref_data = reference_spiral

    total_duration = valid_spiral.data["seconds"].iloc[-1]

    data_start = data[: int(len(data) * 0.25)]
    data_end = data[int(len(data) * 0.75) :]
    data_mid = data[int(len(data) * 0.15) : int(len(data) * 0.85)]

    ref_data_start = ref_data[: int(len(ref_data) * 0.25)]
    ref_data_end = ref_data[int(len(ref_data) * 0.75) :]
    ref_data_mid = ref_data[int(len(ref_data) * 0.15) : int(len(ref_data) * 0.85)]

    dist_matrix = dist.cdist(data, ref_data, "euclidean")
    dist_matrix_start = dist.cdist(data_start, ref_data_start, "euclidean")
    dist_matrix_end = dist.cdist(data_end, ref_data_end, "euclidean")
    dist_matrix_mid = dist.cdist(data_mid, ref_data_mid, "euclidean")

    haus_dist = [
        np.max(np.min(dist_matrix, axis=0)),
        np.max(np.min(dist_matrix, axis=1)),
    ]
    haus_dist_start = [
        np.max(np.min(dist_matrix_start, axis=0)),
        np.max(np.min(dist_matrix_start, axis=1)),
    ]
    haus_dist_end = [
        np.max(np.min(dist_matrix_end, axis=0)),
        np.max(np.min(dist_matrix_end, axis=1)),
    ]
    haus_dist_mid = [
        np.max(np.min(dist_matrix_mid, axis=0)),
        np.max(np.min(dist_matrix_mid, axis=1)),
    ]

    assert metrics["max_haus_dist"] == np.max(haus_dist)
    assert metrics["sum_haus_dist"] == np.sum(haus_dist)
    assert metrics["sum_haus_dist_time"] == np.sum(haus_dist) / total_duration
    assert metrics["iqr_haus_dist"] == stats.iqr(haus_dist)
    assert metrics["max_haus_dist_start"] == np.max(haus_dist_start) / len(data_start)
    assert metrics["max_haus_dist_end"] == np.max(haus_dist_end) / len(data_end)
    assert metrics["max_haus_dist_mid"] == np.max(haus_dist_mid)
    assert metrics["max_haus_dist_mid_time"] == np.max(haus_dist_mid) / total_duration
