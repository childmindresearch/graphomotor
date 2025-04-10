"""Feature extraction module for distance-based metrics in spiral drawing data."""

import numpy as np
from scipy import stats
from scipy.spatial import distance

from graphomotor.core import models


def _segment_data(data: np.ndarray, start_pct: float, end_pct: float) -> np.ndarray:
    """Extract segment of data based on percentage range.

    Args:
        data: Data to segment
        start_pct: Start percentage [0-1)
        end_pct: End percentage (0-1]

    Returns:
        Segmented data
    """
    if not (0 <= start_pct < end_pct <= 1):
        raise ValueError(
            "Percentages must be between 0 and 1, "
            "and start_pct must be less than end_pct"
        )
    num_samples = len(data)
    start_idx = int(start_pct * num_samples)
    end_idx = int(end_pct * num_samples)
    return data[start_idx:end_idx]


def calculate_hausdorff_metrics(
    spiral: models.Spiral, reference_spiral: np.ndarray
) -> dict:
    """Calculate Hausdorff distance metrics for a spiral object.

    This function computes multiple features based on the Hausdorff distance between a
    drawn spiral and a reference (ideal) spiral, as described in [1]. The Hausdorff
    distance measures the maximum distance of a set to the nearest point in the other
    set. This metric and its derivatives capture various aspects of the spatial
    relationship between the drawn and reference spirals. Calculated features include:
        - max_haus_dist: The maximum of the directed Hausdorff distances between the
            data points and the reference data points.
        - sum_haus_dist: The sum of the directed Hausdorff distances.
        - sum_haus_dist_time: The sum of the directed Hausdorff distances divided by
            the total drawing duration.
        - iqr_haus_dist: The interquartile range of the directed Hausdorff distances.
        - max_haus_dist_start: The maximum of the directed Hausdorff distances between
            the beginning segment (0% to 25%) of data points and the beginning segment
            of reference data points divided by the number of data points in the
            beginning segment.
        - max_haus_dist_end: The maximum of the directed Hausdorff distances in the
            ending segment (75% to 100%) of data points and the ending segment of
            reference data points divided by the number of data points in the ending
            segment.
        - max_haus_dist_mid: The maximum of the directed Hausdorff distances in the
            middle segment (15% to 85%) of data points and the ending segment of
            reference data points (this metric is not divided by the number of data
            points in the middle segment unlike previous ones).
        - max_haus_dist_mid_time: The maximum of the directed Hausdorff distances in
            the middle segment divided by the total drawing duration.

    Args:
        spiral: Spiral object with drawing data
        reference_spiral: Reference spiral data for comparison

    Returns:
        Dictionary containing Hausdorff distance-based features

    References:
        [1] Messan, Komi S et al. “Assessment of Smartphone-Based Spiral Tracing in
            Multiple Sclerosis Reveals Intra-Individual Reproducibility as a Major
            Determinant of the Clinical Utility of the Digital Test.” Frontiers in
            medical technology vol. 3 714682. 1 Feb. 2022, doi:10.3389/fmedt.2021.714682
    """
    spiral_data = np.column_stack((spiral.data["x"].values, spiral.data["y"].values))

    total_duration = spiral.data["seconds"].iloc[-1]

    start_segment_data = _segment_data(spiral_data, 0.0, 0.25)
    end_segment_data = _segment_data(spiral_data, 0.75, 1.0)
    mid_segment_data = _segment_data(spiral_data, 0.15, 0.85)

    if (
        len(start_segment_data) == 0
        or len(end_segment_data) == 0
        or len(mid_segment_data) == 0
    ):
        raise ValueError(
            "Segmented data is empty, check spiral data or segment percentages"
        )

    start_segment_ref = _segment_data(reference_spiral, 0.0, 0.25)
    end_segment_ref = _segment_data(reference_spiral, 0.75, 1.0)
    mid_segment_ref = _segment_data(reference_spiral, 0.15, 0.85)

    haus_dist = [
        distance.directed_hausdorff(spiral_data, reference_spiral)[0],
        distance.directed_hausdorff(reference_spiral, spiral_data)[0],
    ]
    haus_dist_start = [
        distance.directed_hausdorff(start_segment_data, start_segment_ref)[0],
        distance.directed_hausdorff(start_segment_ref, start_segment_data)[0],
    ]
    haus_dist_end = [
        distance.directed_hausdorff(end_segment_data, end_segment_ref)[0],
        distance.directed_hausdorff(end_segment_ref, end_segment_data)[0],
    ]
    haus_dist_mid = [
        distance.directed_hausdorff(mid_segment_data, mid_segment_ref)[0],
        distance.directed_hausdorff(mid_segment_ref, mid_segment_data)[0],
    ]

    return {
        "max_haus_dist": np.max(haus_dist),
        "sum_haus_dist": np.sum(haus_dist),
        "sum_haus_dist_time": np.sum(haus_dist) / total_duration,
        "iqr_haus_dist": stats.iqr(haus_dist),
        "max_haus_dist_start": np.max(haus_dist_start) / len(start_segment_data),
        "max_haus_dist_end": np.max(haus_dist_end) / len(end_segment_data),
        "max_haus_dist_mid": np.max(haus_dist_mid),
        "max_haus_dist_mid_time": np.max(haus_dist_mid) / total_duration,
    }
