"""Feature extraction module for distance-based metrics in spiral drawing data."""

import numpy as np
from scipy import stats
from scipy.spatial import distance

from graphomotor.core import models


def _segment_data(data: np.ndarray, start_prop: float, end_prop: float) -> np.ndarray:
    """Extract segment of data based on given proportion range.

    Args:
        data: Data to segment.
        start_prop: Start proportion, [0-1).
        end_prop: End proportion, (0-1].

    Returns:
        Segmented data.
    """
    if not (0 <= start_prop < end_prop <= 1):
        raise ValueError(
            "Proportions must be between 0 and 1, "
            "and start_prop must be less than end_prop"
        )
    num_samples = len(data)
    start_idx = int(start_prop * num_samples)
    end_idx = int(end_prop * num_samples)
    return data[start_idx:end_idx]


def calculate_hausdorff_metrics(
    spiral: models.Spiral, reference_spiral: np.ndarray
) -> dict[str, float]:
    """Calculate Hausdorff distance metrics for a spiral object.

    This function computes multiple features based on the Hausdorff distance between a
    drawn spiral and a reference (ideal) spiral, as described in [1]. Implementation
    is based on the original R script provided with the publication. The Hausdorff
    distance measures the maximum distance of a set to the nearest point in the other
    set. This metric and its derivatives capture various aspects of the spatial
    relationship between the drawn and reference spirals. Calculated features include:
        - hausdorff_distance_maximum: The maximum of the directed Hausdorff distances
            between the data points and the reference data points,
        - hausdorff_distance_sum: The sum of the directed Hausdorff distances,
        - hausdorff_distance_sum_per_second: The sum of the directed Hausdorff distances
            divided by the total drawing duration,
        - hausdorff_distance_interquartile_range: The interquartile range of the
            directed Hausdorff distances,
        - hausdorff_distance_start_segment_maximum_normalized: The maximum of the
            directed Hausdorff distances between the beginning segment (0% to 25%) of
            data points and the beginning segment of reference data points divided by
            the number of data points in the beginning segment,
        - hausdorff_distance_end_segment_maximum_normalized: The maximum of the directed
            Hausdorff distances in the ending segment (75% to 100%) of data points and
            the ending segment of reference data points divided by the number of data
            points in the ending segment,
        - hausdorff_distance_middle_segment_maximum: The maximum of the directed
            Hausdorff distances in the middle segment (15% to 85%) of data points and
            the ending segment of reference data points (this metric is not divided by
            the number of data points in the middle segment unlike previous ones),
        - hausdorff_distance_middle_segment_maximum_per_second: The maximum of the
            directed Hausdorff distances in the middle segment divided by the total
            drawing duration.

    Args:
        spiral: Spiral object with drawing data.
        reference_spiral: Reference spiral data for comparison.

    Returns:
        Dictionary containing Hausdorff distance-based features.

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
        "hausdorff_distance_maximum": np.max(haus_dist),
        "hausdorff_distance_sum": np.sum(haus_dist),
        "hausdorff_distance_sum_per_second": np.sum(haus_dist) / total_duration,
        "hausdorff_distance_interquartile_range": stats.iqr(haus_dist),
        "hausdorff_distance_start_segment_maximum_normalized": np.max(haus_dist_start)
        / len(start_segment_data),
        "hausdorff_distance_end_segment_maximum_normalized": np.max(haus_dist_end)
        / len(end_segment_data),
        "hausdorff_distance_middle_segment_maximum": np.max(haus_dist_mid),
        "hausdorff_distance_middle_segment_maximum_per_second": np.max(haus_dist_mid)
        / total_duration,
    }
