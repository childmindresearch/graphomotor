"""Feature extraction modules for Spiral graphomotor data analysis.

This module contains specialized extractors for computing clinically relevant metrics
across four categories for spiral data: velocity (15 features), distance (8 features),
drawing error (1 feature), and time (1 feature). Each feature extractor follows a
standardized interface and returns a dictionary of computed metrics with descriptive
keys.

Available Features
------------------
The graphomotor pipeline extracts 25 clinically validated features organized into
four categories. These features quantify different aspects of drawing performance
and motor control:

**Distance-based Features (8 features):**

These features use Hausdorff distance to measure spatial accuracy between drawn and
reference spirals. The Hausdorff distance captures the maximum distance from any point
in one set to the nearest point in another set.

- **hausdorff_distance_maximum**: Maximum directed Hausdorff distance between drawn
  and reference spiral points
- **hausdorff_distance_sum**: Sum of directed Hausdorff distances
- **hausdorff_distance_sum_per_second**: Sum of Hausdorff distances normalized by
  total drawing duration
- **hausdorff_distance_interquartile_range**: Interquartile range of directed
  Hausdorff distances, measuring variability in spatial accuracy
- **hausdorff_distance_start_segment_maximum_normalized**: Maximum Hausdorff distance
  in beginning segment (0-25%) normalized by segment length
- **hausdorff_distance_end_segment_maximum_normalized**: Maximum Hausdorff distance
  in ending segment (75-100%) normalized by segment length
- **hausdorff_distance_middle_segment_maximum**: Maximum Hausdorff distance in middle
  segment (15-85%) without normalization
- **hausdorff_distance_middle_segment_maximum_per_second**: Middle segment maximum
  Hausdorff distance normalized by total drawing duration

**Velocity-based Features (15 features):**

These features analyze movement dynamics by computing velocity in three coordinate
systems, each providing 5 statistical measures.

*Linear velocity (5 features):*
Euclidean velocity magnitude in Cartesian coordinates (pixels/second).

- **linear_velocity_sum**: Sum of absolute linear velocity values
- **linear_velocity_median**: Median of absolute linear velocity values
- **linear_velocity_coefficient_of_variation**: Ratio of standard deviation to mean,
  measuring velocity consistency
- **linear_velocity_skewness**: Asymmetry of velocity distribution (positive =
  right-skewed)
- **linear_velocity_kurtosis**: Tailedness of velocity distribution (higher = more
  extreme values)

*Radial velocity (5 features):*
Rate of change in distance from center (pixels/second).

- **radial_velocity_sum**: Sum of absolute radial velocity values
- **radial_velocity_median**: Median of absolute radial velocity values
- **radial_velocity_coefficient_of_variation**: Consistency of radial movement speed
- **radial_velocity_skewness**: Asymmetry of radial velocity distribution (positive =
  right-skewed)
- **radial_velocity_kurtosis**: Tailedness of radial velocity distribution
  (higher = more extreme values)

*Angular velocity (5 features):*
Rate of angular rotation around center (radians/second).

- **angular_velocity_sum**: Sum of absolute angular velocity values
- **angular_velocity_median**: Median of absolute angular velocity values
- **angular_velocity_coefficient_of_variation**: Consistency of rotational speed
- **angular_velocity_skewness**: Asymmetry of angular velocity distribution (positive =
  right-skewed)
- **angular_velocity_kurtosis**: Tailedness of angular velocity distribution
  (higher = more extreme values)

**Time-based Features (1 feature):**

- **duration**: Total time taken to complete the spiral drawing task (seconds)

**Drawing Error Features (1 feature):**

- **area_under_curve**: Total enclosed area between drawn and reference spiral paths,
  computed using polygon intersection algorithms. Lower values indicate better
  adherence to the template spiral.
"""
