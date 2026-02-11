"""Internal data classes for drawing data."""

import dataclasses
import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydantic
import scipy.spatial.distance as dist


class Drawing(pydantic.BaseModel):
    """Class representing a drawing task, encapsulating both raw data and metadata.

    Attributes:
        data: DataFrame containing drawing data with required columns (line_number, x,
            y, UTC_Timestamp, seconds).
        task_name: Name of the drawing task (e.g., 'spiral', 'trails', etc.).
        metadata: Dictionary containing metadata about the drawing:
            - id: Unique identifier for the participant,
            - hand: Hand used ('Dom' for dominant, 'NonDom' for non-dominant),
            - task: Task name,
            - start_time: Start time of drawing,
            - source_path: Path to the source CSV file.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    task_name: str
    metadata: dict[str, str | datetime.datetime]

    @pydantic.field_validator("data")
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        """Validate that DataFrame is not empty.

        Args:
            cls: The class.
            v: The dataframe to validate.

        Returns:
            The dataframe if it is not empty.

        Raises:
            ValueError: If the dataframe is empty.
        """
        if v.empty:
            raise ValueError("DataFrame is empty")

        return v

    @pydantic.field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict) -> dict:
        """Validate metadata dictionary for required keys and correct data types.

        Args:
            cls: The class.
            v: The metadata dictionary to validate.

        Returns:
            The metadata dictionary if it is valid.

        Raises:
            ValueError: If the metadata dictionary has invalid values.
        """
        if not v["id"].startswith("5"):
            raise ValueError("'id' must start with digit 5")
        if len(v["id"]) != 7:
            raise ValueError("'id' must be 7 digits long")

        return v


class SpiralFeatureCategories:
    """Class to hold valid feature categories for Graphomotor."""

    DURATION = "duration"
    VELOCITY = "velocity"
    HAUSDORFF = "hausdorff"
    AUC = "AUC"

    @classmethod
    def all(cls) -> set[str]:
        """Return all valid feature categories."""
        return {
            cls.DURATION,
            cls.VELOCITY,
            cls.HAUSDORFF,
            cls.AUC,
        }

    @classmethod
    def get_extractors(
        cls, spiral: Drawing, reference_spiral: np.ndarray
    ) -> dict[str, Callable[[], dict[str, float]]]:
        """Get all feature extractors with appropriate inputs.

        Args:
            spiral: The spiral data to extract features from.
            reference_spiral: Reference spiral for comparison-based metrics.

        Returns:
            Dictionary mapping category names to their feature extractor functions.
        """
        # Importing feature modules here to avoid circular imports.
        from graphomotor.features import shared_features
        from graphomotor.features.spiral import (
            distance,
            drawing_error,
            velocity,
        )

        return {
            cls.DURATION: lambda: shared_features.get_task_duration(spiral),
            cls.VELOCITY: lambda: velocity.calculate_velocity_metrics(spiral),
            cls.HAUSDORFF: lambda: distance.calculate_hausdorff_metrics(
                spiral, reference_spiral
            ),
            cls.AUC: lambda: drawing_error.calculate_area_under_curve(
                spiral, reference_spiral
            ),
        }


@dataclasses.dataclass
class CircleTarget:
    """Represents a target circle in the drawing task.

    Attributes:
        order: The order of the circle in the sequence.
        label: The label of the circle.
        center_x: The x-coordinate of the circle's center.
        center_y: The y-coordinate of the circle's center.
        radius: The radius of the circle.
    """

    order: int
    label: str
    center_x: float
    center_y: float
    radius: float

    def contains_point(self, x: float, y: float, tolerance: float = 1.5) -> bool:
        """Check if a point is within the circle (with tolerance multiplier).

        Args:
            x: X coordinate of the point.
            y: Y coordinate of the point.
            tolerance: Multiplier for the radius to define tolerance boundary.

        Returns:
            True if the point is within the circle (with tolerance), False otherwise.
        """
        distance = np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)
        return distance <= (self.radius * tolerance)


@dataclasses.dataclass
class LineSegment:
    """Represents a line drawn between two circles.

    Attributes:
        start_label: Label of the starting circle.
        end_label: Label of the ending circle.
        points: DataFrame containing the points in the line segment.
        is_error: Whether the line segment is an error (missed target).
        line_number: The line number of the segment.

        Calculated features:
        ink_time: Time spent drawing the line segment.
        think_time: Time spent thinking before drawing the line segment.
        think_circle_label: Label of the circle associated with think time.
        distance: Total distance drawn outside circles.
        mean_speed: Mean speed of drawing the line segment.
        speed_variance: Variance of speed during the line segment.
        path_optimality: Ratio of actual path length to optimal path length.
        smoothness: Smoothness of the line segment based on curvature changes.
        hesitation_count: Number of hesitations during the line segment.
        hesitation_duration: Total duration of hesitations during the line segment.
        velocities: List of velocities at each point in the line segment.
        accelerations: List of accelerations at each point in the line segment.
    """

    start_label: str
    end_label: str
    points: pd.DataFrame
    is_error: bool
    line_number: int

    ink_time: float = 0.0
    think_time: float = 0.0
    think_circle_label: str = ""
    distance: float = 0.0
    mean_speed: float = 0.0
    speed_variance: float = 0.0
    path_optimality: float = 0.0
    smoothness: float = 0.0
    hesitation_count: int = 0
    hesitation_duration: float = 0.0
    velocities: List[float] = dataclasses.field(default_factory=list)
    accelerations: List[float] = dataclasses.field(default_factory=list)
    ink_points: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)

    def valid_ink_trajectory(
        self,
        start_circle: CircleTarget,
        end_circle: CircleTarget,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Determine whether an ink trajectory exists from a start to end circle.

        An "ink trajectory" is defined as the first contiguous sequence of
        points that:
        1. Begins **after** the pen leaves the start circle, and
        2. Ends when the pen first enters the end circle.

        The function scans point-by-point in order. The ink start index is the
        first point whose (x, y) location is *outside* the start circle. The
        ink end index is the first subsequent point whose (x, y) location falls
        *inside* the end circle. If either of these conditions never occurs,
        the trajectory is considered invalid. If a valid trajectory is found,
        the ink_points attribute is updated to contain only the points within
        this trajectory.

        Args:
            points: DataFrame of points with 'x' and 'y' columns.
            start_circle: CircleTarget representing the start circle.
            end_circle: CircleTarget representing the end circle.

        Returns:
            Tuple of (ink_start_idx: Optional[int], ink_end_idx: Optional[int]) if valid
            trajectory exists, else (None, None).
        """
        ink_start_idx = None
        ink_end_idx = None

        for idx, row in self.points.iterrows():
            if (
                not start_circle.contains_point(row["x"], row["y"])
                and ink_start_idx is None
            ):
                ink_start_idx = idx

            if ink_start_idx is not None and end_circle.contains_point(
                row["x"], row["y"]
            ):
                ink_end_idx = idx
                break

        return ink_start_idx, ink_end_idx

    def calculate_path_optimality(
        self,
        start_circle: CircleTarget,
        end_circle: CircleTarget,
    ) -> None:
        """Calculate path optimality ratio.

        The default value for path optimality in the LineSegment object is 0.0. This
        function updates the path_optimality attribute of the LineSegment object based
        on the optimal distance between the start and end circles, adjusted for their
        radii. If the optimal distance is less than or equal to zero, the path
        optimality remains 0.0.

        Args:
            segment: LineSegment object for which to calculate path optimality.
            start_circle: CircleTarget representing the start circle.
            end_circle: CircleTarget representing the end circle.

        Returns:
            Path optimality ratio.
        """
        optimal_distance = (
            dist.euclidean(
                [start_circle.center_x, start_circle.center_y],
                [end_circle.center_x, end_circle.center_y],
            )
            - start_circle.radius
            - end_circle.radius
        )

        if optimal_distance > 0:
            self.path_optimality = optimal_distance / self.distance
        return

    def calculate_velocity_metrics(self) -> None:
        """Get velocity metrics of a LineSegment.

        Args:
            self: LineSegment object to calculate velocities for.
            ink_points: DataFrame containing the ink points for the line segment.
        """
        dx = np.diff(self.ink_points["x"].values)
        dy = np.diff(self.ink_points["y"].values)
        dt = np.diff(self.ink_points["seconds"].values)

        distances = np.sqrt(dx**2 + dy**2)
        self.distance = np.sum(distances)

        velocities = distances / dt
        self.velocities = velocities.tolist()

        self.mean_speed = np.mean(velocities)
        self.speed_variance = np.var(velocities)

        if len(velocities) >= 2:
            self.accelerations = np.diff(velocities).tolist()

        return

    def detect_hesitations(self, threshold_percentile: int = 20) -> None:
        """Detect hesitations as periods of significantly reduced velocity.

        This function defines a hesitation as any period where the velocity falls below
        a certain threshold, which is determined by the specified percentile of the
        velocity distribution. It counts the number of distinct hesitation periods and
        adds 1 if the line starts with a hesitation. It also calculates the total
        duration of hesitations based on the number of points that fall below the
        threshold and the time interval between points.

        Args:
            ink_points: DataFrame containing the ink points for the line segment.
            threshold_percentile: Percentile to determine the velocity threshold for
                hesitations (default is 20, meaning the bottom 20% of velocities are
                considered hesitations).
        """
        if len(self.velocities) < 3:
            return

        dt = np.diff(self.ink_points["seconds"].values)

        threshold = np.percentile(self.velocities, threshold_percentile)
        hesitations = self.velocities < threshold

        hesitation_changes = np.diff(hesitations.astype(int))
        hesitation_starts = np.where(hesitation_changes == 1)[0] + 1
        hesitation_count = len(hesitation_starts)

        if hesitations[0]:
            hesitation_count += 1

        hesitation_duration = np.sum(hesitations) * dt[0]

        self.hesitation_count = hesitation_count
        self.hesitation_duration = hesitation_duration

        return

    def calculate_smoothness(self) -> None:
        """Calculate path smoothness based on Root Mean Square (RMS) curvature.

        Represents the curvature per unit arc length.
        Lower values indicate smoother drawings. Penalizes sharp corners (e.g.,
        90° turns) and noisy corrections. Normalized by arc length to reduce
        sampling-rate dependence.
        """
        if len(self.ink_points) < 3:
            return 0.0

        xy = self.ink_points[["x", "y"]].to_numpy()

        forward_vector = xy[1:-1] - xy[:-2]
        backward_vector = xy[2:] - xy[1:-1]

        forward_norm = np.linalg.norm(forward_vector, axis=1)
        backward_norm = np.linalg.norm(backward_vector, axis=1)

        valid = (forward_norm > 0) & (backward_norm > 0)
        if not np.any(valid):
            return

        valid_forward_vector = forward_vector[valid]
        valid_backward_vector = backward_vector[valid]
        valid_forward_norm = forward_norm[valid]
        valid_backward_norm = backward_norm[valid]

        cos_angle = (valid_forward_vector * valid_backward_vector).sum(axis=1) / (
            valid_forward_norm * valid_backward_norm
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angles = np.arccos(cos_angle)

        avg_segment_length = (valid_forward_norm + valid_backward_norm) / 2.0
        curvatures = angles / avg_segment_length

        self.smoothness = float(np.sqrt(np.mean(curvatures**2)))

        return

    def compute_segment_metrics(
        self, circles: dict[str, dict[str, CircleTarget]], trail_id: str
    ) -> None:
        """Compute all metrics for a line segment.

        This function computes various metrics for the line segment, including ink time,
        velocity metrics, path optimality, smoothness, and hesitation detection. It
        first determines the valid ink trajectory between the start and end circles. If
        a valid trajectory is found, it updates the ink_points attribute and calculates
        the metrics.

        Args:
            circles: A dictionary mapping each trail type to dictionaries of
                CircleTarget instances (output of load_scaled_circles in config).
            trail_id: Trail identifier for circle lookup.
        """
        circles = circles[trail_id]
        points = self.points.copy()

        if len(points) < 2:
            return

        if self.start_label not in circles or self.end_label not in circles:
            return

        start_circle = circles[self.start_label]
        end_circle = circles[self.end_label]

        ink_start_idx, ink_end_idx = self.valid_ink_trajectory(start_circle, end_circle)

        if (
            ink_start_idx is not None
            and ink_end_idx is not None
            and ink_end_idx > ink_start_idx
        ):
            self.ink_points = self.points.iloc[ink_start_idx : ink_end_idx + 1].copy()

            if len(self.ink_points) >= 2:
                ink_start = self.ink_points.iloc[0]["seconds"]
                ink_end = self.ink_points.iloc[-1]["seconds"]
                self.ink_time = ink_end - ink_start

                self.calculate_velocity_metrics()

                self.calculate_path_optimality(start_circle, end_circle)

                self.calculate_smoothness()

                self.detect_hesitations()

        elif ink_start_idx is not None:
            self.ink_points = points.iloc[ink_start_idx:].copy()
            if len(self.ink_points) >= 2:
                self.ink_time = (
                    self.ink_points.iloc[-1]["seconds"]
                    - self.ink_points.iloc[0]["seconds"]
                )
        return
