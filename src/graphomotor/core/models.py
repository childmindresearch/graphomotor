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
    ink_points: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

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
            Tuple of (ink_start_idx: int, ink_end_idx: int) if valid
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

        if (
            ink_start_idx is not None
            and ink_end_idx is not None
            and ink_end_idx > ink_start_idx
        ):
            self.ink_points = self.points.iloc[ink_start_idx : ink_end_idx + 1].copy()

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

    def calculate_velocity_metrics(self, ink_points: pd.DataFrame) -> None:
        """Get velocity metrics of a LineSegment.

        Args:
            self: LineSegment object to calculate velocities for.
            ink_points: DataFrame of ink points with 'x', 'y', and 'seconds' columns.
        """
        dx = np.diff(ink_points["x"].values)
        dy = np.diff(ink_points["y"].values)
        dt = np.diff(ink_points["seconds"].values)

        distances = np.sqrt(dx**2 + dy**2)
        self.distance = np.sum(distances)

        velocities = distances / dt
        self.velocities = velocities.tolist()

        self.mean_speed = np.mean(velocities)
        self.speed_variance = np.var(velocities)

        if len(velocities) >= 2:
            self.accelerations = np.diff(velocities).tolist()

        return
