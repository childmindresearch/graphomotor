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
        metadata: Dictionary containing metadata about the spiral:
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
class GridCell:
    """Represents a single rectangular region in a grid layout.

    Used to assign strokes to letter regions (Alphabet) or digit regions (DSYM).
    Boundary policy: a point on the exact boundary is considered inside the cell.

    Attributes:
        x_min: Left boundary of the cell.
        x_max: Right boundary of the cell.
        y_min: Bottom boundary of the cell.
        y_max: Top boundary of the cell.
        index: Position of the cell in the grid (0-based).
        label: Display label for the cell (e.g., 'A', 'B', '1').
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    index: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        """Validate that min bounds are strictly less than max bounds.

        Raises:
            ValueError: If x_min >= x_max or y_min >= y_max.
        """
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be less than x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be less than y_max ({self.y_max})"
            )

    def contains_points(self, points: pd.DataFrame) -> bool:
        """Check if a stroke belongs to this cell based on its centroid.

        Computes the centroid (mean x, mean y) of the provided points and checks
        whether it falls within the cell boundaries. Uses half-open intervals
        [min, max) to prevent double-assignment on shared grid edges.

        Args:
            points: DataFrame with 'x' and 'y' columns representing a stroke.

        Returns:
            True if the stroke centroid is within the cell, False otherwise.
        """
        centroid_x = points["x"].mean()
        centroid_y = points["y"].mean()
        return (
            self.x_min <= centroid_x < self.x_max
            and self.y_min <= centroid_y < self.y_max
        )


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
