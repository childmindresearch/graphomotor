"""Internal data class for spiral drawing data."""

import dataclasses
import datetime
import typing

import numpy as np
import pandas as pd
import pydantic

from graphomotor.features import base_metrics


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
    ) -> dict[str, typing.Callable[[], dict[str, float]]]:
        """Get all feature extractors with appropriate inputs.

        Args:
            spiral: The spiral data to extract features from.
            reference_spiral: Reference spiral for comparison-based metrics.

        Returns:
            Dictionary mapping category names to their feature extractor functions.
        """
        # Importing feature modules here to avoid circular imports.
        from graphomotor.features.spiral import distance, drawing_error, velocity

        return {
            cls.DURATION: lambda: base_metrics.get_task_duration(spiral),
            cls.VELOCITY: lambda: velocity.calculate_velocity_metrics(spiral),
            cls.HAUSDORFF: lambda: distance.calculate_hausdorff_metrics(
                spiral, reference_spiral
            ),
            cls.AUC: lambda: drawing_error.calculate_area_under_curve(
                spiral, reference_spiral
            ),
        }


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
    velocities: typing.List[float] = dataclasses.field(default_factory=list)
    accelerations: typing.List[float] = dataclasses.field(default_factory=list)


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
