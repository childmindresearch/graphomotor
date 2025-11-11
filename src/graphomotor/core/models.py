"""Internal data class for spiral drawing data."""

import dataclasses
import datetime
import typing

import numpy as np
import pandas as pd
import pydantic


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
        from graphomotor.features import distance, drawing_error, time, velocity

        return {
            cls.DURATION: lambda: time.get_task_duration(spiral),
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
    """Represents a line drawn between two circles."""

    start_label: str
    end_label: str
    points: pd.DataFrame
    is_error: bool
    line_number: int

    # Computed metrics
    ink_time: float = 0.0
    think_time: float = 0.0
    think_circle_label: str = ""  # Which circle the think time applies to
    distance: float = 0.0  # Total distance drawn outside circles
    mean_speed: float = 0.0
    speed_variance: float = 0.0
    path_optimality: float = 0.0
    smoothness: float = 0.0  # Based on curvature changes
    hesitation_count: int = 0
    hesitation_duration: float = 0.0
    velocities: typing.List[float] = dataclasses.field(default_factory=list)
    accelerations: typing.List[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CircleTarget:
    """Represents a target circle in the drawing task."""

    order: int
    label: str
    center_x: float
    center_y: float
    radius: float

    def contains_point(self, x: float, y: float, tolerance: float = 1.5) -> bool:
        """Check if a point is within the circle (with tolerance multiplier)."""
        distance = np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)
        return distance <= (self.radius * tolerance)
