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
class GridCell:
    """Represents a single rectangular region in a grid layout.

    Used to assign strokes to letter regions (Alphabet) or digit regions (DSYM).
    Boundary policy: uses half-open intervals [min, max) to prevent
    double-assignment on shared grid edges. The outer grid boundaries should be
    padded (e.g., by 0.1) at the Grid level so that centroids on the outermost
    edge are not excluded.

    Attributes:
        x_min: Left boundary of the cell.
        x_max: Right boundary of the cell.
        y_min: Bottom boundary of the cell.
        y_max: Top boundary of the cell.
        index: Position of the cell in the grid (0-based).
        label: Display label for the cell (e.g., 'A', 'B', '1'). Defaults to an
            empty string, which marks the cell as unlabeled: either a purely
            spatial grid where cells are identified by index rather than name,
            or a cell that is not of interest for the analysis (e.g., a spacer
            or ignored region that no stroke should be attributed to).
        strokes: List of Stroke objects assigned to this cell.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    index: int = 0
    label: str = ""
    strokes: List["Stroke"] = dataclasses.field(default_factory=list)

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
class Grid:
    """Represents a rectangular grid composed of multiple GridCell objects.

    The grid divides a bounding box into rows and columns, where each cell can
    hold strokes assigned by centroid location. Cells are ordered left-to-right,
    top-to-bottom (row-major order).

    Attributes:
        cells: List of GridCell objects that compose the grid.
    """

    cells: List[GridCell] = dataclasses.field(default_factory=list)

    @classmethod
    def from_bbox(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n_rows: int,
        n_cols: int,
        labels: Optional[List[str]] = None,
        padding: float = 0.1,
    ) -> "Grid":
        """Create a Grid by subdividing a bounding box into rows and columns.

        Cells are generated in row-major order (left-to-right, top-to-bottom).
        Interior cell boundaries are exact subdivisions of the bounding box;
        only the outermost edges are extended by ``padding`` so that centroids
        falling exactly on the outer boundary are still captured by a cell.

        Args:
            x_min: Left boundary of the bounding box.
            x_max: Right boundary of the bounding box.
            y_min: Bottom boundary of the bounding box.
            y_max: Top boundary of the bounding box.
            n_rows: Number of rows in the grid.
            n_cols: Number of columns in the grid.
            labels: Optional list of labels for each cell, assigned in
                row-major order. Must have length n_rows * n_cols if provided.
                If omitted, every cell is left with an empty-string label,
                marking the grid as unlabeled (cells identified by index only).
                An empty label also denotes a cell that is not of interest for
                the analysis (e.g., a spacer or ignored region).
            padding: Amount to extend the outermost cell edges to capture edge
                centroids; interior boundaries are unaffected (default 0.1).

        Returns:
            A Grid instance populated with GridCell objects.

        Raises:
            ValueError: If n_rows or n_cols is less than 1, or if the length
                of labels does not match n_rows * n_cols.
        """
        if n_rows < 1 or n_cols < 1:
            raise ValueError("n_rows and n_cols must be at least 1.")
        n_cells = n_rows * n_cols
        if labels is not None and len(labels) != n_cells:
            raise ValueError(
                f"labels length ({len(labels)}) must match n_rows * n_cols ({n_cells})."
            )

        col_width = (x_max - x_min) / n_cols
        row_height = (y_max - y_min) / n_rows

        cells: List[GridCell] = []
        index = 0
        for row in range(n_rows):
            for col in range(n_cols):
                cell_x_min = x_min + col * col_width
                if col == 0:
                    cell_x_min -= padding

                cell_x_max = x_min + (col + 1) * col_width
                if col == n_cols - 1:
                    cell_x_max += padding

                cell_y_min = y_max - (row + 1) * row_height
                if row == n_rows - 1:
                    cell_y_min -= padding

                cell_y_max = y_max - row * row_height
                if row == 0:
                    cell_y_max += padding

                label = labels[index] if labels is not None else ""
                cells.append(
                    GridCell(
                        x_min=cell_x_min,
                        x_max=cell_x_max,
                        y_min=cell_y_min,
                        y_max=cell_y_max,
                        index=index,
                        label=label,
                    )
                )
                index += 1

        return cls(cells=cells)

    def get_cell_for_point(self, x: float, y: float) -> int:
        """Return the index of the cell containing the given point.

        Iterates through cells and returns the index of the first cell whose
        half-open interval [min, max) contains the point. Returns -1 if no
        cell contains the point.

        Args:
            x: X coordinate of the point.
            y: Y coordinate of the point.

        Returns:
            The index of the matching cell, or -1 if no cell contains the point.
        """
        for cell in self.cells:
            if cell.x_min <= x < cell.x_max and cell.y_min <= y < cell.y_max:
                return cell.index
        return -1


@dataclasses.dataclass
class Stroke:
    """Represents a single stroke in an Alphabet or DSYM task.

    This class holds stroke data and computed features. Features are populated by
    utility functions after initialization.

    Attributes:
        points: DataFrame with columns including 'x', 'y', and 'seconds'.
        line_number: The line number identifying this stroke in the raw data.
        duration: Total time (s) spent drawing the stroke.
        distance: Total distance (px) of the stroke path.
        mean_speed: Mean drawing speed (px/s).
        speed_variance: Variance of drawing speed.
        smoothness: Smoothness of the stroke based on curvature changes.
        hesitation_count: Number of hesitations during the stroke.
        hesitation_duration: Total duration of hesitations (s).
        velocities: List of velocities at each point in the stroke (px/s).
        accelerations: List of accelerations at each point in the stroke (px/s²).
    """

    points: pd.DataFrame
    line_number: int

    duration: float = 0.0
    distance: float = 0.0
    mean_speed: float = 0.0
    speed_variance: float = 0.0
    smoothness: float = 0.0
    hesitation_count: int = 0
    hesitation_duration: float = 0.0
    velocities: List[float] = dataclasses.field(default_factory=list)
    accelerations: List[float] = dataclasses.field(default_factory=list)


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

        hesitation_count defaults to 0 and hesitation_duration defaults to 0.0 in the
        LineSegment object if there are less than 3 velocity points. This function also
        assumes uniform sampling.

        Args:
            threshold_percentile: Percentile to determine the velocity threshold for
                hesitations (default is 20, meaning the bottom 20% of velocities are
                considered hesitations).
        """
        if len(self.velocities) < 3:
            return

        dt = np.diff(self.ink_points["seconds"].values)

        threshold_velocity = np.percentile(self.velocities, threshold_percentile)
        hesitations = self.velocities < threshold_velocity

        hesitation_changes = np.diff(hesitations.astype(int))
        hesitation_count = np.sum(hesitation_changes == 1)

        if hesitations[0]:
            hesitation_count += 1

        self.hesitation_count = hesitation_count
        self.hesitation_duration = np.sum(hesitations) * dt[0]

        return

    def calculate_smoothness(self) -> None:
        """Calculate path smoothness based on Root Mean Square (RMS) curvature.

        Represents the curvature per unit arc length.
        Lower values indicate smoother drawings. Penalizes sharp corners (e.g.,
        90° turns) and noisy corrections. Normalized by arc length to reduce
        sampling-rate dependence.
        """
        if len(self.ink_points) < 3:
            return

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
        from graphomotor.core import config

        logger = config.get_logger()
        trail_circles = circles[trail_id]
        points = self.points.copy()

        if len(points) < 2:
            logger.warning(
                "Not enough points to calculate metrics for line segment: "
                "start=%s end=%s",
                self.start_label,
                self.end_label,
            )
            return

        if self.start_label not in trail_circles or self.end_label not in trail_circles:
            logger.warning(
                "Missing start/end labels: start=%s end=%s available=%s",
                self.start_label,
                self.end_label,
                list(trail_circles.keys()),
            )
            return

        start_circle = trail_circles[self.start_label]
        end_circle = trail_circles[self.end_label]

        ink_start_idx, ink_end_idx = self.valid_ink_trajectory(start_circle, end_circle)

        if ink_start_idx is None:
            logger.warning(
                "No valid ink trajectory found for line segment: start=%s end=%s",
                self.start_label,
                self.end_label,
            )
            return
        if ink_end_idx is None:
            self.ink_points = points.iloc[ink_start_idx:].copy()
            if len(self.ink_points) < 2:
                logger.warning(
                    "Not enough ink points to calculate metrics for line segment: "
                    "start=%s end=%s",
                    self.start_label,
                    self.end_label,
                )
                return
            self.ink_time = (
                self.ink_points.iloc[-1]["seconds"] - self.ink_points.iloc[0]["seconds"]
            )
            return
        if ink_end_idx <= ink_start_idx:
            logger.warning(
                "Invalid ink trajectory: end index (%d) is not greater than "
                "start index (%d) for line segment: start=%s end=%s",
                ink_end_idx,
                ink_start_idx,
                self.start_label,
                self.end_label,
            )
            return
        self.ink_points = self.points.iloc[ink_start_idx : ink_end_idx + 1].copy()

        if len(self.ink_points) < 2:
            logger.warning(
                "Not enough ink points to calculate metrics for line segment: "
                "start=%s end=%s",
                self.start_label,
                self.end_label,
            )
            return

        self.ink_time = (
            self.ink_points.iloc[-1]["seconds"] - self.ink_points.iloc[0]["seconds"]
        )
        self.calculate_velocity_metrics()
        self.calculate_path_optimality(start_circle, end_circle)
        self.calculate_smoothness()
        self.detect_hesitations()

        return
