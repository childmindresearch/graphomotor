"""Feature extraction module for drawing error-based metrics in spiral drawing data."""

import numpy as np
from shapely import geometry, ops

from graphomotor.core import models


def calculate_area_under_curve(
    drawn_spiral: models.Spiral, reference_spiral: np.ndarray
) -> dict[str, float]:
    """Calculate the area between drawn and reference spirals.

    This function measures the deviation between drawn and reference spirals by
    computing the enclosed area between them using the shapely library. Lower values
    indicate better adherence to the template. The algorithm works by creating polygons
    that connect spiral endpoints, finding intersections between lines, and calculating
    the total area of the resulting polygons.

    Args:
        drawn_spiral: The spiral drawn by the subject.
        reference_spiral: The reference spiral.

    Returns:
        Dictionary containing the area under curve metric
    """
    spiral = drawn_spiral.data[["x", "y"]].values
    line_drawn = geometry.LineString(spiral)
    line_reference = geometry.LineString(reference_spiral)
    first_segment = geometry.LineString([spiral[0], reference_spiral[0]])
    last_segment = geometry.LineString([spiral[-1], reference_spiral[-1]])
    merged_line = ops.unary_union(
        [line_drawn, line_reference, first_segment, last_segment]
    )
    polygons = list(ops.polygonize(merged_line))
    return {"area_under_curve": sum(p.area for p in polygons)}
