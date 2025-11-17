"""Configuration module for graphomotor."""

import dataclasses
import json
import logging
import os
import typing
from importlib import metadata
from typing import Dict

import numpy as np

from graphomotor.core import models


def get_version() -> str:
    """Return graphomotor version."""
    try:
        return metadata.version("graphomotor")
    except metadata.PackageNotFoundError:
        return "Version unknown"


def get_logger() -> logging.Logger:
    """Get the Graphomotor logger."""
    logger = logging.getLogger("graphomotor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)s - %(funcName)s - %(message)s",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()


def set_verbosity_level(verbosity_count: int) -> None:
    """Set the logging level based on verbosity count.

    Args:
        verbosity_count: Number of times -v was specified.
            - 0: WARNING level (quiet - only warnings and errors)
            - 1: INFO level (normal - includes info messages)
            - 2: DEBUG level (verbose - includes debug messages)
    """
    if verbosity_count == 0:
        logger.setLevel(logging.WARNING)
    elif verbosity_count == 1:
        logger.setLevel(logging.INFO)
    elif verbosity_count == 2:
        logger.setLevel(logging.DEBUG)
    else:
        logger.warning(
            f"Invalid verbosity level {verbosity_count}. Defaulting to WARNING level."
        )
        logger.setLevel(logging.WARNING)


@dataclasses.dataclass(frozen=True)
class SpiralConfig:
    """Class for the parameters of anticipated spiral drawing."""

    center_x: float = 50
    center_y: float = 50
    start_radius: float = 0
    growth_rate: float = 1.075
    start_angle: float = 0
    end_angle: float = 8 * np.pi
    num_points: int = 10000

    @classmethod
    def add_custom_params(cls, config_dict: dict[str, float | int]) -> "SpiralConfig":
        """Update the SpiralConfig instance with custom parameters.

        Args:
            config_dict: Dictionary with configuration parameters.

        Returns:
            SpiralConfig instance with updated parameters.
        """
        valid_params = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_params: dict[str, typing.Any] = {}

        for key, value in config_dict.items():
            if key in valid_params:
                filtered_params[key] = value
            else:
                valid_param_names = ", ".join(valid_params)
                logger.warning(
                    f"Unknown configuration parameters will be ignored: {key}. "
                    f"Valid parameters are: {valid_param_names}"
                )

        return cls(**filtered_params)


def load_scaled_circles(filepath: str) -> Dict[str, Dict[str, models.CircleTarget]]:
    """Load circle configurations from trails_points_scaled.json.

    This function reads a JSON file containing circle target definitions for various
    trail types and constructs CircleTarget instances for each defined circle.
    This will be configured only once per run.

    Args:
        filepath: Path to the JSON file containing circle configurations.

    Returns:
        A dictionary mapping each trail type to dictionaries of CircleTarget instances.

    Raises:
        FileNotFoundError: If the specified filepath does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If required fields (x, y, label, radius) are missing from circle data.
        TypeError: If trails_data is not a dictionary or trail_points is not a list.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The path '{filepath}' does not exist. "
            "Confirm that all necessary files are in place."
        )

    with open(filepath, "r") as f:
        trails_data = json.load(f)

        if not isinstance(trails_data, dict):
            raise TypeError(
                "Expected trails_data to be a dictionary, "
                f"got {type(trails_data).__name__}"
            )

        circles = {}
        for trail_screen, trail_points in trails_data.items():
            if not isinstance(trail_points, list):
                raise TypeError(
                    f"Expected trail_points for '{trail_screen}' to be a list, "
                    f"got {type(trail_points).__name__}"
                )

            trail_circles = {}
            for idx, point in enumerate(trail_points):
                if not isinstance(point, dict):
                    raise TypeError(
                        f"Expected point at index {idx} in '{trail_screen}' to be a "
                        f"dictionary, got {type(point).__name__}"
                    )

                # Validate required fields
                required_fields = ["x", "y", "label", "radius"]
                missing_fields = [
                    field for field in required_fields if field not in point
                ]
                if missing_fields:
                    raise KeyError(
                        f"Missing required field(s) {missing_fields} at index {idx} "
                        f"of trail '{trail_screen}'"
                    )

                order = idx + 1
                circle = models.CircleTarget(
                    order=order,
                    center_x=point["x"],
                    center_y=point["y"],
                    label=str(point["label"]),
                    radius=point["radius"],
                )
                trail_circles[circle.label] = circle

            circles[trail_screen] = trail_circles
    return circles


def create_config_from_circles(
    circles: Dict[str, Dict[str, models.CircleTarget]],
) -> Dict:
    """Create a config dict from circles for compatibility.

    Args:
        circles: Dictionary of circle targets per trail.

    Returns:
        Configuration dictionary.
    """
    config = {}
    for trail_id, trail_circles in circles.items():
        config[trail_id] = {
            "items": [
                {
                    "order": circle.order,
                    "center_x": circle.center_x,
                    "center_y": circle.center_y,
                    "label": circle.label,
                }
                for circle in trail_circles.values()
            ]
        }
    return config
    if not isinstance(trails_data, dict):
        raise TypeError(
            f"Expected trails_data to be a dictionary, got {type(trails_data).__name__}"
        )

    circles = {}
    required_fields = ["x", "y", "label", "radius"]
    for trail_task_number, trail_points in trails_data.items():
        if not isinstance(trail_points, list):
            raise TypeError(
                f"Expected trail_points for '{trail_task_number}' to be a list, "
                f"got {type(trail_points).__name__}"
            )

        trail_circles = {}
        for idx, point_dict in enumerate(trail_points):
            if not isinstance(point_dict, dict):
                raise TypeError(
                    f"Expected point at index {idx} in '{trail_task_number}' to be a "
                    f"dictionary, got {type(point_dict).__name__}"
                )

            missing_fields = [
                field for field in required_fields if field not in point_dict
            ]
            if missing_fields:
                raise KeyError(
                    f"Missing required field(s) {missing_fields} at index {idx} "
                    f"of trail '{trail_task_number}'"
                )

            order = idx + 1
            circle = models.CircleTarget(
                order=order,
                center_x=point_dict["x"],
                center_y=point_dict["y"],
                label=str(point_dict["label"]),
                radius=point_dict["radius"],
            )
            trail_circles[circle.label] = circle

        circles[trail_task_number] = trail_circles
    return circles
