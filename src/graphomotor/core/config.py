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


def load_scaled_circles() -> Dict[str, Dict[str, models.CircleTarget]]:
    """Load circle configurations from trails_points_scaled.json.

    This function reads a JSON file containing circle target definitions for various
    trail types and constructs CircleTarget instances for each defined circle.
    This will be configured only once per run.

    Returns:
        A dictionary mapping each trail type to dictionaries of CircleTarget instances.
    """
    if not os.path.exists("src/graphomotor/utils/trails_points_scaled.json"):
        raise FileNotFoundError(
            "The path 'src/graphomotor/utils/trails_points_scaled.json' does not exist."
            "Confirm that all necessary files are in place."
        )

    with open("src/graphomotor/utils/trails_points_scaled.json", "r") as f:
        trails_data = json.load(f)

        circles = {}
        for trail_screen, trail_points in trails_data.items():
            trail_circles = {}
            for idx, point in enumerate(trail_points):
                order = idx + 1
                circle = models.CircleTarget(
                    order=order,
                    cx=point["x"],
                    cy=point["y"],
                    label=str(point["label"]),
                    radius=point["radius"],
                )
                trail_circles[circle.label] = circle

            circles[trail_screen] = trail_circles
    return circles
