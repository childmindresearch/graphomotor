"""Configuration module for Graphomotor."""

import logging

import numpy as np


class _SpiralConfig:
    """Configuration for the reference spiral generation."""

    SPIRAL_CENTER_X = 50
    SPIRAL_CENTER_Y = 50
    SPIRAL_START_RADIUS = 0
    SPIRAL_GROWTH_RATE = 1.075
    SPIRAL_START_ANGLE = 0
    SPIRAL_END_ANGLE = 8 * np.pi
    SPIRAL_NUM_POINTS = 10000


def get_logger() -> logging.Logger:
    """Get the Graphomotor logger."""
    logger = logging.getLogger("graphomotor")
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)s - %(funcName)s - %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
