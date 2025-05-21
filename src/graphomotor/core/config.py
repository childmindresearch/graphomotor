"""Configuration module for Graphomotor."""

import dataclasses
import logging
import warnings

import numpy as np


@dataclasses.dataclass
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
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                valid_params = ", ".join(
                    f.name for f in cls.__dataclass_fields__.values()
                )
                warnings.warn(
                    f"Unknown configuration parameters will be ignored: {key}. "
                    f"Valid parameters are: {valid_params}"
                )
        return config


def get_logger() -> logging.Logger:
    """Get the Graphomotor logger."""
    logger = logging.getLogger("graphomotor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)s - %(funcName)s - %(message)s",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
