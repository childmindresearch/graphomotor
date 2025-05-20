"""Test cases for config.py functions."""

import logging

import numpy as np
import pytest

from graphomotor.core import config


@pytest.mark.parametrize(
    "custom_params, expected_params",
    [
        (
            {
                "center_x": 25,
                "center_y": 25,
                "start_angle": np.pi,
                "end_angle": 2 * np.pi,
                "num_points": 100,
            },
            {
                "center_x": 25,
                "center_y": 25,
                "start_radius": 0,
                "growth_rate": 1.075,
                "start_angle": np.pi,
                "end_angle": 2 * np.pi,
                "num_points": 100,
            },
        ),
    ],
)
def test_spiral_config_add_custom_params_valid(
    custom_params: dict[str, int | float],
    expected_params: dict[str, int | float],
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Test that SpiralConfig.add_custom_params correctly sets parameter values."""
    spiral_config = config.SpiralConfig.add_custom_params(custom_params)

    for key, value in expected_params.items():
        assert getattr(spiral_config, key) == value
        assert len(recwarn) == 0


@pytest.mark.parametrize(
    "custom_params, expected_params, expected_warnings",
    [
        (
            {
                "growth_rate": 1,
                "start_radius": 100,
                "end_radius": 20,
                "meaning_of_life": 42,
            },
            {
                "center_x": 50,
                "center_y": 50,
                "start_radius": 100,
                "growth_rate": 1,
                "start_angle": 0,
                "end_angle": 8 * np.pi,
                "num_points": 10000,
            },
            ["end_radius", "meaning_of_life"],
        ),
    ],
)
def test_spiral_config_add_custom_params_warnings(
    custom_params: dict[str, int | float],
    expected_params: dict[str, int | float],
    expected_warnings: list[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Test that SpiralConfig.add_custom_params issues warnings appropriately."""
    spiral_config = config.SpiralConfig.add_custom_params(custom_params)

    assert len(recwarn) == len(expected_warnings)
    for key, value in expected_params.items():
        assert getattr(spiral_config, key) == value
    for i, param in enumerate(expected_warnings):
        assert f"Unknown configuration parameters will be ignored: {param}" in str(
            recwarn[i].message
        )


def test_get_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Test the graphomotor logger with level set to INFO (20)."""
    if logging.getLogger("graphomotor").handlers:
        logging.getLogger("graphomotor").handlers.clear()
    logger = config.get_logger()

    logger.debug("Debug message here.")
    logger.info("Info message here.")
    logger.warning("Warning message here.")

    assert logger.getEffectiveLevel() == logging.INFO
    assert "Debug message here" not in caplog.text
    assert "Info message here." in caplog.text
    assert "Warning message here." in caplog.text


def test_get_logger_second_call() -> None:
    """Test get logger when a handler already exists."""
    logger = config.get_logger()
    second_logger = config.get_logger()

    assert len(logger.handlers) == len(second_logger.handlers) == 1
    assert logger.handlers[0] is second_logger.handlers[0]
    assert logger is second_logger
