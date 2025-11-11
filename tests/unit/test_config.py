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
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that SpiralConfig.add_custom_params correctly sets parameter values."""
    spiral_config = config.SpiralConfig.add_custom_params(custom_params)

    for key, value in expected_params.items():
        assert getattr(spiral_config, key) == value
        assert len(caplog.records) == 0


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
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that SpiralConfig.add_custom_params issues warnings appropriately."""
    spiral_config = config.SpiralConfig.add_custom_params(custom_params)

    assert len(caplog.records) == len(expected_warnings)
    for key, value in expected_params.items():
        assert getattr(spiral_config, key) == value
    for i, param in enumerate(expected_warnings):
        assert f"Unknown configuration parameters will be ignored: {param}" in str(
            caplog.records[i].message
        )


def test_get_logger(caplog: pytest.LogCaptureFixture) -> None:
    """Test the graphomotor logger with level set to WARNING (30)."""
    if logging.getLogger("graphomotor").handlers:
        logging.getLogger("graphomotor").handlers.clear()
    logger = config.get_logger()

    logger.debug("Debug message here.")
    logger.info("Info message here.")
    logger.warning("Warning message here.")

    assert logger.getEffectiveLevel() == logging.WARNING
    assert "Debug message here" not in caplog.text
    assert "Info message here." not in caplog.text
    assert "Warning message here." in caplog.text


def test_get_logger_second_call() -> None:
    """Test get logger when a handler already exists."""
    logger = config.get_logger()
    second_logger = config.get_logger()

    assert len(logger.handlers) == len(second_logger.handlers) == 1
    assert logger.handlers[0] is second_logger.handlers[0]
    assert logger is second_logger


@pytest.fixture
def circle() -> config.CircleTarget:
    """Create a standard circle at origin with radius 10."""
    return config.CircleTarget(
        order=1, label="test_circle", center_x=0.0, center_y=0.0, radius=10.0
    )


def test_point_at_center(circle: config.CircleTarget) -> None:
    """Point at the exact center should be inside the circle."""
    assert circle.contains_point(0.0, 0.0)


def test_point_on_edge_exact(circle: config.CircleTarget) -> None:
    """Point exactly on the edge should be inside (with default tolerance)."""
    assert circle.contains_point(10.0, 0.0)
    assert circle.contains_point(0.0, 10.0)
    assert circle.contains_point(-10.0, 0.0)
    assert circle.contains_point(0.0, -10.0)


def test_point_just_inside(circle: config.CircleTarget) -> None:
    """Point just inside the radius should be contained."""
    assert circle.contains_point(5.0, 0.0)
    assert circle.contains_point(0.0, 5.0)


def test_point_outside_with_default_tolerance(circle: config.CircleTarget) -> None:
    """Point outside default tolerance boundary should not be contained."""
    assert not circle.contains_point(16.0, 0.0)
    assert not circle.contains_point(0.0, 16.0)
