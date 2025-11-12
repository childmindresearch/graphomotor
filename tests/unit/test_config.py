"""Test cases for config.py functions."""

import json
import logging
import pathlib
import shutil
import tempfile
import typing

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
def temp_dir() -> typing.Generator[str, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def json_file_path(temp_dir: str) -> str:
    """Create a JSON file path in the temporary directory."""
    json_path = temp_dir + "/test_trails_points_scaled.json"
    return json_path


def test_successful_load_with_valid_data(json_file_path: str) -> None:
    """Test that valid JSON data is loaded and parsed with expected behaviors."""
    json_data = {
        "trail_1": [
            {"x": 100, "y": 200, "label": "1", "radius": 25},
            {"x": 300, "y": 400, "label": "2", "radius": 25},
            {"x": 100.5, "y": 200.7, "label": "float", "radius": 25.3},
        ],
        "trail_2": [{"x": 150, "y": 250, "label": "A", "radius": 30}],
    }

    with open(json_file_path, "w") as f:
        json.dump(json_data, f)

    result = config.load_scaled_circles(json_file_path)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert "trail_1" in result
    assert "trail_2" in result

    assert len(result["trail_1"]) == 3
    assert len(result["trail_2"]) == 1

    circle = result["trail_1"]["1"]
    assert circle.order == 1
    assert circle.center_x == 100
    assert circle.center_y == 200
    assert circle.label == "1"
    assert circle.radius == 25


def test_file_not_found_raises_error(
    temp_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that FileNotFoundError is raised when JSON file doesn't exist."""
    monkeypatch.chdir(temp_dir)

    with pytest.raises(
        FileNotFoundError,
        match=r"The path 'test_trails_points_scaled\.json' does not exist\.",
    ):
        config.load_scaled_circles("test_trails_points_scaled.json")


def test_invalid_json_raises_error(json_file_path: str) -> None:
    """Test that malformed JSON raises JSONDecodeError."""
    with open(json_file_path, "w") as f:
        f.write("{ this is not valid json }")

    with pytest.raises(json.JSONDecodeError, match="Expecting"):
        config.load_scaled_circles(json_file_path)


def test_empty_json_returns_empty_dict(json_file_path: str) -> None:
    """Test that empty JSON object returns empty dictionary."""
    with open(json_file_path, "w") as f:
        json.dump({}, f)

    result = config.load_scaled_circles(json_file_path)

    assert result == {}
    assert isinstance(result, dict)


def test_missing_required_field_raises_key_error(json_file_path: str) -> None:
    """Test that missing required fields raise KeyError."""
    json_data = {"trail_a": [{"x": 100, "y": 200, "radius": 25}]}

    with open(json_file_path, "w") as f:
        json.dump(json_data, f)

    with pytest.raises(
        KeyError,
        match=r"Missing required field\(s\) \['label'\] at index 0 of trail 'trail_a'",
    ):
        config.load_scaled_circles(json_file_path)


def test_non_dict_trails_data_raises_type_error(json_file_path: str) -> None:
    """Test that non-dictionary trails_data raises TypeError."""
    with open(json_file_path, "w") as f:
        json.dump([], f)  # Array instead of dict

    with pytest.raises(
        TypeError, match=r"Expected trails_data to be a dictionary, got list"
    ):
        config.load_scaled_circles(json_file_path)


def test_non_list_trail_points_raises_type_error(json_file_path: str) -> None:
    """Test that non-list trail_points raises TypeError."""
    json_data = {"trail_a": "not a list"}

    with open(json_file_path, "w") as f:
        json.dump(json_data, f)

    with pytest.raises(
        TypeError, match=r"Expected trail_points for 'trail_a' to be a list, got str"
    ):
        config.load_scaled_circles(json_file_path)


def test_non_dict_point_raises_type_error(json_file_path: str) -> None:
    """Test that non-dictionary point raises TypeError."""
    json_data = {"trail_a": ["not a dict"]}

    with open(json_file_path, "w") as f:
        json.dump(json_data, f)

    with pytest.raises(
        TypeError,
        match=r"Expected point at index 0 in 'trail_a' to be a dictionary, got str",
    ):
        config.load_scaled_circles(json_file_path)


def test_multiple_missing_fields_raises_key_error(json_file_path: str) -> None:
    """Test that multiple missing fields are reported in error."""
    json_data = {"trail_a": [{"x": 100}]}  # Missing y, label, radius

    with open(json_file_path, "w") as f:
        json.dump(json_data, f)

    with pytest.raises(
        KeyError, match=r"Missing required field\(s\) \['y', 'label', 'radius'\]"
    ):
        config.load_scaled_circles(json_file_path)
