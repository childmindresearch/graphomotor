"""Test cases for generate_reference_spiral.py functions."""

import pathlib
import stat
import tempfile

import numpy as np

from graphomotor.core import config
from graphomotor.utils import generate_reference_spiral


def test_compute_reference_spiral() -> None:
    """Test the generation of a reference spiral."""
    spiral_config = config.SpiralConfig()
    expected_mean_arc_length = generate_reference_spiral._calculate_arc_length(
        spiral_config.end_angle, spiral_config
    ) / (spiral_config.num_points - 1)

    spiral = generate_reference_spiral._compute_reference_spiral(spiral_config)
    arc_lengths = np.linalg.norm(spiral[1:] - spiral[:-1], axis=1)
    mean_arc_length = np.mean(arc_lengths)

    assert isinstance(spiral, np.ndarray)
    assert spiral.shape == (spiral_config.num_points, 2)
    assert np.array_equal(
        spiral[0],
        [spiral_config.center_x, spiral_config.center_y],
    )
    assert np.allclose(
        spiral[-1],
        [
            spiral_config.center_x
            + spiral_config.growth_rate * spiral_config.end_angle,
            spiral_config.center_y,
        ],
        atol=0,
        rtol=1e-8,
    )
    assert np.allclose(arc_lengths, mean_arc_length, atol=0, rtol=1e-3)
    assert np.isclose(mean_arc_length, expected_mean_arc_length, atol=0, rtol=1e-6)


def test_get_spiral_cache_key() -> None:
    """Test cache key generation for spiral configurations."""
    config1 = config.SpiralConfig()
    config2 = config.SpiralConfig(center_x=100.0)
    config3 = config.SpiralConfig(num_points=200)

    key1 = generate_reference_spiral._get_spiral_cache_key(config1)
    key2 = generate_reference_spiral._get_spiral_cache_key(config1)
    key3 = generate_reference_spiral._get_spiral_cache_key(config2)
    key4 = generate_reference_spiral._get_spiral_cache_key(config3)

    assert len(key1) == 32
    assert all(c in "0123456789abcdef" for c in key1)

    assert key1 == key2
    assert key1 != key3
    assert key1 != key4
    assert key3 != key4


def test_get_cache_path() -> None:
    """Test cache path generation, structure, and consistency."""
    config1 = config.SpiralConfig()
    config2 = config.SpiralConfig(center_x=100.0)

    cache_path1 = generate_reference_spiral._get_cache_path(config1)
    cache_path2 = generate_reference_spiral._get_cache_path(config2)
    cache_path3 = generate_reference_spiral._get_cache_path(config1)

    assert isinstance(cache_path1, pathlib.Path)
    assert cache_path1.name.startswith("reference_spiral_")
    assert cache_path1.name.endswith(".npy")

    assert cache_path1 != cache_path2
    assert cache_path1 == cache_path3


def test_get_cache_path_with_write_privileges() -> None:
    """Test directory creation and write access when package dir is writable."""
    spiral_config = config.SpiralConfig()
    cache_path = generate_reference_spiral._get_cache_path(spiral_config)
    cache_dir = cache_path.parent
    test_file = cache_dir / "test_write_check.tmp"

    assert cache_dir.exists(), "Cache directory should be created"
    assert cache_dir.is_dir(), "Cache path parent should be a directory"

    try:
        test_file.write_text("test content")
        assert test_file.exists(), "Should be able to write to cache directory"
        test_file.unlink()
    except PermissionError:
        pass


def test_get_cache_path_readonly_fallback_behavior() -> None:
    """Test fallback to temp directory when package dir is read-only."""
    spiral_config = config.SpiralConfig()

    # Get the normal cache path to identify the package directory
    normal_cache_path = generate_reference_spiral._get_cache_path(spiral_config)
    package_data_dir = normal_cache_path.parent

    try:
        # Make the package data directory read-only temporarily
        original_permissions = package_data_dir.stat().st_mode
        package_data_dir.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        try:
            # Get cache path again - should fallback to temp directory
            readonly_cache_path = generate_reference_spiral._get_cache_path(
                spiral_config
            )
            readonly_cache_dir = readonly_cache_path.parent

            # Verify fallback to temp directory
            temp_root = pathlib.Path(tempfile.gettempdir())
            assert str(readonly_cache_dir).startswith(str(temp_root)), (
                "Should fallback to temp directory when package dir is read-only"
            )

            # Verify directory can be created and is writable
            readonly_cache_dir.mkdir(parents=True, exist_ok=True)
            assert readonly_cache_dir.exists() and readonly_cache_dir.is_dir()

            # Test write access with a single file operation
            test_file = readonly_cache_dir / "test_fallback_write.tmp"
            test_file.write_text("fallback test")
            test_file.unlink()

        finally:
            # Restore original permissions
            package_data_dir.chmod(original_permissions)

    except (PermissionError, OSError):
        # Fallback test for restricted environments
        cache_path = generate_reference_spiral._get_cache_path(spiral_config)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        assert cache_path.parent.exists() and cache_path.parent.is_dir()


def test_load_reference_spiral_no_cache() -> None:
    """Test loading a reference spiral when no cache exists."""
    spiral_config = config.SpiralConfig(center_x=999.0, center_y=999.0, num_points=100)

    result = generate_reference_spiral._load_reference_spiral(spiral_config)

    assert result is None


def test_load_reference_spiral_with_cache() -> None:
    """Test loading a reference spiral from an existing cache file."""
    spiral_config = config.SpiralConfig(center_x=888.0, center_y=888.0, num_points=50)

    test_spiral = np.array([[0, 0], [1, 1], [2, 2]])
    cache_path = generate_reference_spiral._get_cache_path(spiral_config)
    np.save(cache_path, test_spiral)

    try:
        loaded_spiral = generate_reference_spiral._load_reference_spiral(spiral_config)
        assert loaded_spiral is not None
        assert isinstance(loaded_spiral, np.ndarray)
        assert np.array_equal(loaded_spiral, test_spiral)
    finally:
        if cache_path.exists():
            cache_path.unlink()


def test_generate_reference_spiral() -> None:
    """Test the main generate_reference_spiral function."""
    spiral_config = config.SpiralConfig(center_x=777.0, center_y=777.0, num_points=200)

    cache_path = generate_reference_spiral._get_cache_path(spiral_config)

    if cache_path.exists():
        cache_path.unlink()

    try:
        spiral = generate_reference_spiral.generate_reference_spiral(spiral_config)
        spiral2 = generate_reference_spiral.generate_reference_spiral(spiral_config)
        expected_center = [spiral_config.center_x, spiral_config.center_y]

        assert isinstance(spiral, np.ndarray)
        assert spiral.shape == (spiral_config.num_points, 2)
        assert np.array_equal(spiral[0], expected_center)
        assert cache_path.exists(), "Cache file should be created after generation"
        assert np.array_equal(spiral, spiral2)

    finally:
        if cache_path.exists():
            cache_path.unlink()
