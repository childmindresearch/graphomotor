"""Tests for Trails feature inclusion, dispatch, provenance and CSV safety."""

import logging
import pathlib

import pandas as pd
import pytest

from graphomotor.core import config, trails_orchestrator

_SEGMENT_FEATURES = {
    "ink_time",
    "think_time",
    "distance",
    "mean_speed",
    "mean_acceleration",
    "speed_variance",
    "path_optimality",
    "smoothness",
    "hesitation_count",
    "hesitation_duration",
    "min_velocity",
    "max_velocity",
}
_RECORDING_METADATA = {"source_file", "start_time", "participant_id", "trail"}


def _input(path: pathlib.Path, trail: str = "trail2") -> pathlib.Path:
    """Create synthetic correct paths and an incomplete error path."""
    circles = config.load_scaled_circles(str(trails_orchestrator._CIRCLES_PATH))[trail]
    first, second, third, fourth = ("1", "2", "3", "4")
    if trail == "trail4":
        first, second, third, fourth = "1", "A", "2", "B"
    rows: list[dict] = []
    for line, (start, end, error, fractions) in enumerate(
        [
            (first, second, "E0", [0, 0.2, 0.4, 0.6, 0.8, 1]),
            (second, fourth, "E1", [0, 0.2, 0.4]),
            (second, third, "E0", [0, 0.2, 0.4, 0.6, 0.8, 1]),
        ]
    ):
        start_circle, end_circle = circles[start], circles[end]
        for fraction in fractions:
            seconds = len(rows)
            rows.append(
                {
                    "line_number": line,
                    "x": start_circle.center_x
                    + fraction * (end_circle.center_x - start_circle.center_x),
                    "y": start_circle.center_y
                    + fraction * (end_circle.center_y - start_circle.center_y),
                    "UTC_Timestamp": seconds,
                    "seconds": seconds,
                    "epoch_time_in_seconds_start": 0,
                    "error": error,
                    "actual_path": f"{start} ~ {end}",
                    "correct_path": (
                        f"{second} ~ {third}" if error != "E0" else f"{start} ~ {end}"
                    ),
                    "total_time": 42,
                    "total_number_of_errors": 1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("trail", ["trail2", "trail4"])
@pytest.mark.parametrize("string_path", [False, True])
def test_run_pipeline_file_includes_features(
    tmp_path: pathlib.Path, trail: str, string_path: bool
) -> None:
    """Both supported tasks export feature tables and recording provenance."""
    source = _input(tmp_path / f"[5000000]example-{trail}.csv", trail)
    result = trails_orchestrator.run_pipeline(str(source) if string_path else source)

    assert isinstance(result, trails_orchestrator.TrailsResult)
    assert not result.failures
    assert len(result.segments) == 3
    assert len(result.participants) == 1
    assert _RECORDING_METADATA | _SEGMENT_FEATURES <= set(result.segments.columns)
    assert _RECORDING_METADATA | {
        "total_time",
        "duration",
        "total_errors",
        "percent_accurate_paths",
        "total_error_time",
        "correct_segment_count",
        "error_segment_count",
    } <= set(result.participants.columns)
    for group in ("correct", "error"):
        for metric in _SEGMENT_FEATURES - {"min_velocity", "max_velocity"}:
            assert f"mean_{group}_{metric}" in result.participants
            assert f"count_{group}_{metric}" in result.participants
        for metric in (
            "ink_time",
            "think_time",
            "distance",
            "hesitation_count",
            "hesitation_duration",
        ):
            assert f"total_{group}_{metric}" in result.participants
    correct = result.segments.loc[~result.segments["is_error"]]
    assert correct[list(_SEGMENT_FEATURES)].notna().all().all()
    assert result.segments.loc[result.segments["is_error"], "think_time"].isna().all()
    assert result.participants["source_file"].tolist() == [str(source.resolve())]
    assert set(result.segments["source_file"]) == {str(source.resolve())}
    assert result.participants["trail"].tolist() == [trail]

    for suffix, frame in (
        ("_output.csv", result.participants),
        ("_output_segments.csv", result.segments),
    ):
        saved = pd.read_csv(tmp_path / f"{source.stem}{suffix}")
        assert list(saved.columns) == list(frame.columns)
        assert len(saved) == len(frame)
        pd.testing.assert_frame_equal(saved.isna(), frame.isna())


@pytest.mark.parametrize("collision", ["participant", "segment", "directory"])
def test_output_collision_never_overwrites(
    tmp_path: pathlib.Path, collision: str
) -> None:
    """Preflight both output paths before creating any files."""
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    output = tmp_path / "custom.csv"
    companion = tmp_path / "custom_segments.csv"
    occupied = companion if collision == "segment" else output
    if collision == "directory":
        occupied.mkdir()
    else:
        occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        trails_orchestrator.run_pipeline(source, output)
    assert occupied.exists()
    if collision != "directory":
        assert occupied.read_text(encoding="utf-8") == "keep"
    assert not (output if collision == "segment" else companion).exists()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("alias", ["input", "companion", "symlink", "hardlink"])
def test_input_aliases_are_never_overwritten(
    tmp_path: pathlib.Path, overwrite: bool, alias: str
) -> None:
    """Overwrite never permits replacing the source through either output path."""
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    original = source.read_bytes()
    output = tmp_path / "custom.csv"
    if alias == "input":
        output = source
    elif alias == "companion":
        source = source.rename(tmp_path / "custom_segments.csv")
    elif alias == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    with pytest.raises(ValueError, match="aliases input"):
        trails_orchestrator.run_pipeline(source, output, overwrite=overwrite)
    assert source.read_bytes() == original


def test_directory_continues_and_reports_failures(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Batch outputs include only successful direct children and structured errors."""
    source = tmp_path / "input"
    source.mkdir()
    destination = tmp_path / "output"
    destination.mkdir()
    good = _input(source / "[5000000]good-trail2.csv")
    bad = source / "[5000001]bad-trail4.csv"
    bad.write_text("broken\n1\n", encoding="utf-8")
    (source / "unsupported.csv").write_text("broken", encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    _input(nested / "[5000002]nested-trail2.csv")
    with caplog.at_level(logging.ERROR):
        result = trails_orchestrator.run_pipeline(str(source), str(destination))
    assert len(result.segments) == 3
    assert result.participants["source_file"].tolist() == [str(good.resolve())]
    assert set(result.failures) == {bad.resolve()}
    assert result.failures[bad.resolve()]
    assert bad.name in caplog.text
    assert (destination / f"{good.stem}_output.csv").is_file()
    assert (destination / f"{good.stem}_output_segments.csv").is_file()
    saved = pd.read_csv(destination / "directory_summary_statistics.csv")
    assert list(saved.columns) == list(result.participants.columns)
    assert len(saved) == len(result.participants)


@pytest.mark.parametrize("contents", ["empty", "unsupported", "nested"])
def test_no_supported_inputs_raises(tmp_path: pathlib.Path, contents: str) -> None:
    """No matching recordings is an error, not an empty successful summary."""
    if contents == "unsupported":
        _input(tmp_path / "[5000000]example-trail1.csv")
    elif contents == "nested":
        nested = tmp_path / "nested"
        nested.mkdir()
        _input(nested / "[5000000]example-trail2.csv")
    with pytest.raises(ValueError, match="No supported Trails CSV"):
        trails_orchestrator.run_pipeline(tmp_path)
    assert not (tmp_path / "directory_summary_statistics.csv").exists()


def test_all_inputs_fail_without_writing_summary(tmp_path: pathlib.Path) -> None:
    """A completely failed batch raises with input context and can be retried."""
    bad = tmp_path / "[5000000]bad-trail2.csv"
    bad.write_text("broken\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not extract features from any file"):
        trails_orchestrator.run_pipeline(tmp_path)
    assert not (tmp_path / "directory_summary_statistics.csv").exists()
    _input(bad)
    assert not trails_orchestrator.run_pipeline(tmp_path).participants.empty


def test_summary_collision_prevents_per_file_writes(tmp_path: pathlib.Path) -> None:
    """An occupied batch summary fails before processing individual recordings."""
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    summary = tmp_path / "directory_summary_statistics.csv"
    summary.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        trails_orchestrator.run_pipeline(tmp_path)
    assert summary.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / f"{source.stem}_output.csv").exists()


def test_repeated_participants_retain_recording_identity(
    tmp_path: pathlib.Path,
) -> None:
    """Multiple sessions and trails remain distinct, attributable summary rows."""
    sources = [
        _input(tmp_path / "[5000000]session1-trail2.csv"),
        _input(tmp_path / "[5000000]session2-trail2.csv"),
        _input(tmp_path / "[5000000]session1-trail4.csv", "trail4"),
    ]
    result = trails_orchestrator.run_pipeline(tmp_path)
    assert len(result.participants) == len(sources)
    assert set(result.participants["participant_id"]) == {"5000000"}
    assert result.participants["source_file"].is_unique
    assert set(result.participants["source_file"]) == {
        str(p.resolve()) for p in sources
    }
    assert set(result.segments["source_file"]) == set(
        result.participants["source_file"]
    )


def test_repeated_paths_preserve_existing_grouping(tmp_path: pathlib.Path) -> None:
    """Repeated error paths remain grouped rather than redefining segmentation."""
    source = _input(tmp_path / "[5000000]repeat-trail2.csv")
    data = pd.read_csv(source)
    repeated = data.loc[data["line_number"] == 1].copy()
    repeated["line_number"] = 3
    repeated["seconds"] += len(data)
    repeated["UTC_Timestamp"] += len(data)
    pd.concat([data, repeated], ignore_index=True).to_csv(source, index=False)

    result = trails_orchestrator.run_pipeline(source)

    assert len(result.segments) == data["actual_path"].nunique()
    assert not result.segments.duplicated(["start_label", "end_label"]).any()
    assert set(result.segments["source_file"]) == {str(source.resolve())}


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("phase", ["serialization", "publication"])
@pytest.mark.parametrize("failed_index", [0, 1])
def test_pair_failure_rolls_back_and_allows_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    phase: str,
    failed_index: int,
) -> None:
    """Failure at either output preserves previous files or removes new files."""
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    output = tmp_path / "custom.csv"
    companion = tmp_path / "custom_segments.csv"
    targets = [output, companion]
    if overwrite:
        for path in targets:
            path.write_text("keep", encoding="utf-8")
    original_csv = pd.DataFrame.to_csv
    original_replace = pathlib.Path.replace
    calls = 0

    def failing_csv(frame: pd.DataFrame, path: pathlib.Path, *, index: bool) -> None:
        nonlocal calls
        current = calls
        calls += 1
        if current == failed_index:
            raise OSError("simulated serialization failure")
        original_csv(frame, path, index=index)

    def failing_replace(path: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if target == targets[failed_index] and path.suffix == ".csv":
            raise OSError("simulated publication failure")
        return original_replace(path, target)

    with monkeypatch.context() as context:
        if phase == "serialization":
            context.setattr(pd.DataFrame, "to_csv", failing_csv)
        else:
            context.setattr(pathlib.Path, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated"):
            trails_orchestrator.run_pipeline(source, output, overwrite=overwrite)
    for path in targets:
        if overwrite:
            assert path.read_text(encoding="utf-8") == "keep"
        else:
            assert not path.exists()
    assert not list(tmp_path.glob(".graphomotor-*"))
    result = trails_orchestrator.run_pipeline(source, output, overwrite=overwrite)
    assert not result.participants.empty
    assert all(path.is_file() for path in targets)


def test_failed_pair_is_excluded_and_later_files_continue(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed pair is cleaned up and reported without stopping later inputs."""
    bad = _input(tmp_path / "[5000000]bad-trail2.csv")
    good = _input(tmp_path / "[5000001]good-trail2.csv")
    original_replace = pathlib.Path.replace

    def failing_replace(path: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if target.name == f"{bad.stem}_output_segments.csv":
            raise OSError("simulated publication failure")
        return original_replace(path, target)

    with monkeypatch.context() as context:
        context.setattr(pathlib.Path, "replace", failing_replace)
        result = trails_orchestrator.run_pipeline(tmp_path)
    assert set(result.failures) == {bad.resolve()}
    assert result.participants["source_file"].tolist() == [str(good.resolve())]
    assert not (tmp_path / f"{bad.stem}_output.csv").exists()
    assert not (tmp_path / f"{bad.stem}_output_segments.csv").exists()
    retry = trails_orchestrator.run_pipeline(tmp_path, overwrite=True)
    assert not retry.failures
    assert len(retry.participants) == 2


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("phase", ["serialization", "publication"])
def test_summary_failure_preserves_successful_pairs_and_allows_retry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
    phase: str,
) -> None:
    """Summary failure propagates, preserves prior summary, and permits retry."""
    sources = [_input(tmp_path / f"[500000{i}]example-trail2.csv") for i in range(2)]
    summary = tmp_path / "directory_summary_statistics.csv"
    if overwrite:
        summary.write_text("keep", encoding="utf-8")
    original_replace = pathlib.Path.replace
    original_csv = pd.DataFrame.to_csv

    def failing_replace(path: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
        if target == summary and path.suffix == ".csv":
            raise OSError("simulated summary failure")
        return original_replace(path, target)

    def failing_csv(frame: pd.DataFrame, path: pathlib.Path, *, index: bool) -> None:
        if "line_number" not in frame and len(frame) == len(sources):
            raise OSError("simulated summary failure")
        original_csv(frame, path, index=index)

    with monkeypatch.context() as context:
        if phase == "serialization":
            context.setattr(pd.DataFrame, "to_csv", failing_csv)
        else:
            context.setattr(pathlib.Path, "replace", failing_replace)
        with pytest.raises(OSError, match="simulated summary"):
            trails_orchestrator.run_pipeline(tmp_path, overwrite=overwrite)
    if overwrite:
        assert summary.read_text(encoding="utf-8") == "keep"
    else:
        assert not summary.exists()
    for source in sources:
        assert (tmp_path / f"{source.stem}_output.csv").is_file()
        assert (tmp_path / f"{source.stem}_output_segments.csv").is_file()
    retry = trails_orchestrator.run_pipeline(tmp_path, overwrite=True)
    assert len(retry.participants) == len(sources)
    assert not retry.failures
    assert summary.is_file()


def test_small_trajectory_retains_available_features(tmp_path: pathlib.Path) -> None:
    """Short trajectories retain speed features without uncomputed defaults."""
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    data = pd.read_csv(source)
    data.loc[data["line_number"] == 0].iloc[[0, 1, 5]].to_csv(source, index=False)
    result = trails_orchestrator.run_pipeline(source)
    row = result.segments.iloc[0]
    summary = result.participants.iloc[0]
    unavailable = [
        "smoothness",
        "hesitation_count",
        "hesitation_duration",
        "mean_acceleration",
        "min_velocity",
        "max_velocity",
    ]

    assert row[["distance", "mean_speed", "speed_variance"]].notna().all()
    assert row[unavailable].isna().all()
    assert pd.notna(summary["mean_correct_speed_variance"])
    assert pd.isna(summary["mean_correct_smoothness"])
    assert pd.isna(summary["mean_correct_mean_acceleration"])


def test_missing_circle_exit_writes_nothing(tmp_path: pathlib.Path) -> None:
    """Required think-time failures propagate before creating outputs."""
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    data = pd.read_csv(source)
    data.loc[data["line_number"] == 0, ["x", "y"]] = data.loc[0, ["x", "y"]].values
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing exit"):
        trails_orchestrator.run_pipeline(source)
    assert not (tmp_path / f"{source.stem}_output.csv").exists()
    assert not (tmp_path / f"{source.stem}_output_segments.csv").exists()


def test_invalid_paths_and_unsupported_tasks(tmp_path: pathlib.Path) -> None:
    """Reject invalid inputs, unsupported tasks and invalid output destinations."""
    with pytest.raises(ValueError, match="Input path does not exist"):
        trails_orchestrator.run_pipeline(tmp_path / "missing.csv")
    unsupported = _input(tmp_path / "[5000000]example-trail1.csv")
    with pytest.raises(ValueError, match="Unsupported"):
        trails_orchestrator.run_pipeline(unsupported)
    source = _input(tmp_path / "[5000000]example-trail2.csv")
    with pytest.raises(ValueError, match=".csv extension"):
        trails_orchestrator.run_pipeline(source, tmp_path / "output.txt")
    with pytest.raises(NotADirectoryError):
        trails_orchestrator.run_pipeline(source, tmp_path / "missing" / "output.csv")
    with pytest.raises(NotADirectoryError):
        trails_orchestrator.run_pipeline(tmp_path, source)
