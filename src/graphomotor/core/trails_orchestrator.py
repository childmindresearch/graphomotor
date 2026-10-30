"""Compose existing Trails metrics into per-segment and per-recording CSVs.

Input timestamps are assumed to have been cleaned as required by the reader.
Writes are staged and rolled back on exceptions, assuming a single writer; they
are not crash-atomic across files. Think time can include error time, and segments
can group repeated paths.
"""

import dataclasses
import pathlib
import tempfile

import pandas as pd

from graphomotor.core import config, models
from graphomotor.features import shared_features
from graphomotor.features.trails import drawing_metrics, time
from graphomotor.io import reader
from graphomotor.utils import trails_utils

_METRICS = (
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
)
_TOTAL_METRICS = frozenset(
    ("ink_time", "think_time", "distance", "hesitation_count", "hesitation_duration")
)
_SEGMENT_COLUMNS = (
    "source_file",
    "start_time",
    "participant_id",
    "trail",
    "line_number",
    "start_label",
    "end_label",
    "is_error",
    "ink_time",
    "think_time",
    "think_circle_label",
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
)
_PARTICIPANT_COLUMNS = (
    "source_file",
    "start_time",
    "participant_id",
    "trail",
    "total_time",
    "duration",
    "total_errors",
    "percent_accurate_paths",
    "total_error_time",
    "correct_segment_count",
    "error_segment_count",
) + tuple(
    column
    for group in ("correct", "error")
    for metric in _METRICS
    for column in (
        f"mean_{group}_{metric}",
        f"count_{group}_{metric}",
        *((f"total_{group}_{metric}",) if metric in _TOTAL_METRICS else ()),
    )
)
_CIRCLES_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "utils"
    / "trails_points_scaled.json"
)
_LOGGER = config.get_logger()


@dataclasses.dataclass
class TrailsResult:
    """Results for successful recordings and diagnostics for failed batch inputs.

    Attributes:
        segments: Segment features with recording provenance.
        participants: One summary row per input recording, not per unique person.
        failures: Input paths mapped to error descriptions; empty for single files.
    """

    segments: pd.DataFrame
    participants: pd.DataFrame
    failures: dict[pathlib.Path, str] = dataclasses.field(default_factory=dict)


def _segment_row(
    segment: models.LineSegment,
    drawing: models.Drawing,
    circles: dict[str, dict[str, models.CircleTarget]],
) -> dict:
    """Export only metrics actually calculated by the segment helper."""
    segment.points = segment.points.reset_index(drop=True)
    segment.compute_segment_metrics(circles, drawing.task_name)
    row = dict.fromkeys(_SEGMENT_COLUMNS)
    row.update(
        source_file=str(drawing.metadata["source_path"]),
        start_time=str(drawing.metadata["start_time"]),
        participant_id=str(drawing.metadata["id"]),
        trail=drawing.task_name,
        line_number=segment.line_number,
        start_label=segment.start_label,
        end_label=segment.end_label,
        is_error=segment.is_error,
    )
    if segment.think_circle_label and not segment.is_error:
        row.update(
            think_time=segment.think_time, think_circle_label=segment.think_circle_label
        )
    for metric in segment.available_metrics.intersection(_SEGMENT_COLUMNS):
        row[metric] = getattr(segment, metric)
    return row


def _participant_row(drawing: models.Drawing, segments: pd.DataFrame) -> dict:
    """Summarize available metrics separately for correct and error path groups."""
    row = {
        "source_file": str(drawing.metadata["source_path"]),
        "start_time": str(drawing.metadata["start_time"]),
        "participant_id": drawing.metadata["id"],
        "trail": drawing.task_name,
        "total_time": drawing.data["total_time"].iloc[0],
        **shared_features.get_task_duration(drawing),
        **drawing_metrics.get_total_errors(drawing),
        **drawing_metrics.percent_accurate_paths(drawing),
        **time.calculate_total_error_time(drawing),
    }
    for group, is_error in (("correct", False), ("error", True)):
        selected = segments[segments["is_error"] == is_error]
        row[f"{group}_segment_count"] = len(selected)
        for metric in _METRICS:
            values = selected[metric].dropna()
            row[f"count_{group}_{metric}"] = len(values)
            row[f"mean_{group}_{metric}"] = values.mean() if len(values) else None
            if metric in _TOTAL_METRICS:
                row[f"total_{group}_{metric}"] = values.sum() if len(values) else None
    return row


def _validate_output(
    path: pathlib.Path, inputs: list[pathlib.Path], overwrite: bool
) -> None:
    """Reject input aliases, non-CSV outputs and unsafe existing targets."""
    if any(
        path.resolve() == source.resolve() or (path.exists() and path.samefile(source))
        for source in inputs
    ):
        raise ValueError(f"Output aliases input: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Output file must have a .csv extension: {path}")
    if path.is_symlink() or (path.exists() and (not overwrite or not path.is_file())):
        raise FileExistsError(f"Output already exists: {path}")
    if not path.parent.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {path.parent}")


def _write_csvs(outputs: dict[pathlib.Path, pd.DataFrame], overwrite: bool) -> None:
    """Stage related CSVs and restore previous outputs if publication raises.

    All targets must share an existing directory and have passed validation.
    Exclusive reservation protects against accidentally replacing a newly created
    file. Rollback assumes a single writer and a functioning filesystem; it cannot
    provide a multi-file transaction across process or machine crashes.
    """
    directory = next(iter(outputs)).parent
    with tempfile.TemporaryDirectory(prefix=".graphomotor-", dir=directory) as work:
        staging = pathlib.Path(work)
        staged = {}
        for index, (path, frame) in enumerate(outputs.items()):
            staged[path] = staging / f"{index}.csv"
            frame.to_csv(staged[path], index=False)

        created: list[pathlib.Path] = []
        backups: dict[pathlib.Path, pathlib.Path] = {}
        try:
            for index, path in enumerate(outputs):
                if overwrite and path.exists():
                    backup = staging / f"{index}.backup"
                    path.replace(backup)
                    backups[path] = backup
                with path.open("x", encoding="utf-8"):
                    created.append(path)
                staged[path].replace(path)
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            for path, backup in backups.items():
                backup.replace(path)
            raise


def _process_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and write both feature tables for a supported input recording."""
    source = input_path.resolve()
    _LOGGER.debug("Processing Trails file: %s", source)
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise ValueError(f"Expected an existing CSV file: {source}")
    participant_path = (
        output_path
        if output_path is not None
        else source.with_name(f"{source.stem}_output.csv")
    )
    segment_path = participant_path.with_name(f"{participant_path.stem}_segments.csv")
    for path in (participant_path, segment_path):
        _validate_output(path, [source], overwrite)

    drawing = reader.load_drawing_data(source)
    _LOGGER.debug("Loaded %d rows for task %s", len(drawing.data), drawing.task_name)
    if drawing.task_name not in ("trail2", "trail4"):
        raise ValueError(f"Unsupported Trails task: {drawing.task_name}")

    circles = config.load_scaled_circles(str(_CIRCLES_PATH))
    data = drawing.data.copy()
    data["is_error"] = data["error"] != "E0"

    segments = trails_utils.segment_lines(data, drawing.task_name, circles)
    _LOGGER.debug("Calculating features for %d segments", len(segments))
    time.calculate_think_times(
        [s for s in segments if not s.is_error], circles, drawing.task_name
    )
    rows = [_segment_row(segment, drawing, circles) for segment in segments]

    segments_df = pd.DataFrame(rows, columns=_SEGMENT_COLUMNS)
    participant_df = pd.DataFrame(
        [_participant_row(drawing, segments_df)], columns=_PARTICIPANT_COLUMNS
    )
    _LOGGER.debug(
        "Writing recording and segment CSVs: %s, %s", participant_path, segment_path
    )
    _write_csvs(
        {participant_path: participant_df, segment_path: segments_df}, overwrite
    )
    _LOGGER.debug("Successfully processed Trails file: %s", source)
    return segments_df, participant_df


def _process_directory(
    source: pathlib.Path, destination: pathlib.Path, overwrite: bool
) -> TrailsResult:
    """Process direct children, retain successful pairs and summarize recordings."""
    _LOGGER.debug(
        "Processing Trails directory: %s; output directory: %s", source, destination
    )
    for directory in (source, destination):
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {directory}")

    summary_path = destination / "directory_summary_statistics.csv"
    inputs = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.stem.rsplit("-", 1)[-1] in ("trail2", "trail4")
    )
    _LOGGER.debug("Found %d supported Trails CSV files", len(inputs))
    if not inputs:
        raise ValueError(f"No supported Trails CSV files found in directory: {source}")

    _validate_output(summary_path, inputs, overwrite)

    segment_frames: list[pd.DataFrame] = []
    participant_frames: list[pd.DataFrame] = []
    failures: dict[pathlib.Path, str] = {}
    for file_index, path in enumerate(inputs, 1):
        _LOGGER.debug("Processing file %d/%d: %s", file_index, len(inputs), path.name)
        try:
            segment_df, participant_df = _process_file(
                path, destination / f"{path.stem}_output.csv", overwrite
            )
            segment_frames.append(segment_df)
            participant_frames.append(participant_df)
        except Exception as exc:
            failures[path.resolve()] = f"{type(exc).__name__}: {exc}"
            _LOGGER.error("Failed to process %s: %s", path.name, exc)

    if not participant_frames:
        details = "; ".join(f"{path.name}: {error}" for path, error in failures.items())
        raise ValueError(f"Could not extract features from any file: {details}")
    segments = pd.concat(segment_frames, ignore_index=True)
    participants = pd.concat(participant_frames, ignore_index=True)
    _LOGGER.debug("Writing directory summary CSV: %s", summary_path)
    _write_csvs({summary_path: participants}, overwrite)
    _LOGGER.debug(
        "Trails directory processing complete: %d succeeded, %d failed",
        len(participant_frames),
        len(failures),
    )
    return TrailsResult(segments, participants, failures)


def run_pipeline(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path | None = None,
    *,
    overwrite: bool = False,
) -> TrailsResult:
    """Extract Trails features from one cleaned CSV or a directory of recordings.

    Args:
        input_path: Curious trail2/trail4 CSV or a directory searched nonrecursively.
        output_path: Summary CSV for a single input, or an existing directory for
            batch outputs. Defaults to writing beside the input(s).
        overwrite: Replace existing regular output files. Symlinks and input aliases
            are always rejected. Use explicitly to retry a batch after a failure.

    Returns:
        Segment and recording-summary DataFrames in a TrailsResult, with per-file
        batch errors in failures. Each input produces a summary CSV and companion
        ``_segments.csv``. Batches also produce ``summary_statistics.csv``. Summary
        rows remain separate for repeated participants, trails and sessions.

    Raises:
        ValueError: If inputs or output suffixes are invalid, no supported files
            exist, all batch inputs fail, or required metrics cannot be computed.
        OSError: If outputs cannot be written or a collision is rejected.

    Notes:
        Successful pairs remain if writing the batch summary fails; the exception
        propagates, and overwrite=True permits retry. Ordinary write exceptions
        restore prior outputs, but multi-file writes are not crash-atomic. Use one
        writer per output directory. Existing metric formulas and path grouping
        are unchanged; unavailable calculated metrics are exported as nulls.
    """
    source = pathlib.Path(input_path)
    destination = pathlib.Path(output_path) if output_path is not None else None

    if source.is_file():
        segments, participants = _process_file(source, destination, overwrite)
        return TrailsResult(segments, participants)

    if source.is_dir():
        return _process_directory(source, destination or source, overwrite)
    raise ValueError(f"Input path does not exist or is not a file/directory: {source}")
