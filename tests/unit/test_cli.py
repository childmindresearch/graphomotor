"""Test cases for cli.py functions."""

import logging
import pathlib
import re

import pandas as pd
import pytest
from typer import testing

from graphomotor.core import cli, config


@pytest.fixture
def runner() -> testing.CliRunner:
    """Create a CLI test runner."""
    return testing.CliRunner()


def _clean_output(output: str) -> str:
    """Clean the output string by removing ANSI escape sequences and box characters."""
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    clean_output = "".join(char for char in clean_output if char not in set("╰─╯╭╮│"))
    return " ".join(clean_output.split())


def _write_trails_input(path: pathlib.Path, trail: str) -> pathlib.Path:
    """Write a clean synthetic recording with correct and error segments."""
    circles_path = (
        pathlib.Path(config.__file__).parent.parent / "utils/trails_points_scaled.json"
    )
    circles = config.load_scaled_circles(str(circles_path))[trail]
    first, second, third, fourth = (
        ("1", "2", "3", "4") if trail == "trail2" else ("1", "A", "2", "B")
    )
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


def _assert_trails_outputs(
    output: pathlib.Path, source: pathlib.Path, trail: str
) -> None:
    """Check both exported tables retain recording identity and computed features."""
    participants = pd.read_csv(output)
    segments = pd.read_csv(output.with_name(f"{output.stem}_segments.csv"))
    assert participants["source_file"].tolist() == [str(source.resolve())]
    assert participants["trail"].tolist() == [trail]
    assert participants["total_errors"].tolist() == [1]
    assert participants["correct_segment_count"].tolist() == [2]
    assert participants["error_segment_count"].tolist() == [1]
    assert len(segments) == 3
    assert set(segments["source_file"]) == {str(source.resolve())}
    assert set(segments["trail"]) == {trail}
    assert {"distance", "mean_speed", "think_time", "is_error"} <= set(segments.columns)


def test_cli_trails_help(runner: testing.CliRunner) -> None:
    """Root and command help advertise Trails without Spiral-only options."""
    root = runner.invoke(cli.app, ["--help"])
    result = runner.invoke(cli.app, ["extract-trails", "--help"])
    help_text = _clean_output(result.stdout)
    assert root.exit_code == result.exit_code == 0
    assert "extract-trails" in _clean_output(root.stdout)
    assert "input_path" in help_text.lower() and "output_path" in help_text.lower()
    assert "--overwrite" in help_text
    for option in (
        "--features",
        "--center-x",
        "--center-y",
        "--start-radius",
        "--growth-rate",
        "--start-angle",
        "--end-angle",
        "--num-points",
        "--include-reference",
        "--color-segments",
    ):
        assert option not in help_text


@pytest.mark.parametrize("trail", ["trail2", "trail4"])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_trails_single_outputs(
    runner: testing.CliRunner, tmp_path: pathlib.Path, trail: str, explicit_output: bool
) -> None:
    """Single recordings write both CSVs to explicit or default destinations."""
    source = _write_trails_input(tmp_path / f"[5000000]example-{trail}.csv", trail)
    output = tmp_path / (
        "custom.csv" if explicit_output else f"{source.stem}_output.csv"
    )
    args = ["extract-trails", str(source)]
    if explicit_output:
        args.append(str(output))
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.stderr
    _assert_trails_outputs(output, source, trail)


@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_trails_batch_outputs(
    runner: testing.CliRunner, tmp_path: pathlib.Path, explicit_output: bool
) -> None:
    """Batch output preserves repeated participants as separate recordings."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    sources = [
        _write_trails_input(source_dir / f"[5000000]example-{trail}.csv", trail)
        for trail in ("trail2", "trail4")
    ]
    destination = tmp_path / "outputs" if explicit_output else source_dir
    if explicit_output:
        destination.mkdir()
    args = ["extract-trails", str(source_dir)]
    if explicit_output:
        args.append(str(destination))
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.stderr
    for source, trail in zip(sources, ("trail2", "trail4")):
        _assert_trails_outputs(destination / f"{source.stem}_output.csv", source, trail)
    summary = pd.read_csv(destination / "directory_summary_statistics.csv")
    assert len(summary) == 2
    assert set(summary["source_file"]) == {str(source.resolve()) for source in sources}
    assert set(summary["trail"]) == {"trail2", "trail4"}


def test_cli_trails_requires_input(runner: testing.CliRunner) -> None:
    """Input path is required, while output is optional."""
    result = runner.invoke(cli.app, ["extract-trails"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "input_path" in result.stderr.lower()


@pytest.mark.parametrize("verbosity, level", [("-v", "INFO"), ("-vv", "DEBUG")])
def test_cli_trails_verbosity(
    runner: testing.CliRunner,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    verbosity: str,
    level: str,
) -> None:
    """Root verbosity controls logging during Trails extraction."""
    source = _write_trails_input(tmp_path / "[5000000]example-trail2.csv", "trail2")
    with caplog.at_level(logging.DEBUG, logger="graphomotor"):
        result = runner.invoke(cli.app, [verbosity, "extract-trails", str(source)])
        assert result.exit_code == 0, result.stderr
        assert cli.logger.level == getattr(logging, level)
        if level == "DEBUG":
            assert "Running Graphomotor pipeline with these arguments" in caplog.text


def test_cli_trails_collision_and_overwrite(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Existing outputs are protected unless overwrite is explicitly requested."""
    source = _write_trails_input(tmp_path / "[5000000]example-trail2.csv", "trail2")
    output = tmp_path / "result.csv"
    output.write_text("keep", encoding="utf-8")
    args = ["extract-trails", str(source), str(output)]
    rejected = runner.invoke(cli.app, args)
    assert rejected.exit_code != 0
    assert "Output already exists" in rejected.stderr
    assert output.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "result_segments.csv").exists()
    accepted = runner.invoke(cli.app, [*args, "--overwrite"])
    assert accepted.exit_code == 0, accepted.stderr
    _assert_trails_outputs(output, source, "trail2")


def test_cli_trails_overwrite_never_replaces_input(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Overwrite cannot bypass the orchestrator's input-alias protection."""
    source = _write_trails_input(tmp_path / "[5000000]example-trail2.csv", "trail2")
    original = source.read_bytes()
    result = runner.invoke(
        cli.app, ["extract-trails", str(source), str(source), "--overwrite"]
    )
    assert result.exit_code != 0
    assert "Output aliases input" in result.stderr
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "unsupported",
        "bad_extension",
        "missing_parent",
        "batch_file",
        "batch_missing_dir",
    ],
)
def test_cli_trails_invalid_paths(
    runner: testing.CliRunner, tmp_path: pathlib.Path, case: str
) -> None:
    """Invalid input and output paths surface orchestrator errors on stderr."""
    source = _write_trails_input(tmp_path / "[5000000]example-trail2.csv", "trail2")
    args = ["extract-trails", str(source)]
    expected = {
        "missing": "Input path does not exist",
        "unsupported": "Unsupported Trails task",
        "bad_extension": ".csv extension",
        "missing_parent": "Output directory does not exist",
        "batch_file": "Directory does not exist",
        "batch_missing_dir": "Directory does not exist",
    }[case]
    if case == "missing":
        args[1] = str(tmp_path / "missing.csv")
    elif case == "unsupported":
        args[1] = str(source.rename(tmp_path / "[5000000]example-trail1.csv"))
    elif case == "bad_extension":
        args.append(str(tmp_path / "out.txt"))
    elif case == "missing_parent":
        args.append(str(tmp_path / "missing" / "out.csv"))
    else:
        args[1] = str(tmp_path)
        args.append(str(source if case == "batch_file" else tmp_path / "missing"))
    result = runner.invoke(cli.app, args)
    assert result.exit_code != 0
    assert "Error:" in result.stderr
    assert expected in result.stderr


def test_cli_trails_partial_failure(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Mixed batch returns 1 with recording counts while preserving successful CSVs."""
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    destination = tmp_path / "outputs"
    destination.mkdir()
    good = [
        _write_trails_input(source_dir / f"[5000000]good-{trail}.csv", trail)
        for trail in ("trail2", "trail4")
    ]
    for index in range(3):
        (source_dir / f"[500000{index}]bad-trail2.csv").write_text(
            "broken\n1\n", encoding="utf-8"
        )
    result = runner.invoke(
        cli.app, ["extract-trails", str(source_dir), str(destination)]
    )
    assert result.exit_code == 1
    assert "2 succeeded" in result.stderr
    assert "3 failed" in result.stderr
    assert "Error:" not in result.stderr
    for source, trail in zip(good, ("trail2", "trail4")):
        _assert_trails_outputs(destination / f"{source.stem}_output.csv", source, trail)
    summary = pd.read_csv(destination / "directory_summary_statistics.csv")
    assert set(summary["source_file"]) == {str(source.resolve()) for source in good}


def test_cli_trails_all_failed(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """An all-failed batch reports the orchestrator error and writes no summary."""
    bad = tmp_path / "[5000000]bad-trail2.csv"
    bad.write_text("broken\n1\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["extract-trails", str(tmp_path)])
    assert result.exit_code != 0
    assert "Error: Could not extract features from any file" in result.stderr
    assert bad.name in result.stderr
    assert not (tmp_path / "directory_summary_statistics.csv").exists()


def test_cli_root_no_command(runner: testing.CliRunner) -> None:
    """Test CLI root shows help when no command is provided."""
    result = runner.invoke(cli.app, [])
    clean_stdout = _clean_output(result.stdout)

    assert result.exit_code == 0
    assert "Usage:" in clean_stdout
    assert "Graphomotor: A Python toolkit" in clean_stdout
    assert "extract" in clean_stdout
    assert "--verbosity" in clean_stdout
    assert "--version" in clean_stdout


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_cli_version_flags(runner: testing.CliRunner, flag: str) -> None:
    """Test that both --version and -V flags display version information."""
    result = runner.invoke(cli.app, [flag])

    assert result.exit_code == 0
    assert "Graphomotor version:" in result.stdout


def test_cli_extract_with_single_input_all_parameters(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI with single input and all available parameters."""
    output_file = tmp_path / "full_params_output.csv"

    result = runner.invoke(
        cli.app,
        [
            "-v",
            "extract-spiral",
            str(sample_spiral_data),
            str(output_file),
            "--features",
            "duration",
            "--features",
            "AUC",
            "--center-x",
            "100.0",
            "--center-y",
            "200.0",
            "--start-radius",
            "5.0",
            "--growth-rate",
            "1.5",
            "--start-angle",
            "0.5",
            "--end-angle",
            "20.0",
            "--num-points",
            "5000",
        ],
    )
    content = output_file.read_text()

    assert result.exit_code == 0
    assert output_file.exists()
    assert "5000000" in content
    assert "duration" in content
    assert "area_under_curve" in content
    assert "hausdorff" not in content
    assert "velocity" not in content


def test_cli_extract_with_directory_input(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test CLI with directory path containing CSV files."""
    output_file = tmp_path / "batch_output.csv"
    input_dir = sample_spiral_data.parent

    result = runner.invoke(
        cli.app,
        ["extract-spiral", str(input_dir), str(output_file)],
    )
    content = output_file.read_text()

    assert result.exit_code == 0
    assert output_file.exists()
    assert "5000000" in content


@pytest.mark.parametrize(
    "verbosity_level, expected_log_level", [(1, "INFO"), (2, "DEBUG")]
)
def test_cli_extract_with_verbosity(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    verbosity_level: int,
    expected_log_level: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test CLI with different verbosity levels."""
    output_file = tmp_path / "verbose_output.csv"
    verbosity_args = ["-" + "v" * verbosity_level]

    result = runner.invoke(
        cli.app,
        verbosity_args + ["extract-spiral", str(sample_spiral_data), str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert expected_log_level == caplog.get_records("call")[0].levelname


def test_cli_extract_verbosity_invalid_level(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test CLI with invalid verbosity level."""
    output_file = tmp_path / "edge_case_output.csv"
    verbosity_args = ["-" + "v" * 3]
    result = runner.invoke(
        cli.app,
        verbosity_args + ["extract-spiral", str(sample_spiral_data), str(output_file)],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    assert "Invalid verbosity level" in caplog.text


def test_cli_extract_help_flag(runner: testing.CliRunner) -> None:
    """Test --help flag displays expected information."""
    result = runner.invoke(cli.app, ["extract-spiral", "--help"])
    clean_stdout = _clean_output(result.stdout)

    assert result.exit_code == 0
    assert "Usage:" in clean_stdout
    assert "Extract features from spiral drawing data" in clean_stdout
    assert "--features" in clean_stdout
    assert "duration|velocity" in clean_stdout
    assert "--center-x" in clean_stdout
    assert "--growth-rate" in clean_stdout


def test_cli_extract_missing_arguments(runner: testing.CliRunner) -> None:
    """Test CLI fails with missing required arguments."""
    result = runner.invoke(cli.app, ["extract-spiral"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "input_path" in result.stderr.lower()


def test_cli_extract_missing_output_path(
    runner: testing.CliRunner, sample_spiral_data: pathlib.Path
) -> None:
    """Test CLI fails with missing output path."""
    result = runner.invoke(cli.app, ["extract-spiral", str(sample_spiral_data)])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "output_path" in result.stderr.lower()


def test_cli_extract_nonexistent_input_path(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Test CLI handles nonexistent input path."""
    nonexistent_path = tmp_path / "nonexistent.csv"
    output_file = tmp_path / "output.csv"

    result = runner.invoke(
        cli.app,
        ["extract-spiral", str(nonexistent_path), str(output_file)],
    )

    assert result.exit_code != 0
    assert "Error: Input path does not exist" in result.stderr


def test_cli_extract_invalid_features(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid feature categories."""
    output_file = tmp_path / "output.csv"

    result = runner.invoke(
        cli.app,
        [
            "extract-spiral",
            str(sample_spiral_data),
            str(output_file),
            "--features",
            "invalid_feature",
        ],
    )

    assert result.exit_code != 0
    assert (
        "'invalid_feature' is not one of 'duration', 'velocity', 'hausdorff', 'AUC'."
        in _clean_output(result.stderr)
    )


def test_cli_extract_invalid_output_extension(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid output file extension."""
    output_file = tmp_path / "output.txt"

    result = runner.invoke(
        cli.app,
        ["extract-spiral", str(sample_spiral_data), str(output_file)],
    )

    assert result.exit_code != 0
    assert "Error: Output file must have a .csv extension" in result.stderr


@pytest.mark.parametrize(
    "option, invalid_value",
    [
        ("--center-x", "what"),
        ("--center-y", "is"),
        ("--start-radius", "the"),
        ("--growth-rate", "meaning"),
        ("--start-angle", "of"),
        ("--end-angle", "life"),
        ("--num-points", "?"),
    ],
)
def test_cli_extract_invalid_option_types(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    option: str,
    invalid_value: str,
) -> None:
    """Test CLI handles invalid types for numeric options and shows error messages."""
    output_file = tmp_path / "output.csv"

    result = runner.invoke(
        cli.app,
        [
            "extract-spiral",
            str(sample_spiral_data),
            str(output_file),
            option,
            invalid_value,
        ],
    )

    assert result.exit_code != 0


def test_cli_plot_features_help_flag(runner: testing.CliRunner) -> None:
    """Test --help flag displays expected information for plot-features command."""
    result = runner.invoke(cli.app, ["plot-features", "--help"])
    clean_stdout = _clean_output(result.stdout)

    assert result.exit_code == 0
    assert "Usage:" in clean_stdout
    assert "Generate plots from extracted features" in clean_stdout
    assert "--plot-types" in clean_stdout
    assert "dist|trends|boxplot" in clean_stdout
    assert "--features" in clean_stdout


def test_cli_plot_features_missing_arguments(runner: testing.CliRunner) -> None:
    """Test CLI fails with missing required arguments for plot-features command."""
    result = runner.invoke(cli.app, ["plot-features"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "input_path" in result.stderr.lower()


def test_cli_plot_features_missing_output_path(
    runner: testing.CliRunner,
    sample_batch_features_df: pd.DataFrame,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI fails with missing output path for plot-features command."""
    input_file = tmp_path / "features.csv"
    sample_batch_features_df.to_csv(input_file, index=False)

    result = runner.invoke(cli.app, ["plot-features", str(input_file)])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "output_path" in result.stderr.lower()


def test_cli_plot_features_nonexistent_input_file(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Test CLI handles nonexistent input file for plot-features command."""
    nonexistent_file = tmp_path / "nonexistent.csv"
    output_dir = tmp_path / "plots"

    result = runner.invoke(
        cli.app,
        ["plot-features", str(nonexistent_file), str(output_dir)],
    )

    assert result.exit_code == 1
    assert (
        f"Error: Input path {nonexistent_file} must be an existing CSV file"
        in result.stderr
    )


def test_cli_plot_features_invalid_input_extension(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Test CLI handles invalid input file extension for plot-features command."""
    input_file = tmp_path / "features.txt"
    input_file.write_text("some text")
    output_dir = tmp_path / "plots"

    result = runner.invoke(
        cli.app,
        ["plot-features", str(input_file), str(output_dir)],
    )

    assert result.exit_code == 1
    assert "must be an existing CSV file" in result.stderr


def test_cli_plot_features_mkdir_permission_error(
    sample_batch_features: pathlib.Path,
    runner: testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI handles permission error when creating output directory."""
    output_dir = tmp_path / "plots"

    monkeypatch.setattr(
        "pathlib.Path.mkdir",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("Permission denied")
        ),
    )

    result = runner.invoke(
        cli.app,
        ["plot-features", str(sample_batch_features), str(output_dir)],
    )

    assert result.exit_code != 0
    assert "Error creating output directory" in result.stderr


def test_cli_plot_features_output_path_is_file(
    sample_batch_features: pathlib.Path,
    runner: testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles output path that is a file instead of directory."""
    output_file = tmp_path / "output.txt"
    output_file.write_text("existing file")

    result = runner.invoke(
        cli.app,
        ["plot-features", str(sample_batch_features), str(output_file)],
    )

    assert result.exit_code != 0
    assert "Error creating output directory" in result.stderr


def test_cli_plot_features_invalid_plot_types(
    sample_batch_features: pathlib.Path,
    runner: testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid plot types for plot-features command."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    result = runner.invoke(
        cli.app,
        [
            "plot-features",
            str(sample_batch_features),
            str(output_dir),
            "--plot-types",
            "invalid_plot_type",
        ],
    )

    assert result.exit_code != 0
    assert (
        "'invalid_plot_type' is not one of 'dist', 'trends', 'boxplot', 'cluster'."
        in _clean_output(result.stderr)
    )


def test_cli_plot_features_invalid_features(
    sample_batch_features: pathlib.Path,
    runner: testing.CliRunner,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid features for plot-features command."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    result = runner.invoke(
        cli.app,
        [
            "plot-features",
            str(sample_batch_features),
            str(output_dir),
            "--features",
            "invalid_feature",
        ],
    )

    assert result.exit_code != 0
    assert "Error generating plots" in result.stderr


def test_cli_plot_features_with_specific_parameters(
    runner: testing.CliRunner,
    sample_batch_features: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI calls specified plot function with specific features."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    calls = []
    monkeypatch.setattr(
        "graphomotor.plot.feature_plots.plot_feature_distributions",
        lambda **kwargs: calls.append(("plot_feature_distributions", kwargs)),
    )

    result = runner.invoke(
        cli.app,
        [
            "plot-features",
            str(sample_batch_features),
            str(output_dir),
            "--plot-types",
            "dist",
            "--features",
            "duration",
        ],
    )

    assert result.exit_code == 0
    assert f"All plots saved to: {output_dir}" in result.stdout
    assert len(calls) == 1
    assert calls[0][0] == "plot_feature_distributions"
    assert calls[0][1] == {
        "data": sample_batch_features,
        "output_path": output_dir,
        "features": ["duration"],
    }


def test_cli_plot_features_all_plot_types_default(
    runner: testing.CliRunner,
    sample_batch_features: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI calls all plot functions when none specified."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    plot_functions = [
        "plot_feature_distributions",
        "plot_feature_trends",
        "plot_feature_boxplots",
        "plot_feature_clusters",
    ]

    calls = []
    mock_functions = {
        func_name: (lambda fn: lambda **kwargs: calls.append((fn, kwargs)))(func_name)
        for func_name in plot_functions
    }

    for func_name, mock_func in mock_functions.items():
        monkeypatch.setattr(f"graphomotor.plot.feature_plots.{func_name}", mock_func)

    expected_args = {
        "data": sample_batch_features,
        "output_path": output_dir,
        "features": None,
    }
    expected_calls = [(func_name, expected_args) for func_name in plot_functions]

    result = runner.invoke(
        cli.app,
        ["plot-features", str(sample_batch_features), str(output_dir)],
    )

    assert result.exit_code == 0
    assert f"All plots saved to: {output_dir}" in result.stdout
    assert len(calls) == 4
    for expected_call in expected_calls:
        assert expected_call in calls


def test_cli_plot_spiral_help_flag(runner: testing.CliRunner) -> None:
    """Test --help flag displays expected information for plot-spiral command."""
    result = runner.invoke(cli.app, ["plot-spiral", "--help"])
    clean_stdout = _clean_output(result.stdout)

    assert result.exit_code == 0
    assert "Usage:" in clean_stdout
    assert "Visualize spiral drawing trajectories" in clean_stdout
    assert "--include-reference" in clean_stdout
    assert "--color-segments" in clean_stdout
    assert "--center-x" in clean_stdout
    assert "--growth-rate" in clean_stdout


def test_cli_plot_spiral_missing_arguments(runner: testing.CliRunner) -> None:
    """Test CLI fails with missing required arguments for plot-spiral command."""
    result = runner.invoke(cli.app, ["plot-spiral"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "input_path" in result.stderr.lower()


def test_cli_plot_spiral_missing_output_path(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
) -> None:
    """Test CLI fails with missing output path for plot-spiral command."""
    result = runner.invoke(cli.app, ["plot-spiral", str(sample_spiral_data)])
    assert result.exit_code != 0
    assert "Missing argument" in result.stderr
    assert "output_path" in result.stderr.lower()


def test_cli_plot_spiral_nonexistent_input_path(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Test CLI handles nonexistent input path for plot-spiral command."""
    nonexistent_path = tmp_path / "nonexistent.csv"
    output_dir = tmp_path / "plots"

    result = runner.invoke(
        cli.app,
        ["plot-spiral", str(nonexistent_path), str(output_dir)],
    )

    assert result.exit_code == 1
    assert f"Error: Input path {nonexistent_path} does not exist" in result.stderr


def test_cli_plot_spiral_invalid_input_extension_file(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Test CLI handles invalid input file extension for plot-spiral command."""
    input_file = tmp_path / "spiral.txt"
    input_file.write_text("some text")
    output_dir = tmp_path / "plots"

    result = runner.invoke(
        cli.app,
        ["plot-spiral", str(input_file), str(output_dir)],
    )

    assert result.exit_code == 1
    assert f"Error: Input file {input_file} must have .csv extension" in result.stderr


def test_cli_plot_spiral_mkdir_permission_error(
    sample_spiral_data: pathlib.Path,
    runner: testing.CliRunner,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI handles permission error when creating output directory."""
    output_dir = tmp_path / "plots"

    monkeypatch.setattr(
        "pathlib.Path.mkdir",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("Permission denied")
        ),
    )

    result = runner.invoke(
        cli.app,
        ["plot-spiral", str(sample_spiral_data), str(output_dir)],
    )

    assert result.exit_code != 0
    assert "Error creating output directory" in result.stderr


def test_cli_plot_spiral_single_file_with_all_parameters(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with single file input and all available parameters."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    calls = []
    monkeypatch.setattr(
        "graphomotor.plot.spiral_plots.plot_single_spiral",
        lambda **kwargs: calls.append(("plot_single_spiral", kwargs)),
    )

    result = runner.invoke(
        cli.app,
        [
            "plot-spiral",
            str(sample_spiral_data),
            str(output_dir),
            "--include-reference",
            "--color-segments",
            "--center-x",
            "100.0",
            "--center-y",
            "200.0",
            "--start-radius",
            "5.0",
            "--growth-rate",
            "1.5",
            "--start-angle",
            "0.5",
            "--end-angle",
            "20.0",
            "--num-points",
            "5000",
        ],
    )
    call_kwargs = calls[0][1]
    spiral_config = call_kwargs["spiral_config"]

    assert result.exit_code == 0
    assert f"Single spiral plot saved to: {output_dir}" in result.stdout
    assert len(calls) == 1
    assert calls[0][0] == "plot_single_spiral"

    assert call_kwargs["data"] == sample_spiral_data
    assert call_kwargs["output_path"] == output_dir
    assert call_kwargs["include_reference"] is True
    assert call_kwargs["color_segments"] is True

    assert spiral_config.center_x == 100.0
    assert spiral_config.center_y == 200.0
    assert spiral_config.start_radius == 5.0
    assert spiral_config.growth_rate == 1.5
    assert spiral_config.start_angle == 0.5
    assert spiral_config.end_angle == 20.0
    assert spiral_config.num_points == 5000


def test_cli_plot_spiral_directory_input(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with directory input for batch plotting."""
    input_dir = sample_spiral_data.parent
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    calls = []
    monkeypatch.setattr(
        "graphomotor.plot.spiral_plots.plot_batch_spirals",
        lambda **kwargs: calls.append(("plot_batch_spirals", kwargs)),
    )

    result = runner.invoke(
        cli.app,
        ["plot-spiral", str(input_dir), str(output_dir)],
    )
    call_kwargs = calls[0][1]

    assert result.exit_code == 0
    assert f"Batch spiral plots saved to: {output_dir}" in result.stdout
    assert len(calls) == 1
    assert calls[0][0] == "plot_batch_spirals"

    assert call_kwargs["data"] == input_dir
    assert call_kwargs["output_path"] == output_dir
    assert call_kwargs["include_reference"] is False
    assert call_kwargs["color_segments"] is False


def test_cli_plot_spiral_with_boolean_flags(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI with boolean flags for include-reference and color-segments."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    calls = []
    monkeypatch.setattr(
        "graphomotor.plot.spiral_plots.plot_single_spiral",
        lambda **kwargs: calls.append(("plot_single_spiral", kwargs)),
    )

    result = runner.invoke(
        cli.app,
        [
            "plot-spiral",
            str(sample_spiral_data),
            str(output_dir),
            "--include-reference",
            "--color-segments",
        ],
    )
    call_kwargs = calls[0][1]

    assert result.exit_code == 0
    assert len(calls) == 1

    assert call_kwargs["include_reference"] is True
    assert call_kwargs["color_segments"] is True


def test_cli_plot_spiral_invalid_option_types(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid types for numeric options."""
    output_dir = tmp_path / "plots"

    result = runner.invoke(
        cli.app,
        [
            "plot-spiral",
            str(sample_spiral_data),
            str(output_dir),
            "--center-x",
            "invalid",
        ],
    )

    assert result.exit_code != 0
    assert "'invalid' is not a valid float" in _clean_output(result.stderr)


def test_cli_plot_spiral_plot_function_exception(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI handles exceptions from plot functions."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    monkeypatch.setattr(
        "graphomotor.plot.spiral_plots.plot_single_spiral",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("Test plotting error")),
    )

    result = runner.invoke(
        cli.app,
        ["plot-spiral", str(sample_spiral_data), str(output_dir)],
    )

    assert result.exit_code != 0
    assert "Error generating spiral plots" in result.stderr


def test_cli_plot_spiral_directory_vs_file_behavior(
    runner: testing.CliRunner,
    sample_spiral_data: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CLI calls different functions for file vs directory input."""
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    input_dir = sample_spiral_data.parent

    calls = []

    monkeypatch.setattr(
        "graphomotor.plot.spiral_plots.plot_single_spiral",
        lambda **kwargs: calls.append(("single", kwargs)),
    )
    monkeypatch.setattr(
        "graphomotor.plot.spiral_plots.plot_batch_spirals",
        lambda **kwargs: calls.append(("batch", kwargs)),
    )

    result_file = runner.invoke(
        cli.app,
        ["plot-spiral", str(sample_spiral_data), str(output_dir)],
    )

    result_dir = runner.invoke(
        cli.app,
        ["plot-spiral", str(input_dir), str(output_dir)],
    )

    assert result_file.exit_code == 0
    assert "Single spiral plot saved" in result_file.stdout

    assert result_dir.exit_code == 0
    assert "Batch spiral plots saved" in result_dir.stdout

    assert len(calls) == 2
    assert calls[0][0] == "single"
    assert calls[1][0] == "batch"
