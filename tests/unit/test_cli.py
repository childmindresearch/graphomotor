"""Test cases for cli.py functions."""

import pathlib

import pytest
import typer
from typer import testing

from graphomotor.core import cli


@pytest.fixture
def runner() -> testing.CliRunner:
    """Create a CLI test runner."""
    return testing.CliRunner()


def test_version_callback_displays_version() -> None:
    """Test that version callback displays version and exits."""
    with pytest.raises(typer.Exit):
        cli.version_callback(True)


def test_version_callback_no_action_when_false() -> None:
    """Test that version callback does nothing when False."""
    cli.version_callback(False)


def test_cli_with_single_input_all_parameters(
    runner: testing.CliRunner,
    sample_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI with single input and all available parameters."""
    output_file = tmp_path / "full_params_output.csv"

    result = runner.invoke(
        cli.app,
        [
            str(sample_data),
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
            "-v",
        ],
    )
    content = output_file.read_text()

    assert result.exit_code == 0
    assert output_file.exists()
    assert "5123456" in content
    assert "duration" in content
    assert "area_under_curve" in content
    assert "hausdorff" not in content
    assert "velocity" not in content


def test_cli_with_directory_input(
    runner: testing.CliRunner,
    sample_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI with directory path containing CSV files."""
    output_file = tmp_path / "batch_output.csv"
    input_dir = sample_data.parent

    result = runner.invoke(
        cli.app,
        [str(input_dir), str(output_file)],
    )
    content = output_file.read_text()

    assert result.exit_code == 0
    assert output_file.exists()
    assert "5000000" in content
    assert "5123456" in content


@pytest.mark.parametrize(
    "verbosity_level, expected_log_level", [(1, "INFO"), (2, "DEBUG")]
)
def test_cli_with_verbosity(
    runner: testing.CliRunner,
    sample_data: pathlib.Path,
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
        [str(sample_data), str(output_file)] + verbosity_args,
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert expected_log_level == caplog.get_records("call")[0].levelname


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_cli_version_flags(runner: testing.CliRunner, flag: str) -> None:
    """Test that both --version and -V flags display version information."""
    result = runner.invoke(cli.app, [flag])

    assert result.exit_code == 0
    assert "Graphomotor version:" in result.stdout


def test_cli_other_options_ignored_with_version_flag(
    runner: testing.CliRunner, sample_data: pathlib.Path
) -> None:
    """Test --version flag ignores other options."""
    result = runner.invoke(
        cli.app, ["--version", str(sample_data), "--center-x", "100"]
    )

    assert result.exit_code == 0
    assert "Graphomotor version:" in result.stdout
    assert "--center-x" not in result.stdout


def test_cli_help_flag(runner: testing.CliRunner) -> None:
    """Test --help flag displays expected information."""
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "Graphomotor: A Python toolkit" in result.stdout
    assert "--features" in result.stdout
    assert "duration, velocity, hausdorff, AUC" in result.stdout
    assert "--center-x" in result.stdout
    assert "--growth-rate" in result.stdout


def test_cli_missing_arguments(runner: testing.CliRunner) -> None:
    """Test CLI fails with missing required arguments."""
    result = runner.invoke(cli.app, [])
    assert result.exit_code != 0


def test_cli_missing_output_path(
    runner: testing.CliRunner, sample_data: pathlib.Path
) -> None:
    """Test CLI fails with missing output path."""
    result = runner.invoke(cli.app, [str(sample_data)])
    assert result.exit_code != 0


def test_cli_nonexistent_input_path(
    runner: testing.CliRunner, tmp_path: pathlib.Path
) -> None:
    """Test CLI handles nonexistent input path."""
    nonexistent_path = tmp_path / "nonexistent.csv"
    output_file = tmp_path / "output.csv"

    result = runner.invoke(
        cli.app,
        [str(nonexistent_path), str(output_file)],
    )

    assert result.exit_code != 0
    assert "Error: Input path does not exist" in result.stderr


def test_cli_invalid_features(
    runner: testing.CliRunner,
    sample_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid feature categories."""
    output_file = tmp_path / "output.csv"

    result = runner.invoke(
        cli.app,
        [
            str(sample_data),
            str(output_file),
            "--features",
            "invalid_feature",
        ],
    )

    assert result.exit_code != 0
    assert "Error: No valid feature categories provided" in result.stderr


def test_cli_invalid_output_extension(
    runner: testing.CliRunner,
    sample_data: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Test CLI handles invalid output file extension."""
    output_file = tmp_path / "output.txt"

    result = runner.invoke(
        cli.app,
        [str(sample_data), str(output_file)],
    )

    assert result.exit_code != 0
    assert "Error: Output file must have a .csv extension" in result.stderr


@pytest.mark.parametrize(
    "option, invalid_value, expected_type",
    [
        ("--center-x", "what", "float"),
        ("--center-y", "is", "float"),
        ("--start-radius", "the", "float"),
        ("--growth-rate", "meaning", "float"),
        ("--start-angle", "of", "float"),
        ("--end-angle", "life", "float"),
        ("--num-points", "?", "integer"),
    ],
)
def test_cli_invalid_option_types(
    runner: testing.CliRunner,
    sample_data: pathlib.Path,
    tmp_path: pathlib.Path,
    option: str,
    invalid_value: str,
    expected_type: str,
) -> None:
    """Test CLI handles invalid types for numeric options and shows error messages."""
    output_file = tmp_path / "output.csv"

    result = runner.invoke(
        cli.app,
        [str(sample_data), str(output_file), option, invalid_value],
    )
    clean_stderr = "".join(char for char in result.stderr if char not in set("╰─╯╭╮│"))
    clean_stderr = " ".join(clean_stderr.split())

    assert result.exit_code != 0
    assert f"'{invalid_value}' is not a valid {expected_type}" in clean_stderr
