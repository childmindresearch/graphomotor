"""Command-line interface for the Graphomotor."""

import enum
import pathlib
import typing

import typer

from graphomotor.core import config, orchestrator

logger = config.get_logger()
app = typer.Typer()


class ValidFeatureCategories(str, enum.Enum):
    """Valid feature categories for extraction."""

    DURATION = "duration"
    VELOCITY = "velocity"
    HAUSDORFF = "hausdorff"
    AUC = "AUC"


def version_callback(version: bool) -> None:
    """Print the current version of graphomotor."""
    if version:
        typer.echo(f"Graphomotor version: {config.get_version()}")
        raise typer.Exit()


@app.command(
    name="graphomotor",
    help=(
        "Graphomotor: A Python toolkit for analyzing graphomotor data "
        "collected via Curious. See the README for usage details."
    ),
    epilog=(
        "Please report issues at "
        "https://github.com/ChildMindInstitute/graphomotor/issues"
    ),
    no_args_is_help=True,
)
def main(
    input_path: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            help=(
                "Path to a CSV file or directory containing CSV files "
                "with Curious drawing data."
            ),
        ),
    ],
    output_path: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            help=(
                "Output path for extracted metadata and features. If a directory, "
                "auto-generates filename. If a file, must have .csv extension."
            ),
        ),
    ],
    features: typing.Annotated[
        typing.Optional[list[ValidFeatureCategories]],
        typer.Option(
            "--features",
            "-f",
            help=(
                "Feature categories to extract. Can be specified multiple times. "
                "If omitted, all available features are extracted."
            ),
        ),
    ] = None,
    center_x: typing.Annotated[
        float,
        typer.Option(
            "--center-x",
            "-x",
            help="X-coordinate of the reference spiral center.",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.center_x,
    center_y: typing.Annotated[
        float,
        typer.Option(
            "--center-y",
            "-y",
            help="Y-coordinate of the reference spiral center.",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.center_y,
    start_radius: typing.Annotated[
        float,
        typer.Option(
            "--start-radius",
            "-r",
            help="Starting radius of the reference spiral.",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.start_radius,
    growth_rate: typing.Annotated[
        float,
        typer.Option(
            "--growth-rate",
            "-g",
            help="Growth rate of the reference spiral.",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.growth_rate,
    start_angle: typing.Annotated[
        float,
        typer.Option(
            "--start-angle",
            "-s",
            help="Starting angle of the reference spiral.",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.start_angle,
    end_angle: typing.Annotated[
        float,
        typer.Option(
            "--end-angle",
            "-e",
            help="Ending angle of the reference spiral (in radians).",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.end_angle,
    num_points: typing.Annotated[
        int,
        typer.Option(
            "--num-points",
            "-n",
            help="Number of points in the reference spiral.",
            rich_help_panel="Spiral Configuration",
        ),
    ] = config.SpiralConfig.num_points,
    verbosity: typing.Annotated[
        int,
        typer.Option(
            "--verbosity",
            "-v",
            count=True,
            help=(
                "Increase verbosity level. Can be repeated: "
                "0 (default), warnings/errors only; "
                "1 (-v), info messages; "
                "2 (-vv), debug messages."
            ),
        ),
    ] = 0,
    version: typing.Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version information and exit.",
        ),
    ] = False,
) -> None:
    """Extract features from spiral drawing data."""
    if verbosity > 0:
        config.set_verbosity_level(verbosity)

    logger.debug("Running Graphomotor pipeline with these arguments: %s", locals())

    config_params: dict[orchestrator.ConfigParams, float | int] = {
        "center_x": center_x,
        "center_y": center_y,
        "start_radius": start_radius,
        "growth_rate": growth_rate,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "num_points": num_points,
    }

    try:
        orchestrator.run_pipeline(
            input_path=input_path,
            output_path=output_path,
            feature_categories=typing.cast(
                list[orchestrator.FeatureCategories], features
            ),
            config_params=config_params,
        )
    except Exception as e:
        typer.secho(f"Error: {e}", fg="red", err=True)
        raise


if __name__ == "__main__":
    app()
