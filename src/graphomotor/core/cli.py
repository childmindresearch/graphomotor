"""Command-line interface for graphomotor."""

import enum
import pathlib
import typing

import typer

from graphomotor.core import config, orchestrator
from graphomotor.plot import feature_plots

logger = config.get_logger()
app = typer.Typer(
    name="graphomotor",
    help=(
        "Graphomotor: A Python toolkit for analyzing graphomotor data "
        "collected via Curious. See the README for usage details."
    ),
    epilog=(
        "Please report issues at "
        "https://github.com/childmindresearch/graphomotor/issues."
    ),
)


class ValidFeatureCategories(str, enum.Enum):
    """Valid feature categories for extraction."""

    DURATION = "duration"
    VELOCITY = "velocity"
    HAUSDORFF = "hausdorff"
    AUC = "AUC"


class ValidFeaturePlotTypes(str, enum.Enum):
    """Valid plot types for feature visualization."""

    DIST = "dist"
    TRENDS = "trends"
    BOXPLOT = "boxplot"
    CLUSTER = "cluster"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbosity: typing.Annotated[
        int,
        typer.Option(
            "--verbosity",
            "-v",
            count=True,
            show_default=False,
            help=(
                "Increase logging verbosity by counting "
                "the number of times the flag is used. "
                "Default: warnings/errors only. "
                "-v: info level. "
                "-vv: debug level."
            ),
        ),
    ] = 0,
    version: typing.Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            is_eager=True,
            help="Show version information and exit.",
        ),
    ] = False,
) -> None:
    """Main entry point for the Graphomotor CLI."""
    if version:
        typer.echo(f"Graphomotor version: {config.get_version()}")
        raise typer.Exit()

    if verbosity > 0:
        config.set_verbosity_level(verbosity)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command(
    name="extract",
    help=(
        "Extract features from spiral drawing data. "
        "Supports both single-file and batch (directory) processing."
    ),
    epilog=(
        "For more information on data format requirements, see the README at "
        "https://github.com/childmindresearch/graphomotor/blob/main/README.md."
    ),
)
def extract(
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
        list[ValidFeatureCategories] | None,
        typer.Option(
            "--features",
            "-f",
            help=(
                "Feature categories to extract. "
                "If omitted, all available features are extracted. "
                "To input multiple feature categories, "
                "specify this option multiple times."
            ),
            show_default=False,
            rich_help_panel="Feature Category Options",
        ),
    ] = None,
    center_x: typing.Annotated[
        float,
        typer.Option(
            "--center-x",
            "-x",
            help="X-coordinate of the reference spiral center.",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.center_x,
    center_y: typing.Annotated[
        float,
        typer.Option(
            "--center-y",
            "-y",
            help="Y-coordinate of the reference spiral center.",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.center_y,
    start_radius: typing.Annotated[
        float,
        typer.Option(
            "--start-radius",
            "-r",
            help="Starting radius of the reference spiral.",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.start_radius,
    growth_rate: typing.Annotated[
        float,
        typer.Option(
            "--growth-rate",
            "-g",
            help="Growth rate of the reference spiral.",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.growth_rate,
    start_angle: typing.Annotated[
        float,
        typer.Option(
            "--start-angle",
            "-s",
            help="Starting angle of the reference spiral.",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.start_angle,
    end_angle: typing.Annotated[
        float,
        typer.Option(
            "--end-angle",
            "-e",
            help="Ending angle of the reference spiral (in radians).",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.end_angle,
    num_points: typing.Annotated[
        int,
        typer.Option(
            "--num-points",
            "-n",
            help="Number of points in the reference spiral.",
            rich_help_panel="Spiral Configuration Options",
        ),
    ] = config.SpiralConfig.num_points,
) -> None:
    """Extract features from spiral drawing data."""
    logger.debug(f"Running Graphomotor pipeline with these arguments: {locals()}")

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


@app.command(
    name="plot-features",
    help=(
        "Generate plots from extracted features. Supports distribution, trend, box, "
        "and cluster plots."
    ),
    epilog=(
        "For more information, see the README at "
        "https://github.com/childmindresearch/graphomotor/blob/main/README.md."
    ),
)
def plot_features(
    input_path: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to a CSV file containing extracted features.",
        ),
    ],
    output_path: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            help="Output path to a directory where plots will be saved.",
        ),
    ],
    plot_types: typing.Annotated[
        list[ValidFeaturePlotTypes] | None,
        typer.Option(
            "--plot-types",
            "-p",
            help=(
                "Types of plots to generate. If omitted, all plot types are generated. "
                "To specify multiple types, use this option multiple times."
            ),
            show_default=False,
        ),
    ] = None,
    features: typing.Annotated[
        list[str] | None,
        typer.Option(
            "--features",
            "-f",
            help=(
                "Specific features to plot. If omitted, all features are plotted. "
                "To specify multiple features, use this option multiple times."
            ),
            show_default=False,
        ),
    ] = None,
) -> None:
    """Generate plots from extracted features."""
    logger.debug(f"Generating feature plots with arguments: {locals()}")

    if not input_path.is_file() or input_path.suffix.lower() != ".csv":
        typer.secho(
            f"Error: Input path {input_path} must be an existing CSV file",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        typer.secho(
            f"Error creating output directory {output_path}: {e}",
            fg="red",
            err=True,
        )
        raise

    if plot_types is None:
        plot_types = [
            ValidFeaturePlotTypes.DIST,
            ValidFeaturePlotTypes.TRENDS,
            ValidFeaturePlotTypes.BOXPLOT,
            ValidFeaturePlotTypes.CLUSTER,
        ]

    try:
        plot_functions = {
            ValidFeaturePlotTypes.DIST: feature_plots.plot_feature_distributions,
            ValidFeaturePlotTypes.TRENDS: feature_plots.plot_feature_trends,
            ValidFeaturePlotTypes.BOXPLOT: feature_plots.plot_feature_boxplots,
            ValidFeaturePlotTypes.CLUSTER: feature_plots.plot_feature_clusters,
        }

        for plot_type in plot_types:
            plot_functions[plot_type](
                data=input_path, output_path=output_path, features=features
            )
            typer.secho(f"Generated {plot_type.value} plot successfully", fg="green")

        typer.secho(f"All plots saved to: {output_path}", fg="green")

    except Exception as e:
        typer.secho(f"Error generating plots: {e}", fg="red", err=True)
        raise


if __name__ == "__main__":
    app()
