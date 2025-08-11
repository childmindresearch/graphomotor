"""Command-line interface for the Graphomotor."""

import enum
import pathlib
import typing

import typer

from graphomotor.core import config, orchestrator

logger = config.get_logger()
app = typer.Typer(
    name="graphomotor",
    help=(
        "Graphomotor: A Python toolkit for analyzing graphomotor data "
        "collected via Curious. See the README for usage details."
    ),
    epilog=(
        "Please report issues at "
        "https://github.com/childmindresearch/graphomotor/issues"
    ),
)


class ValidFeatureCategories(str, enum.Enum):
    """Valid feature categories for extraction."""

    DURATION = "duration"
    VELOCITY = "velocity"
    HAUSDORFF = "hausdorff"
    AUC = "AUC"


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
        "https://github.com/childmindresearch/graphomotor/blob/main/README.md"
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
        typing.Optional[list[ValidFeatureCategories]],
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


@app.command(
    name="plot-spiral",
    help=(
        "Plot spiral drawing data for quality control visualization. "
        "Supports both single-file and batch (directory) processing with grouping options."
    ),
    epilog=(
        "Examples:\n"
        "  graphomotor plot-spiral data.csv --output spiral.png --include-reference\n"
        "  graphomotor plot-spiral data/ --output plots/ --group-by participant\n"
        "  graphomotor plot-spiral data/ --output plots/ --color-by-segment --max-per-row 3"
    ),
)
def plot_spiral(
    input_path: typing.Annotated[
        pathlib.Path,
        typer.Argument(
            help=(
                "Path to a spiral CSV file or directory containing CSV files "
                "with Curious drawing data."
            ),
        ),
    ],
    output: typing.Annotated[
        typing.Optional[pathlib.Path],
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output path for plots. For single files, can be a specific filename. "
                "For directories, should be the output directory path. "
                "Auto-generates if not specified."
            ),
            rich_help_panel="Output Options",
        ),
    ] = None,
    group_by: typing.Annotated[
        typing.Optional[str],
        typer.Option(
            "--group-by",
            help=(
                "Group spirals in batch processing. Options: 'participant', 'task', 'hand'. "
                "If None, creates individual plots for each spiral."
            ),
            rich_help_panel="Grouping Options",
        ),
    ] = None,
    max_per_row: typing.Annotated[
        int,
        typer.Option(
            "--max-per-row",
            help="Maximum number of subplots per row when using group-by option.",
            rich_help_panel="Grouping Options",
        ),
    ] = 4,
    include_reference: typing.Annotated[
        bool,
        typer.Option(
            "--include-reference/--no-reference",
            help="Include reference spiral overlay for comparison.",
            rich_help_panel="Visualization Options",
        ),
    ] = False,
    color_by_segment: typing.Annotated[
        bool,
        typer.Option(
            "--color-by-segment/--single-color",
            help="Use different colors for line segments vs single color.",
            rich_help_panel="Visualization Options",
        ),
    ] = False,
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
    """Plot spiral drawing data for quality control visualization."""
    from graphomotor.plot import spiral_plots
    
    logger.debug("Running spiral plotting with these arguments: %s", locals())
    
    # Validate group_by parameter
    valid_group_by = {'participant', 'task', 'hand', None}
    if group_by not in valid_group_by:
        typer.secho(
            f"Error: group-by must be one of {valid_group_by}, got: {group_by}",
            fg="red",
            err=True,
        )
        raise typer.Exit(1)
    
    # Create spiral configuration if reference is requested
    spiral_config = None
    if include_reference:
        spiral_config = config.SpiralConfig.add_custom_params({
            "center_x": center_x,
            "center_y": center_y,
            "start_radius": start_radius,
            "growth_rate": growth_rate,
            "start_angle": start_angle,
            "end_angle": end_angle,
            "num_points": num_points,
        })
    
    try:
        if input_path.is_file():
            # Single file plotting
            spiral_plots.plot_single_spiral(
                input_path=input_path,
                output_path=output,
                include_reference=include_reference,
                color_by_segment=color_by_segment,
                spiral_config=spiral_config,
            )
            typer.secho(f"✓ Successfully created plot for {input_path}", fg="green")
        elif input_path.is_dir():
            # Batch plotting
            if output is None:
                output = input_path / "plots"
            
            spiral_plots.plot_spiral_batch(
                input_path=input_path,
                output_path=output,
                group_by=group_by,
                max_per_row=max_per_row,
                include_reference=include_reference,
                color_by_segment=color_by_segment,
                spiral_config=spiral_config,
            )
            typer.secho(f"✓ Successfully created plots in {output}", fg="green")
        else:
            typer.secho(f"Error: Input path does not exist: {input_path}", fg="red", err=True)
            raise typer.Exit(1)
            
    except Exception as e:
        typer.secho(f"Error during plotting: {e}", fg="red", err=True)
        raise


if __name__ == "__main__":
    app()
