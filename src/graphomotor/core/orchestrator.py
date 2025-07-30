"""Runner for the Graphomotor pipeline."""

import datetime
import pathlib
import time
import typing

import numpy as np
import pandas as pd
import tqdm

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import center_spiral, generate_reference_spiral

logger = config.get_logger()

FeatureCategories = typing.Literal["duration", "velocity", "hausdorff", "AUC"]


def _validate_feature_categories(
    feature_categories: list[FeatureCategories],
) -> set[str]:
    """Validate requested feature categories and return valid ones.

    Args:
        feature_categories: List of feature categories to validate.

    Returns:
        Set of valid feature categories.

    Raises:
        ValueError: If no valid feature categories are provided.
    """
    feature_categories_set: set[str] = set(feature_categories)
    supported_categories_set = models.FeatureCategories.all()
    unknown_categories = feature_categories_set - supported_categories_set
    valid_requested_categories = feature_categories_set & supported_categories_set

    if unknown_categories:
        logger.warning(
            "Unknown feature categories requested, these categories will be ignored: "
            f"{unknown_categories}"
        )

    if not valid_requested_categories:
        error_msg = (
            "No valid feature categories provided. "
            f"Supported categories: {supported_categories_set}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return valid_requested_categories


def extract_features(
    spiral: models.Spiral,
    feature_categories: list[FeatureCategories],
    reference_spiral: np.ndarray,
) -> dict[str, str]:
    """Extract feature categories from spiral drawing data.

    This function chooses which feature categories to extract based on the provided
    sequence of valid category names and returns a dictionary containing the extracted
    features with metadata.

    Args:
        spiral: Spiral object containing drawing data and metadata.
        feature_categories: List of feature categories to extract. Valid options are:
            - "duration": Extract task duration.
            - "velocity": Extract velocity-based metrics.
            - "hausdorff": Extract Hausdorff distance metrics.
            - "AUC": Extract area under the curve metric.
        reference_spiral: Reference spiral for comparison.

    Returns:
        Dictionary containing the extracted features with metadata.
    """
    valid_categories = sorted(_validate_feature_categories(feature_categories))

    feature_extractors = models.FeatureCategories.get_extractors(
        spiral, reference_spiral
    )

    features: dict[str, float] = {}
    for category in valid_categories:
        logger.debug(f"Extracting {category} features")
        category_features = feature_extractors[category]()
        features.update(category_features)
        logger.debug(f"{category.capitalize()} features extracted")

    formatted_features = {k: f"{v:.15f}" for k, v in features.items()}

    formatted_features_with_metadata = {
        "source_file": str(spiral.metadata.get("source_path", "")),
        "participant_id": str(spiral.metadata.get("id", "")),
        "task": str(spiral.metadata.get("task", "")),
        "hand": str(spiral.metadata.get("hand", "")),
        "start_time": str(spiral.metadata.get("start_time", "")),
        **formatted_features,
    }

    return formatted_features_with_metadata


def export_features_to_csv(
    features_df: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    """Export extracted features to a CSV file.

    Args:
        features_df: DataFrame containing all metadata and features.
        output_path: Path to the output CSV file.
    """
    if not output_path.suffix:
        if not output_path.exists():
            logger.debug(f"Creating directory that doesn't exist: {output_path}")
            output_path.mkdir(parents=True)
        if features_df.shape[0] == 1:
            filename = (
                f"{features_df['participant_id'].iloc[0]}_"
                f"{features_df['task'].iloc[0]}_"
                f"{features_df['hand'].iloc[0]}_features_"
            )
        else:
            filename = "batch_features_"
        output_file = (
            output_path
            / f"{filename}{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
    else:
        parent_dir = output_path.parent
        if not parent_dir.exists():
            logger.debug(f"Creating parent directory that doesn't exist: {parent_dir}")
            parent_dir.mkdir(parents=True)
        output_file = output_path

    logger.debug(f"Saving extracted features to {output_file}")

    if output_file.exists():
        logger.debug(f"Overwriting existing file: {output_file}")

    try:
        features_df.to_csv(output_file)
        logger.debug(f"Features saved successfully to {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save features to {output_file}: {str(e)}")


def _run_file(
    input_path: pathlib.Path,
    feature_categories: list[FeatureCategories],
    spiral_config: config.SpiralConfig,
) -> dict[str, str]:
    """Process a single file for feature extraction.

    Args:
        input_path: Path to the input CSV file containing spiral drawing data.
        feature_categories: List of feature categories to extract.
        spiral_config: Configuration for spiral parameters.

    Returns:
        Dictionary containing the extracted features with metadata.
    """
    logger.debug(f"Processing file: {input_path}")
    spiral = reader.load_spiral(input_path)
    centered_spiral = center_spiral.center_spiral(spiral)
    reference_spiral = generate_reference_spiral.generate_reference_spiral(
        spiral_config
    )
    centered_reference_spiral = center_spiral.center_spiral(reference_spiral)

    return extract_features(
        centered_spiral, feature_categories, centered_reference_spiral
    )


def _run_directory(
    input_path: pathlib.Path,
    feature_categories: list[FeatureCategories],
    spiral_config: config.SpiralConfig,
) -> list[dict[str, str]]:
    """Process all CSV files in a directory and its subdirectories.

    Args:
        input_path: Path to the input directory containing CSV files.
        feature_categories: List of feature categories to extract.
        spiral_config: Configuration for spiral parameters.

    Returns:
        List of dictionaries, each containing extracted features with metadata
        for one processed file.

    Raises:
        ValueError: If no CSV files are found in the directory.
    """
    logger.debug(f"Processing directory: {input_path}")

    csv_files = list(input_path.rglob("*.csv"))

    if not csv_files:
        error_msg = f"No CSV files found in directory: {input_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Found {len(csv_files)} CSV files to process")

    results: list[dict[str, str]] = []
    failed_files: list[str] = []

    progress_bar = tqdm.tqdm(
        csv_files,
        desc="Processing files",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    for file_index, csv_file in enumerate(progress_bar, 1):
        try:
            progress_bar.set_postfix({"Current": csv_file.name})
            logger.debug(
                f"Processing file {csv_file.name} ({file_index}/{len(csv_files)})"
            )
            features = _run_file(csv_file, feature_categories, spiral_config)
            results.append(features)
            logger.debug(f"Successfully processed {csv_file.name}")
        except Exception as e:
            logger.warning(f"Failed to process {csv_file.name}: {str(e)}")
            failed_files.append(csv_file.name)
            continue

    if not results:
        error_msg = "Could not extract features from any file in the directory."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")

    return results


def run_pipeline(
    input_path: pathlib.Path | str,
    output_path: pathlib.Path | str | None = None,
    feature_categories: list[FeatureCategories] = [
        "duration",
        "velocity",
        "hausdorff",
        "AUC",
    ],
    config_params: dict[
        typing.Literal[
            "center_x",
            "center_y",
            "start_radius",
            "growth_rate",
            "start_angle",
            "end_angle",
            "num_points",
        ],
        float | int,
    ]
    | None = None,
) -> pd.DataFrame:
    """Run the Graphomotor pipeline to extract features from spiral drawing data.

    Supports both single-file and batch (directory) processing.

    Args:
        input_path: Path to a CSV file (single-file mode) or a directory containing CSV
            files (batch mode).
        output_path: Path to save extracted features. If specifying a file, the path
            must have a `.csv` extension.
            - If None, features are not saved.
            - If path has a CSV file extension, features are saved to that file.
            - If path is a directory, features are saved to a CSV file with a custom
              name and timestamp.
        feature_categories: List of feature categories to extract. Defaults to all
            available:
            - "duration": Task duration.
            - "velocity": Velocity-based metrics.
            - "hausdorff": Hausdorff distance metrics.
            - "AUC": Area under the curve metric.
        config_params: Dictionary of custom spiral configuration parameters for
            reference spiral generation and centering. If None, default configuration is
            used. Supported parameters are:
            - "center_x" (float): X-coordinate of the spiral center. Default is 50.
            - "center_y" (float): Y-coordinate of the spiral center. Default is 50.
            - "start_radius" (float): Starting radius of the spiral. Default is 0.
            - "growth_rate" (float): Growth rate of the spiral. Default is 1.075.
            - "start_angle" (float): Starting angle of the spiral. Default is 0.
            - "end_angle" (float): Ending angle of the spiral. Default is 8π.
            - "num_points" (int): Number of points in the spiral. Default is 10000.

    Returns:
        DataFrame containing the metadata and extracted features.

    Raises:
        ValueError: If the input path does not exist or is not a file/directory, if the
            output path does not have a .csv extension, or if no valid feature
            categories are provided.
    """
    start_time = time.time()

    logger.info("Starting Graphomotor pipeline")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Feature categories: {feature_categories}")

    input_path = pathlib.Path(input_path)

    if not input_path.exists() or (
        not input_path.is_file() and not input_path.is_dir()
    ):
        error_msg = (
            f"Input path does not exist or is not a file/directory: {input_path}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if output_path:
        output_path = pathlib.Path(output_path)
        if output_path.suffix and output_path.suffix.lower() != ".csv":
            error_msg = (
                f"Output file must have a .csv extension, got: {output_path.suffix}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    if config_params:
        logger.info(f"Custom spiral configuration: {config_params}")
        spiral_config = config.SpiralConfig.add_custom_params(
            typing.cast(dict, config_params)
        )
    else:
        spiral_config = config.SpiralConfig()

    if input_path.is_file():
        logger.info("Processing single file")
        features = [_run_file(input_path, feature_categories, spiral_config)]
        logger.info(
            "Single file processing complete, "
            f"successfully extracted {len(features[0]) - 5} features"
        )
    else:
        logger.info("Processing directory")
        features = _run_directory(input_path, feature_categories, spiral_config)
        logger.info(
            f"Batch processing complete, successfully processed {len(features)} files"
        )

    features_df = pd.DataFrame(features)
    features_df = features_df.set_index("source_file")

    if output_path:
        export_features_to_csv(features_df, output_path)

    logger.info(
        "Graphomotor pipeline completed successfully in "
        f"{time.time() - start_time:.2f} seconds"
    )

    return features_df
