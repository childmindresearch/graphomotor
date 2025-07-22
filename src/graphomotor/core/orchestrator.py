"""Runner for the Graphomotor pipeline."""

import datetime
import pathlib
import typing

import numpy as np
import pandas as pd

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import center_spiral, generate_reference_spiral

logger = config.get_logger()

FeatureCategories = typing.Literal["duration", "velocity", "hausdorff", "AUC"]


def _ensure_path(path: pathlib.Path | str) -> pathlib.Path:
    """Ensure that the input is a Path object.

    Args:
        path: Input path, can be string or Path

    Returns:
        Path object
    """
    return pathlib.Path(path) if isinstance(path, str) else path


def _validate_output_path(output_path: pathlib.Path) -> None:
    """Validate that the output path has a .csv extension if it's a file.

    Args:
        output_path: Path to validate

    Raises:
        ValueError: If the output path has an extension but it's not .csv
    """
    if output_path.suffix and output_path.suffix.lower() != ".csv":
        error_msg = f"Output file must have a .csv extension, got: {output_path.suffix}"
        logger.error(error_msg)
        raise ValueError(error_msg)


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


def _export_features_to_csv(
    spiral: models.Spiral,
    features: dict[str, str],
    output_path: pathlib.Path,
) -> None:
    """Export extracted features to a CSV file.

    Args:
        spiral: The spiral data used for feature extraction.
        features: Dictionary containing the extracted features.
        output_path: Path to the output CSV file.
    """
    logger.info(f"Saving extracted features to {output_path}")

    source_path = spiral.metadata.get("source_path")
    participant_id = spiral.metadata.get("id")
    task = spiral.metadata.get("task")
    hand = spiral.metadata.get("hand")

    if not output_path.suffix:
        if not output_path.exists():
            logger.info(f"Creating directory that doesn't exist: {output_path}")
            output_path.mkdir(parents=True)
        filename = (
            f"{participant_id}_{task}_{hand}_features_"
            f"{datetime.datetime.today().strftime('%Y%m%d')}.csv"
        )
        output_file = output_path / filename
    else:
        parent_dir = output_path.parent
        if not parent_dir.exists():
            logger.info(f"Creating parent directory that doesn't exist: {parent_dir}")
            parent_dir.mkdir(parents=True)
        output_file = output_path

    if output_file.exists():
        logger.info(f"Overwriting existing file: {output_file}")

    metadata = {
        "source_file": source_path,
        "participant_id": participant_id,
        "task": task,
        "hand": hand,
    }

    features_df = pd.DataFrame(
        {
            "variable": list(metadata.keys()) + list(features.keys()),
            "value": list(metadata.values()) + list(features.values()),
        }
    )

    try:
        features_df.to_csv(output_file, index=False, header=False)
        logger.info(f"Features saved successfully to {output_file}")
    except Exception as e:
        # Allowed to pass in Jupyter Notebook scenarios.
        logger.error(f"Failed to save features to {output_file}: {str(e)}")


def _export_directory_features_to_csv(
    results_df: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    """Export batch processing results to a single CSV file.

    Args:
        results_df: DataFrame containing all features from batch processing.
        output_path: Path to the output CSV file.
    """
    logger.info(f"Saving batch features to single CSV file: {output_path}")

    parent_dir = output_path.parent
    if not parent_dir.exists():
        logger.info(f"Creating parent directory that doesn't exist: {parent_dir}")
        parent_dir.mkdir(parents=True)

    if output_path.exists():
        logger.info(f"Overwriting existing file: {output_path}")

    try:
        results_df.to_csv(output_path, index=True)
        logger.info(f"Batch features saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save batch features to {output_path}: {str(e)}")


def _load_and_center_spiral(input_path: pathlib.Path) -> models.Spiral:
    """Load spiral from file and center it.

    Args:
        input_path: Path to the input CSV file containing spiral drawing data.

    Returns:
        Centered spiral object.
    """
    spiral = reader.load_spiral(input_path)
    return center_spiral.center_spiral(spiral)


def _generate_reference_spiral(spiral_config: config.SpiralConfig | None) -> np.ndarray:
    """Generate and center a reference spiral for feature extraction.

    Args:
        spiral_config: Configuration for spiral parameters. If None, uses default.

    Returns:
        Centered reference spiral.
    """
    config_to_use = spiral_config or config.SpiralConfig()
    reference_spiral = generate_reference_spiral.generate_reference_spiral(
        spiral_config=config_to_use
    )
    return center_spiral.center_spiral(reference_spiral)


def extract_features(
    spiral: models.Spiral,
    output_path: pathlib.Path | None,
    feature_categories: list[FeatureCategories],
    reference_spiral: np.ndarray,
) -> dict[str, str]:
    """Extract feature categories from spiral drawing data.

    This function chooses which feature categories to extract based on the provided
    sequence of valid category names, exports the features to the specified output path
    if provided, and returns a dictionary containing the extracted features.

    Args:
        spiral: Spiral object containing drawing data and metadata.
        output_path: Path to the output directory for saving extracted features. If
            None, features are not saved.
        feature_categories: List of feature categories to extract. Valid options are:
            - "duration": Extract task duration.
            - "velocity": Extract velocity-based metrics.
            - "hausdorff": Extract Hausdorff distance metrics.
            - "AUC": Extract area under the curve metric.
        reference_spiral: Reference spiral for comparison.

    Returns:
        Dictionary containing the extracted features.
    """
    valid_categories = _validate_feature_categories(feature_categories)

    feature_extractors = models.FeatureCategories.get_extractors(
        spiral, reference_spiral
    )

    features = {}
    for category in valid_categories:
        logger.debug(f"Extracting {category} features")
        category_features = feature_extractors[category]()
        features.update(category_features)
        logger.debug(f"{category.capitalize()} features extracted: {category_features}")

    logger.info(f"Feature extraction complete. Extracted {len(features)} features")

    formatted_features = {k: f"{v:.15f}" for k, v in features.items()}

    if output_path:
        _export_features_to_csv(spiral, formatted_features, output_path)

    return formatted_features


def _run_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None,
    feature_categories: list[FeatureCategories],
    spiral_config: config.SpiralConfig | None,
) -> dict[str, str]:
    """Process a single file for feature extraction.

    Args:
        input_path: Path to the input CSV file containing spiral drawing data.
        output_path: Path to the output directory for saving extracted features.
        feature_categories: List of feature categories to extract.
        spiral_config: Configuration for spiral parameters.

    Returns:
        Dictionary containing the extracted features.
    """
    logger.info(f"Processing file: {input_path}")
    spiral = _load_and_center_spiral(input_path)
    reference_spiral = _generate_reference_spiral(spiral_config)

    return extract_features(spiral, output_path, feature_categories, reference_spiral)


def _run_directory(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None,
    feature_categories: list[FeatureCategories],
    spiral_config: config.SpiralConfig | None,
) -> pd.DataFrame:
    """Process all CSV files in a directory and its subdirectories.

    Args:
        input_path: Path to the input directory containing CSV files.
        output_path: Path to the output directory for saving extracted features. If
            output_path has a .csv extension, all features will be saved to that single
            CSV file instead of individual files.
        feature_categories: List of feature categories to extract.
        spiral_config: Configuration for spiral parameters.

    Returns:
        DataFrame with source_file as index and columns: participant_id, task, hand,
        followed by calculated features.

    Raises:
        ValueError: If no CSV files are found in the directory.
    """
    logger.info(f"Processing directory: {input_path}")

    csv_files = list(input_path.rglob("*.csv"))

    if not csv_files:
        error_msg = f"No CSV files found in directory: {input_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Found {len(csv_files)} CSV files to process")

    reference_spiral = _generate_reference_spiral(spiral_config)
    logger.debug("Reference spiral generated for batch processing")

    save_to_single_csv = output_path and output_path.suffix.lower() == ".csv"
    individual_output_path = None if save_to_single_csv else output_path

    results = []
    failed_files = []

    for file_index, csv_file in enumerate(csv_files, 1):
        try:
            logger.info(
                f"Processing file {csv_file.name} ({file_index}/{len(csv_files)})"
            )

            spiral = _load_and_center_spiral(csv_file)

            features = extract_features(
                spiral,
                individual_output_path,
                feature_categories,
                reference_spiral,
            )

            row_data = {
                "source_file": spiral.metadata.get("source_path"),
                "participant_id": spiral.metadata.get("id"),
                "task": spiral.metadata.get("task"),
                "hand": spiral.metadata.get("hand"),
                **features,
            }

            results.append(row_data)
            logger.info(f"Successfully processed {csv_file.name}")
        except Exception as e:
            logger.error(f"Failed to process {csv_file.name}: {str(e)}")
            failed_files.append(csv_file.name)
            continue

    if failed_files:
        failed_files_str = "\n".join(failed_files)
        logger.warning(
            f"Failed to process {len(failed_files)} files:\n{failed_files_str}"
        )

    logger.info(
        f"Batch processing complete. Successfully processed {len(results)} files"
    )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.set_index("source_file")

    if save_to_single_csv:
        assert output_path is not None
        _export_directory_features_to_csv(df, output_path)

    return df


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
) -> dict[str, str] | pd.DataFrame:
    """Run the Graphomotor pipeline to extract features from spiral drawing data.

    Supports both single-file and batch (directory) processing.

    Args:
        input_path: Path to a CSV file (single-file mode) or a directory containing CSV
            files (batch mode).
        output_path: Path to save extracted features.
            - If None, features are not saved.
            - If path has a file extension, features are saved to that file.
            - If path is a directory, output files are created per
              participant/task/hand/date.
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
        If input_path is a file: Dictionary of extracted features.
        If input_path is a directory: DataFrame indexed by source file path, with
            columns for participant_id, task, hand, and extracted features.

    Raises:
        ValueError: If input_path does not exist, is not a file/directory, or if no
            valid feature categories are provided.
    """
    logger.info("Starting Graphomotor pipeline")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Feature categories: {feature_categories}")

    input_path = _ensure_path(input_path)

    output_path_validated: pathlib.Path | None = None
    if output_path:
        output_path_validated = _ensure_path(output_path)
        _validate_output_path(output_path_validated)

    spiral_config = None
    if config_params:
        logger.info(f"Custom spiral configuration: {config_params}")
        spiral_config = config.SpiralConfig.add_custom_params(
            typing.cast(dict, config_params)
        )

    features: dict[str, str] | pd.DataFrame
    if input_path.is_file():
        logger.info("Processing single file")
        features = _run_file(
            input_path, output_path_validated, feature_categories, spiral_config
        )
    elif input_path.is_dir():
        logger.info("Processing directory")
        features = _run_directory(
            input_path, output_path_validated, feature_categories, spiral_config
        )
    else:
        error_msg = (
            f"Input path does not exist or is not a file/directory: {input_path}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Graphomotor pipeline completed successfully")
    return features
