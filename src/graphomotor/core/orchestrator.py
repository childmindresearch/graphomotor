"""Runner for the Graphomotor pipeline."""

import os
import pathlib
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import center_spiral, generate_reference_spiral

logger = config.get_logger()

FeatureCategories = Literal["duration", "velocity", "hausdorff", "AUC"]


def _ensure_path(path: pathlib.Path | str) -> pathlib.Path:
    """Ensure that the input is a Path object.

    Args:
        path: Input path, can be string or Path

    Returns:
        Path object
    """
    return pathlib.Path(path) if isinstance(path, str) else path


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


def _get_feature_categories(
    spiral: models.Spiral,
    reference_spiral: np.ndarray,
    feature_categories: list[FeatureCategories],
) -> dict[str, float]:
    """Feature categories dispatcher.

    This function chooses which feature categories to extract based on the provided
    sequence of valid category names and returns a dictionary containing the extracted
    features.

    Args:
        spiral: The spiral data to extract features from.
        reference_spiral: The reference spiral used for calculating features.
        feature_categories: List of feature categories to extract.

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

    return features


def _export_features_to_csv(
    spiral: models.Spiral,
    features: dict[str, str],
    input_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Export extracted features to a CSV file.

    Args:
        spiral: The spiral data used for feature extraction.
        features: Dictionary containing the extracted features.
        input_path: Path to the input CSV file.
        output_path: Path to the output CSV file.
    """
    logger.info(f"Saving extracted features to {output_path}")

    participant_id = spiral.metadata.get("id")
    task = spiral.metadata.get("task")
    hand = spiral.metadata.get("hand")

    filename = (
        f"{participant_id}_{task}_{hand}_features_"
        f"{datetime.today().strftime('%Y%m%d')}.csv"
    )

    if not output_path.suffix:
        if not os.path.exists(output_path):
            logger.info(f"Creating directory that doesn't exist: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        output_file = output_path / filename
    else:
        parent_dir = output_path.parent
        if not os.path.exists(parent_dir):
            logger.info(f"Creating parent directory that doesn't exist: {parent_dir}")
        os.makedirs(parent_dir, exist_ok=True)
        output_file = output_path

    if os.path.exists(output_file):
        logger.info(f"Overwriting existing file: {output_file}")

    metadata = {
        "participant_id": participant_id,
        "task": task,
        "hand": hand,
        "source_file": str(input_path),
    }

    features_df = pd.DataFrame(
        {
            "variable": list(metadata.keys()) + list(features.keys()),
            "value": list(metadata.values()) + list(features.values()),
        }
    )

    try:
        features_df.to_csv(output_file, index=False, header=False)
        logger.debug(f"Features saved successfully to {output_file}")
    except Exception as e:
        # Allowed to pass in Jupyter Notebook scenarios.
        logger.error(f"Failed to save features to {output_file}: {str(e)}")


def extract_features(
    input_path: pathlib.Path | str,
    output_path: pathlib.Path | str | None,
    feature_categories: list[FeatureCategories],
    spiral_config: config.SpiralConfig | None,
) -> dict[str, str]:
    """Extract features from spiral drawing data.

    Args:
        input_path: Path to the input CSV file containing spiral drawing data.
        output_path: Path to the output directory for saving extracted features. If
            None, features are not saved.
        feature_categories: List of feature categories to extract. Valid options are:
            - "duration": Extract task duration.
            - "velocity": Extract velocity-based metrics.
            - "hausdorff": Extract Hausdorff distance metrics.
            - "AUC": Extract area under the curve metric.
        spiral_config: Optional configuration for spiral parameters. If None, default
            parameters are used.

    Returns:
        Dictionary containing the extracted features.
    """
    logger.debug(f"Loading spiral data from {input_path}")
    input_path = _ensure_path(input_path)
    spiral = reader.load_spiral(input_path)
    centered_spiral = center_spiral.center_spiral(spiral)

    logger.debug("Generating reference spiral to calculate features")
    config_to_use = spiral_config or config.SpiralConfig()
    reference_spiral = generate_reference_spiral.generate_reference_spiral(
        config=config_to_use
    )
    centered_reference_spiral = center_spiral.center_spiral(reference_spiral)

    features = _get_feature_categories(
        centered_spiral, centered_reference_spiral, feature_categories
    )
    logger.info(f"Feature extraction complete. Extracted {len(features)} features")

    formatted_features = {k: f"{v:.15f}" for k, v in features.items()}

    if output_path:
        output_path = _ensure_path(output_path)
        _export_features_to_csv(spiral, formatted_features, input_path, output_path)

    return formatted_features


def run_pipeline(
    input_path: pathlib.Path | str,
    output_path: pathlib.Path | str | None,
    feature_categories: list[FeatureCategories],
    config_params: dict[str, float | int] | None = None,
) -> dict[str, str]:
    """Run the Graphomotor pipeline to extract features from spiral drawing data.

    Args:
        input_path: Path to the input CSV file containing spiral drawing data.
        output_path: Path where extracted features will be saved. Three behaviors are
            possible:
            - If None: Features are calculated but not saved to disk.
            - If path includes file extension (e.g., "/path/to/output.csv"): This exact
                file path is used.
            - If path has no file extension (e.g., "/path/to/output"): A file is
                created in that directory with auto-generated filename:
                "/path/to/output/{participant_id}_{task}_{hand}_features_{date}.csv".
        feature_categories: List of feature categories to extract. Options are:
            - "duration": Extract task duration.
            - "velocity": Extract velocity-based metrics.
            - "hausdorff": Extract Hausdorff distance metrics.
            - "AUC": Extract area under the curve metric.
        config_params: Optional configuration parameters for spiral drawing. If None,
            default parameters are used.

    Returns:
        Dictionary containing the extracted features.
    """
    logger.info("Starting Graphomotor pipeline")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Feature categories: {feature_categories}")

    spiral_config = None
    if config_params:
        logger.info(f"Custom spiral configuration: {config_params}")
        spiral_config = config.SpiralConfig.add_custom_params(config_params)

    features = extract_features(
        input_path, output_path, feature_categories, spiral_config
    )

    logger.info("Graphomotor pipeline completed successfully")
    return features
