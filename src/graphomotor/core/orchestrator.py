"""Runner for the Graphomotor pipeline."""

import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

from graphomotor.core import config, models
from graphomotor.io import reader
from graphomotor.utils import center_spiral, generate_reference_spiral

logger = config.get_logger()


def _ensure_path(path: pathlib.Path | str) -> pathlib.Path:
    """Ensure that the input is a Path object.

    Args:
        path: Input path, can be string or Path

    Returns:
        Path object
    """
    return pathlib.Path(path) if isinstance(path, str) else path


def _validate_feature_categories(feature_categories: list[str]) -> set[str]:
    """Validate requested feature categories and return valid ones.

    Args:
        feature_categories: List of feature categories to validate.

    Returns:
        Set of valid feature categories.

    Raises:
        ValueError: If no valid feature categories are provided.
    """
    unknown_categories = set(feature_categories) - config.FeatureCategories.all()
    valid_requested_categories = (
        set(feature_categories) & config.FeatureCategories.all()
    )

    if unknown_categories:
        logger.warning(
            "Unknown feature categories requested, these categories will be ignored: "
            f"{unknown_categories}"
        )

    if not valid_requested_categories:
        error_msg = (
            "No valid feature categories provided. "
            f"Supported categories: {config.FeatureCategories.all()}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return valid_requested_categories


def get_feature_categories(
    spiral: models.Spiral,
    reference_spiral: np.ndarray,
    feature_categories: list[str],
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

    feature_extractors = config.FeatureCategories.get_extractors(
        spiral, reference_spiral
    )

    features = {}
    for category in valid_categories:
        logger.debug(f"Extracting {category} features")
        category_features = feature_extractors[category]()
        features.update(category_features)
        logger.debug(f"{category.capitalize()} features extracted: {category_features}")

    return features


def export_features_to_csv(
    spiral: models.Spiral,
    features: dict[str, float],
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

    os.makedirs(output_path.parent, exist_ok=True)

    participant_id = spiral.metadata.get("id")
    task = spiral.metadata.get("task")
    hand = spiral.metadata.get("hand")

    filename = (
        f"{participant_id}_{task}_{hand}_features_"
        f"{datetime.today().strftime('%Y%m%d')}.csv"
    )

    if output_path.is_dir() or not output_path.suffix:
        output_file = output_path / filename
    else:
        output_file = output_path

    features_df = pd.DataFrame([features]).T

    features_df["participant_id"] = participant_id
    features_df["task"] = task
    features_df["hand"] = hand
    features_df["source_file"] = str(input_path)

    features_df.to_csv(output_file, index=False)
    logger.debug(f"Features saved successfully to {output_file}")


def extract_features(
    input_path: pathlib.Path | str,
    output_path: pathlib.Path | str | None,
    feature_categories: list[str],
    spiral_config: config.SpiralConfig | None,
) -> dict[str, float]:
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
    # logger.info(f"Starting feature extraction for {input_path}")
    # logger.info(f"Requested feature categories: {feature_categories}")

    logger.debug(f"Loading spiral data from {input_path}")
    input_path = _ensure_path(input_path)
    spiral = reader.load_spiral(input_path)
    spiral = center_spiral.center_spiral(spiral)

    logger.debug("Generating reference spiral to calculate features")
    config_to_use = spiral_config or config.SpiralConfig()
    reference_spiral = generate_reference_spiral.generate_reference_spiral(
        config=config_to_use
    )

    features = get_feature_categories(spiral, reference_spiral, feature_categories)
    logger.info(f"Feature extraction complete. Extracted {len(features)} features")

    if output_path:
        output_path = _ensure_path(output_path)
        export_features_to_csv(spiral, features, input_path, output_path)

    return features


def run_pipeline(
    input_path: pathlib.Path | str,
    output_path: pathlib.Path | str | None,
    feature_categories: list[str],
    config_params: dict[str, float | int] | None = None,
) -> dict[str, float]:
    """Run the Graphomotor pipeline to extract features from spiral drawing data.

    Args:
        input_path: Path to the input CSV file containing spiral drawing data.
        output_path: Path to save the extracted features. If None, features are not
            saved.
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
        spiral_config = config.SpiralConfig.from_dict(config_params)

    features = extract_features(
        input_path, output_path, feature_categories, spiral_config
    )

    logger.info("Graphomotor pipeline completed successfully")
    return features
