"""Core functionality for graphomotor data processing and analysis.

This module provides the essential infrastructure for the graphomotor toolkit:

## Data Models and Validation

- **`Spiral`**: Pydantic model for spiral drawing data validation with strict
  metadata requirements (7-digit participant IDs starting with '5', Dom/NonDom
  hand designation, spiral_trace/spiral_recall tasks numbered 1-5)
- **`FeatureCategories`**: Registry of all 4 feature extractors (duration,
  velocity, hausdorff, AUC) with dynamic extractor mapping

## Configuration and Logging

- **`SpiralConfig`**: Immutable dataclass for Archimedean spiral parameters
  (center coordinates, growth rate, angles, point density) with custom
  parameter override support
- **Logger**: Centralized logging with configurable verbosity levels
  (WARNING/INFO/DEBUG) and structured formatting

## Pipeline Orchestration

- **`run_pipeline()`**: Main entry point supporting both single-file and
  batch directory processing with progress tracking, error handling, and
  configurable feature extraction
- **Feature Extraction**: Coordinates spiral centering, reference generation,
  and execution of all feature extractors
- **Export System**: Automated CSV export with metadata preservation and
  timestamp-based naming

## Command-Line Interface

- **`extract`**: CLI command for feature extraction with extensive configuration
  options and help documentation
- **`plot-features`**: CLI command for generating publication-ready visualizations
  from extracted feature datasets

The core module serves as the foundation for all graphomotor operations, ensuring
data quality, processing consistency, and user-friendly interfaces.
"""
