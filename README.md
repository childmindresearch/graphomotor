# Graphomotor Study Toolkit

A Python toolkit for analysis of graphomotor data collected via Curious.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15800191.svg)](https://doi.org/10.5281/zenodo.15800191)

[![Build](https://github.com/childmindresearch/graphomotor/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/graphomotor/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/graphomotor/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/graphomotor)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/graphomotor/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/graphomotor)

Welcome to `graphomotor`, a specialized Python library for analyzing graphomotor data collected via [Curious](https://www.gettingcurious.com/). This toolkit aims to provide comprehensive tools for processing, analyzing, and visualizing data from various graphomotor assessment tasks, including spiral drawing, trails making, alphabetic writing, digit symbol substitution, and the Rey-Osterrieth Complex Figure Test.

> [!IMPORTANT]
> This package is under active development. Currently, the focus is on the spiral drawing task. After finalizing feature extraction, the next steps will involve implementing both preprocessing and visualization for this task. Once these parts are in place, we plan to extend support to other tasks.

## Feature Extraction Capabilities

The toolkit extracts clinically relevant metrics from digitized drawing data. Currently implemented features include:

- **Temporal Features**: Task completion duration.
- **Velocity Features**: Velocity analysis including linear, radial, and angular velocity components with statistical measures (sum, median, variation, skewness, kurtosis).
- **Distance Features**: Spatial accuracy measurements using Hausdorff distance metrics with temporal normalizations and segment-specific analysis.
- **Drawing Error Features**: Area under the curve (AUC) calculations between drawn paths and ideal reference trajectories to quantify spatial accuracy.

## Installation

Install the graphomotor package from PyPI:

```sh
pip install graphomotor
```

Or install the latest development version directly from GitHub:

```sh
pip install git+https://github.com/childmindresearch/graphomotor
```

## Quick Start

> [!CAUTION]
> This library requires input data to adhere to a specific format matching the standard output from [Curious drawing responses](https://mindlogger.atlassian.net/servicedesk/customer/portal/3/article/859242501). See more details in the [Data Format Requirements](#data-format-requirements) section below.

### Extracting Features from Spiral Drawing Data

#### Single File Processing

```python
from graphomotor.core import orchestrator

# Path to your spiral drawing data file
input_file = "path/to/your/spiral_data.csv"

# Option 1: Process file without saving any CSV file
# Only return the DataFrame with extracted features
features_df = orchestrator.run_pipeline(
    input_path=input_file
)

# Features are returned as a pandas DataFrame with source file as index
print(f"Extracted features: {list(features_df.columns)}")

# Access the single file's data (features_df has one row)
file_path = features_df.index[0]
print(f"File: {file_path}")
print(f"Participant: {features_df.loc[file_path, 'participant_id']}")
print(f"Task: {features_df.loc[file_path, 'task']}")
print(f"Hand: {features_df.loc[file_path, 'hand']}")
print(f"Duration: {features_df.loc[file_path, 'duration']}")
```

```python
# Option 2: Save to a directory with auto-generated filename
# Creates a CSV file with auto-generated name in the specified directory
# Format: {participant_id}_{task}_{hand}_features_{YYYYMMDD_HHMM}.csv
features_df = orchestrator.run_pipeline(
    input_path=input_file,
    output_path="path/to/output/directory"
)
```

```python
# Option 3: Save to a specific CSV file
# Features will be saved to the specified file path
features_df = orchestrator.run_pipeline(
    input_path=input_file,
    output_path="path/to/features.csv"
)
```

#### Batch Processing

```python
from graphomotor.core import orchestrator

# Path to directory containing multiple spiral drawing data files
input_dir = "path/to/your/spiral_data_directory"

# Option 1: Process files without saving any CSV files
# Only return the DataFrame with extracted features
features_df = orchestrator.run_pipeline(
    input_path=input_dir,
)

# Features are returned as a pandas DataFrame with source files as index
# Columns include: participant_id, task, hand, start_time, and calculated features
print(f"Successfully processed {len(features_df)} files")

# Access metadata and features for a specific file
for file_path in features_df.index:
    print(f"File: {file_path}")
    print(f"Participant: {features_df.loc[file_path, 'participant_id']}")
    print(f"Task: {features_df.loc[file_path, 'task']}")
    print(f"Hand: {features_df.loc[file_path, 'hand']}")
    print(f"Duration: {features_df.loc[file_path, 'duration']}")

```

```python
# Option 2: Save to a directory with auto-generated filename
# Creates a single consolidated CSV file with auto-generated name
# Format: batch_features_{YYYYMMDD_HHMM}.csv
features_df = orchestrator.run_pipeline(
    input_path=input_dir,
    output_path="path/to/output/directory"
)
```

```python
# Option 3: Save to a specific CSV file (single consolidated file)
# All features will be written to one specified file
features_df = orchestrator.run_pipeline(
    input_path=input_dir,
    output_path="path/to/consolidated_features.csv"
)
```

> [!NOTE]
> Currently, `graphomotor` is available as an importable Python library. CLI functionality is planned for future releases.

For detailed configuration options and additional parameters, refer to the [`run_pipeline` documentation](https://childmindresearch.github.io/graphomotor/graphomotor/core/orchestrator.html#run_pipeline).

## Development Progress

| Task | Preprocessing | Feature Extraction | Visualization |
| :--- | :---: | :---: | :---: |
| Spiral | ![Spiral: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Spiral: Feature Extraction In Progress](https://img.shields.io/badge/in_progress-yellow) | ![Spiral: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Rey-Osterrieth Complex Figure | ![Rey-Osterrieth: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Rey-Osterrieth: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Rey-Osterrieth: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Alphabetic Writing | ![Alphabetic Writing: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Alphabetic Writing: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Alphabetic Writing: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Digit Symbol Substitution | ![Digit Symbol Substitution: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Digit Symbol Substitution: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Digit Symbol Substitution: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Trails Making | ![Trails Making: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Trails Making: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Trails Making: Visualization Pending](https://img.shields.io/badge/pending-red) |

## Data Format Requirements

When exporting drawing data from Curious, you typically receive the following files:

- **report.csv**: Contains the participants' actual responses.
- **activity_user_journey.csv**: Logs the entire journey through the activity, including button actions like "Next", "Skip", "Back", and "Undo", regardless of whether a response was provided.
- **drawing-responses-{date}.zip**: A ZIP archive with raw drawing response CSV files for each participant (e.g., `drawing-responses-Mon May 29 2023.zip`).
- **media-responses-{date}.zip**: A ZIP archive containing SVG files for the drawing responses (e.g., `media-responses-Mon May 29 2023.zip`).
- **trails-responses-{date}.zip**: A ZIP archive with raw trail making response CSV files (if there are any) for each participant (e.g., `trails-responses-Mon May 29 2023.zip`).

For Spiral tasks, the toolkit uses only the CSV files from the drawing responses ZIP. Support for additional tasks will be added in future releases.

### File Naming Convention

Your spiral data files must follow this naming convention:

```text
[5123456]a7f3b2e9-d4c8-f1a6-e5b9-c2d7f8a3e6b4-spiral_trace1_Dom.csv
```

Where:

- **Participant ID**: Must be enclosed in brackets `[]` and be a 7-digit number starting with `5` (e.g., `[5123456]`) that matches the `target_secret_id` column in the **report.csv** file.
- **Activity Submission ID**: Must be a 32-character hexadecimal string (e.g., `18f2-45ea-a1e4-2334e07cc706`) that matches the `id` column in the **report.csv** file.
- **Task**: Must be one of the following that matches the `item` column in the **report.csv** file:
  - `spiral_trace1_Dom` through `spiral_trace5_Dom` (dominant hand tracing tasks)
  - `spiral_trace1_NonDom` through `spiral_trace5_NonDom` (non-dominant hand tracing tasks)
  - `spiral_recall1_Dom` through `spiral_recall3_Dom` (dominant hand recall tasks)
  - `spiral_recall1_NonDom` through `spiral_recall3_NonDom` (non-dominant hand recall tasks)

### Data Format

Your spiral data CSV file must contain the following columns:

```text
line_number, x, y, UTC_Timestamp, seconds, epoch_time_in_seconds_start
```

This format represents the standard output from [Curious drawing responses data dictionary](https://mindlogger.atlassian.net/servicedesk/customer/portal/3/article/596082739).

## Future Directions

The Graphomotor Study Toolkit is under active development. For more detailed information about upcoming features and development plans, please refer to our [GitHub Issues](https://github.com/childmindresearch/graphomotor/issues) page.

## Contributing

We welcome contributions from the community! If you're interested in contributing, please review our [Contributing Guidelines](CONTRIBUTING.md) for information on how to get started, coding standards, and the pull request process.

## References

1. Messan, K. S., Kia, S. M., Narayan, V. A., Redmond, S. J., Kogan, A., Hussain, M. A., McKhann, G. M. II, & Vahdat, S. (2022). Assessment of Smartphone-Based Spiral Tracing in Multiple Sclerosis Reveals Intra-Individual Reproducibility as a Major Determinant of the Clinical Utility of the Digital Test. Frontiers in Medical Technology, 3, 714682. [https://doi.org/10.3389/fmedt.2021.714682](https://doi.org/10.3389/fmedt.2021.714682)
