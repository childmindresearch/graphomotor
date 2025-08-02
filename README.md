# `graphomotor`

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
> `graphomotor` is under active development. Currently, the focus is on the spiral drawing task. After finalizing feature extraction, the next steps will involve implementing both preprocessing and visualization for this task. Once these parts are in place, we plan to extend support to other tasks.

## Feature Extraction Capabilities

The toolkit extracts clinically relevant metrics from digitized drawing data. Currently implemented features include:

- **Temporal Features**: Task completion duration.
- **Velocity Features**: Velocity analysis including linear, radial, and angular velocity components with statistical measures (sum, median, variation, skewness, kurtosis).
- **Distance Features**: Spatial accuracy measurements using Hausdorff distance metrics with temporal normalizations and segment-specific analysis.
- **Drawing Error Features**: Area under the curve (AUC) calculations between drawn paths and ideal reference trajectories to quantify spatial accuracy.

## Installation

Install `graphomotor` from PyPI:

```sh
pip install graphomotor
```

Or install the latest development version directly from GitHub:

```sh
pip install git+https://github.com/childmindresearch/graphomotor
```

## Quick Start

`graphomotor` is available both as a **command-line interface** and as an importable **Python library** for easy integration into analysis workflows.

> [!CAUTION]
> Input data must follow the [Curious drawing responses format](https://mindlogger.atlassian.net/servicedesk/customer/portal/3/article/859242501). See [Data Format Requirements](#data-format-requirements) below.

### Command-Line Interface

**Run a single file:**

```bash
graphomotor /path/to/data.csv /path/to/output/features.csv
```

**Run entire directories:**

```bash
graphomotor /path/to/data_directory/ /path/to/output/features.csv
```

**For a full list of arguments and options, run:**

```bash
graphomotor --help
```

### Python Library

**Run a single file:**

```python
from graphomotor.core import orchestrator

# Define input path
input_path = "path/to/data.csv"

# Define output path to save results to disk
# If output path is a directory, file name will be auto-generated as
# `{participant_id}_{task}_{hand}_features_{YYYYMMDD_HHMM}.csv`
output_path = "path/to/output/features.csv"

# Run the pipeline
results_df = orchestrator.run_pipeline(input_path=input_path, output_path=output_path)
```

**Run entire directories:**

```python
from graphomotor.core import orchestrator

# Define input directory
input_dir = "path/to/data_directory/"

# Define output path to save results to disk
# If output path is a directory, file name will be auto-generated as
# `batch_features_{YYYYMMDD_HHMM}.csv`
output_dir = "path/to/output/"

# Run the pipeline
results_df = orchestrator.run_pipeline(input_path=input_dir, output_path=output_dir)
```

**Access results:**

```python
# Pipeline returns a DataFrame with extracted metadata and features
print(f"Processed {len(results_df)} files")
print(f"Extracted metadata and features: {results_df.columns.tolist()}")

# Get data for first file
file_path = results_df.index[0]
participant = results_df.loc[file_path, 'participant_id']
task = results_df.loc[file_path, 'task']
duration = results_df.loc[file_path, 'duration']
```

> [!NOTE]
> For detailed configuration options and additional parameters, refer to the [`run_pipeline` documentation](https://childmindresearch.github.io/graphomotor/graphomotor/core/orchestrator.html#run_pipeline).

## Development Progress

| Task | Preprocessing | Feature Extraction | Visualization |
| :--- | :---: | :---: | :---: |
| Spiral | ![Spiral: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Spiral: Feature Extraction In Progress](https://img.shields.io/badge/in_progress-yellow) | ![Spiral: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Rey-Osterrieth Complex Figure | ![Rey-Osterrieth: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Rey-Osterrieth: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Rey-Osterrieth: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Alphabetic Writing | ![Alphabetic Writing: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Alphabetic Writing: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Alphabetic Writing: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Digit Symbol Substitution | ![Digit Symbol Substitution: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Digit Symbol Substitution: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Digit Symbol Substitution: Visualization Pending](https://img.shields.io/badge/pending-red) |
| Trails Making | ![Trails Making: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Trails Making: Feature Extraction Pending](https://img.shields.io/badge/pending-red) | ![Trails Making: Visualization Pending](https://img.shields.io/badge/pending-red) |

## Data Format Requirements

When exporting drawing data from Curious, the export typically includes the following files:

- **report.csv**: Contains the participants' actual responses.
- **activity_user_journey.csv**: Logs the entire journey through the activity, including button actions like "Next", "Skip", "Back", and "Undo", regardless of whether a response was provided.
- **drawing-responses-{date}.zip**: A ZIP archive with raw drawing response CSV files for each participant (e.g., `drawing-responses-Mon May 29 2023.zip`).
- **media-responses-{date}.zip**: A ZIP archive containing SVG files for the drawing responses (e.g., `media-responses-Mon May 29 2023.zip`).
- **trails-responses-{date}.zip**: A ZIP archive with raw trail making response CSV files (if there are any) for each participant (e.g., `trails-responses-Mon May 29 2023.zip`).

For Spiral tasks, the toolkit uses only the CSV files from the drawing responses ZIP. Support for additional tasks will be added in future releases.

### File Naming Convention

Spiral data files must follow this naming convention:

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

Spiral data CSV files must contain the following columns:

```text
line_number, x, y, UTC_Timestamp, seconds, epoch_time_in_seconds_start
```

This format represents the standard output from [Curious drawing responses data dictionary](https://mindlogger.atlassian.net/servicedesk/customer/portal/3/article/596082739).

## Future Directions

The Graphomotor Study Toolkit is under active development. For more detailed information about upcoming features and development plans, please refer to our [GitHub Issues](https://github.com/childmindresearch/graphomotor/issues) page.

## Contributing

Contributions from the community are welcome! Please review the [Contributing Guidelines](CONTRIBUTING.md) for information on how to get started, coding standards, and the pull request process.

## References

1. Messan, K. S., Kia, S. M., Narayan, V. A., Redmond, S. J., Kogan, A., Hussain, M. A., McKhann, G. M. II, & Vahdat, S. (2022). Assessment of Smartphone-Based Spiral Tracing in Multiple Sclerosis Reveals Intra-Individual Reproducibility as a Major Determinant of the Clinical Utility of the Digital Test. Frontiers in Medical Technology, 3, 714682. [https://doi.org/10.3389/fmedt.2021.714682](https://doi.org/10.3389/fmedt.2021.714682)
