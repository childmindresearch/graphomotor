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
> `graphomotor` is under active development. So far, the feature extraction and visualization components for the spiral drawing task are complete. The next steps involve implementing preprocessing for this task and extending support to other tasks.

## Feature Extraction Capabilities

The toolkit extracts 25 clinically relevant metrics from digitized drawing data. Currently implemented feature categories include:

- **Velocity Features (15)**: Velocity analysis including linear, radial, and angular velocity components with statistical measures (sum, median, variation, skewness, kurtosis).
- **Distance Features (8)**: Spatial accuracy measurements using Hausdorff distance metrics with temporal normalizations and segment-specific analysis.
- **Drawing Error Features (1)**: Area under the curve (AUC) calculations between drawn paths and ideal reference trajectories to quantify spatial accuracy.
- **Temporal Features (1)**: Task completion duration.

## Feature Visualization Capabilities

The toolkit provides several plotting functions to visualize extracted features:

- **Distribution Plots**: Kernel density estimation plots showing feature distributions grouped by task type and hand.
- **Trend Plots**: Line plots displaying feature progression across task sequences with individual participant trajectories and group means.
- **Box Plots**: Box-and-whisker plots comparing feature distributions across different tasks and hand conditions.
- **Cluster Heatmaps**: Hierarchically clustered heatmaps of z-score standardized features to identify patterns across conditions.

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

`graphomotor` provides two main functionalities: [feature extraction](#feature-extraction) from raw drawing data and [visualization](#feature-visualization) of extracted features. Both are available as a **command-line interface** (CLI) and an **importable Python library**.

> [!CAUTION]
> Input data must follow the [Curious drawing responses format](https://mindlogger.atlassian.net/servicedesk/customer/portal/3/article/859242501). See [Data Format Requirements](#data-format-requirements) below.

**To see all available commands and global options in the CLI:**

```bash
graphomotor --help
```

### Feature Extraction

The `extract` command computes clinically relevant features from spiral drawing data. It processes drawing CSV files exported from Curious and outputs a structured dataset containing participant metadata and [extracted features for each drawing.

#### CLI Usage for Feature Extraction

**To extract features from a single file:**

```bash
graphomotor extract /path/to/data.csv /path/to/output/features.csv
```

**To extract features from entire directories:**

```bash
graphomotor extract /path/to/data_directory/ /path/to/output/features.csv
```

**To see all available options for `extract`:**

```bash
graphomotor extract --help
```

#### Python Library Usage for Feature Extraction

**To extract features from a single file:**

```python
from graphomotor.core import orchestrator

# Define input path
input_path = "/path/to/data.csv"

# Define output path to save results to disk
# If output path is a directory, file name will be auto-generated as
# `{participant_id}_{task}_{hand}_features_{YYYYMMDD_HHMM}.csv`
output_path = "/path/to/output/features.csv"

# Run the pipeline
results_df = orchestrator.run_pipeline(input_path=input_path, output_path=output_path)
```

**To extract features from entire directories:**

```python
from graphomotor.core import orchestrator

# Define input directory
input_dir = "/path/to/data_directory/"

# Define output path to save results to disk
# If output path is a directory, file name will be auto-generated as
# `batch_features_{YYYYMMDD_HHMM}.csv`
output_dir = "/path/to/output/"

# Run the pipeline
results_df = orchestrator.run_pipeline(input_path=input_dir, output_path=output_dir)
```

**To access the results:**

```python
# run_pipeline() returns a DataFrame with extracted metadata and features
print(f"Processed {len(results_df)} files")
print(f"Extracted metadata and features: {results_df.columns.tolist()}")

# Get data for first file
# DataFrame is indexed by file path
file_path = results_df.index[0]
participant = results_df.loc[file_path, 'participant_id']
task = results_df.loc[file_path, 'task']
duration = results_df.loc[file_path, 'duration']
```

### Feature Visualization

Generate plots directly from the feature output produced by the `extract` functionality. The plotting functions expect CSV files in the same format as those generated by `graphomotor extract`, with the first five columns reserved for metadata (`source_file`, `participant_id`, `task`, `hand`, `start_time`) and all subsequent columns treated as numerical features. Output plots will be saved to the specified output directory.

> [!TIP]
> **Custom Features**: You can add custom feature columns to your output CSV files alongside the standard graphomotor features. The plotting functions will automatically detect and include any additional columns after the first 5 metadata columns.

#### CLI Usage for Feature Visualization

**To generate all available plot types with all features in the input file:**

```bash
graphomotor plot-features /path/to/feature_output.csv /path/to/plots/
```

**To generate only selected plot types for specific features:**

```bash
graphomotor plot-features /path/to/feature_output.csv /path/to/plots/ -p dist -p trend -f area_under_curve -f duration
```

**To see all available options for `plot-features`:**

```bash
graphomotor plot-features --help
```

#### Python Library Usage for Feature Visualization

**To generate a distribution plot for specific features and save it:**

```python
import matplotlib.pyplot as plt
from graphomotor.plot import feature_plots

# Define paths to data, output directory and features to plot
data = '/path/to/batch_features.csv'
output_dir = '/path/to/plots/'
features = ['linear_velocity_median', 'hausdorff_distance_maximum']

# Generate and save distribution plots for selected features
feature_plots.plot_feature_distributions(
  data=data,
  output_path=output_dir,
  features=features
)
```

**To generate boxplots for all available features:**

```python
import matplotlib.pyplot as plt
from graphomotor.plot import feature_plots

# Define paths to data and output directory
data = '/path/to/batch_features.csv'
output_dir = '/path/to/plots/'

# Generate boxplots for all available features and return the figure object
fig = feature_plots.plot_feature_boxplots(
  data=data,
  output_path=output_dir
)

# It is possible to customize the figure object further before displaying or saving it.
# Set a global title
fig.suptitle('Boxplots of All Extracted Features', fontsize=18, fontweight='bold')

# Customize axes: rotate x-tick labels, set grid, highlight outliers
for ax in fig.get_axes():
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)
    for line in ax.get_lines():
        if line.get_label() == 'fliers':
            line.set_markerfacecolor('red')
            line.set_markeredgecolor('red')

# Save the figure after the changes
fig.savefig(f"{output_dir}/customized_boxplots.png", dpi=300)

plt.show()
```

> [!NOTE]
> For detailed configuration options and additional parameters for feature extraction, refer to the [`run_pipeline` documentation](https://childmindresearch.github.io/graphomotor/graphomotor/core/orchestrator.html#run_pipeline).
>
> For all available feature plotting options, refer to the [`feature_plots` documentation](https://childmindresearch.github.io/graphomotor/graphomotor/plot/feature-plots.html).

## Development Progress

| Task | Preprocessing | Feature Extraction | Visualization |
| :--- | :---: | :---: | :---: |
| Spiral | ![Spiral: Preprocessing Pending](https://img.shields.io/badge/pending-red) | ![Spiral: Feature Extraction Complete](https://img.shields.io/badge/complete-green) | ![Spiral: Visualization Complete](https://img.shields.io/badge/complete-green) |
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

These columns constitute the standard output from [Curious drawing responses data dictionary](https://mindlogger.atlassian.net/servicedesk/customer/portal/3/article/596082739).

## Future Directions

The `graphomotor` is under active development. For more detailed information about upcoming features and development plans, please refer to our [GitHub Issues](https://github.com/childmindresearch/graphomotor/issues) page.

## Contributing

Contributions from the community are welcome! Please review the [Contributing Guidelines](CONTRIBUTING.md) for information on how to get started, coding standards, and the pull request process.

## References

1. Messan, K. S., Kia, S. M., Narayan, V. A., Redmond, S. J., Kogan, A., Hussain, M. A., McKhann, G. M. II, & Vahdat, S. (2022). Assessment of Smartphone-Based Spiral Tracing in Multiple Sclerosis Reveals Intra-Individual Reproducibility as a Major Determinant of the Clinical Utility of the Digital Test. Frontiers in Medical Technology, 3, 714682. [https://doi.org/10.3389/fmedt.2021.714682](https://doi.org/10.3389/fmedt.2021.714682)
