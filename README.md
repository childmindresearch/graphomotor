# Graphomotor Study Toolkit

A Python toolkit for analysis of graphomotor data collected via Curious.

[![Build](https://github.com/childmindresearch/graphomotor/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/graphomotor/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/graphomotor/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/graphomotor)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/graphomotor/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/graphomotor)

Welcome to `graphomotor`, a specialized Python library for analyzing graphomotor data collected via [Curious](https://www.gettingcurious.com/). This toolkit provides comprehensive tools for processing, analyzing, and visualizing data from various graphomotor assessment tasks including spiral drawing, trails making, alphabetic writing, digit symbol substitution, and the Rey-Osterrieth Complex Figure Test.

## Feature Extraction Capabilities

The toolkit extracts clinically relevant metrics from digitized drawing data. Currently implemented features include:

- **Temporal Features**: Task completion duration
- **Velocity Features**: Velocity analysis including linear, radial, and angular velocity components with statistical measures (sum, median, variation, skewness, kurtosis)
- **Distance Features**: Spatial accuracy measurements using Hausdorff distance metrics with temporal normalizations and segment-specific analysis
- **Drawing Error Features**: Area under the curve (AUC) calculations between drawn paths and ideal reference trajectories to quantify spatial accuracy

> **Note:** This toolkit is under active development, with a primary focus on the spiral drawing task and planned expansion to additional graphomotor assessments in subsequent releases.

## Development Progress

| Task | Preprocessing | Feature Extraction | Visualization |
| :--- | :---: | :---: | :---: |
| Spiral | ![pending status](https://img.shields.io/badge/pending-red) | ![in progress status](https://img.shields.io/badge/in_progress-yellow) | ![pending status](https://img.shields.io/badge/pending-red) |
| Rey-Osterrieth Complex Figure | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) |
| Alphabetic Writing | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) |
| Digit Symbol Substitution | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) |
| Trails Making | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) | ![pending status](https://img.shields.io/badge/pending-red) |

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

Currently, `graphomotor` is available as an importable Python library. CLI functionality is planned for future releases.

### Extracting Features from Spiral Drawing Data

```python
from graphomotor.core import orchestrator

# Path to your spiral drawing data file
input_file = "path/to/your/spiral_data.csv"

# Directory where extracted features will be saved
output_dir = "path/to/output/directory"

# Run the analysis pipeline
features = orchestrator.run_pipeline(
    input_path=input_file,
    output_path=output_dir
)

# Features are returned as a dictionary and saved as CSV
print(f"Successfully extracted {len(features)} feature categories")
```

For detailed configuration options and additional parameters, refer to the [`run_pipeline` documentation](https://childmindresearch.github.io/graphomotor/graphomotor/core/orchestrator.html#run_pipeline).

> **Note:** Currently, only single file processing is supported, with batch processing planned for future releases.

## Future Directions

The Graphomotor Study Toolkit is under active development. For more detailed information about upcoming features and development plans, please refer to the [GitHub Issues](https://github.com/childmindresearch/graphomotor/issues) page.

## Contributing

We welcome contributions from the community! If you're interested in contributing, please review our [Contributing Guidelines](CONTRIBUTING.md) for information on how to get started, coding standards, and the pull request process.

## References

1. Messan, K. S., Kia, S. M., Narayan, V. A., Redmond, S. J., Kogan, A., Hussain, M. A., McKhann, G. M. II, & Vahdat, S. (2022). Assessment of Smartphone-Based Spiral Tracing in Multiple Sclerosis Reveals Intra-Individual Reproducibility as a Major Determinant of the Clinical Utility of the Digital Test. Frontiers in Medical Technology, 3, 714682. [https://doi.org/10.3389/fmedt.2021.714682](https://doi.org/10.3389/fmedt.2021.714682)
