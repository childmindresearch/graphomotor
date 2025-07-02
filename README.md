# Graphomotor Study Toolkit

A Python toolkit for analysis of graphomotor data collected via Curious.

[![Build](https://github.com/childmindresearch/graphomotor/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/graphomotor/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/graphomotor/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/graphomotor)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![LGPL--2.1 License](https://img.shields.io/badge/license-LGPL--2.1-blue.svg)](https://github.com/childmindresearch/graphomotor/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/graphomotor)

## Description

The Graphomotor Study Toolkit is a comprehensive Python package designed for analyzing graphomotor data collected via [Curious](https://www.gettingcurious.com/). It provides tools for processing, analyzing, and visualizing data collected from various graphomotor tasks such as spiral drawing, trails making, alphabetic writing, digit symbol substitute and Rey-Osterrieth Complex Figure.

The toolkit focuses on extracting meaningful features from digitized drawing data, including metrics related to:

- Temporal characteristics (duration)
- Spatial accuracy (drawing error, distance-based metrics)
- Kinematic properties (velocity-based metrics)

Currently, the toolkit is in active development with the spiral drawing task as the primary focus, with plans to expand to other graphomotor tasks in future releases.

## Progress

| Task name | Preprocessing | Feature extraction | Visualization |
| :--- | :---: | :---: | :---: |
| Spiral | ![data_cleaning](https://img.shields.io/badge/pending-red) | ![feature_extraction](https://img.shields.io/badge/in_progress-yellow) | ![visualization](https://img.shields.io/badge/pending-red) |
| Rey-Osterrieth Complex Figure | ![data_cleaning](https://img.shields.io/badge/pending-red) | ![feature_extraction](https://img.shields.io/badge/pending-red) | ![visualization](https://img.shields.io/badge/pending-red) |
| Alphabetic Writing | ![data_cleaning](https://img.shields.io/badge/pending-red)|  ![feature_extraction](https://img.shields.io/badge/pending-red) | ![visualization](https://img.shields.io/badge/pending-red) |
| Digit Symbol Substitute | ![data_cleaning](https://img.shields.io/badge/pending-red)|  ![feature_extraction](https://img.shields.io/badge/pending-red) | ![visualization](https://img.shields.io/badge/pending-red) |
| Trails Making |  ![data_cleaning](https://img.shields.io/badge/pending-red) | ![feature_extraction](https://img.shields.io/badge/pending-red) | ![visualization](https://img.shields.io/badge/pending-red) |

## Installation

### From PyPI (coming soon)

```sh
pip install graphomotor
```

### From GitHub (development version)

```sh
pip install git+https://github.com/childmindresearch/graphomotor
```

## Example Usage

### Analyzing a Spiral Drawing

```python
from graphomotor.core import orchestrator

# Define input path
input_file = "path/to/your/spiral_data.csv"

# Define output directory if you want to save the results
output_dir = "path/to/output/directory"

# Run the analysis pipeline to save the extracted features to the output directory and return the extracted features as a dictionary
extracted_features = orchestrator.run_pipeline(
    input_path=input_file,
    output_path=output_dir,
    feature_categories=["duration", "velocity", "hausdorff", "AUC"],
    config_params=None,
)
```

## Future Directions

- Implement batch processing
- Implement CLI interface
- Implement caching for reference spiral generation
- Implement preprocessing for Spiral task
- Implement visualization tools for Spiral task that enables plotting single spiral from CSV either with a reference spiral or coloring each individual line differently and plotting all spirals in a directory
- Move on to other graphomotor tasks such as Rey-Osterrieth Complex Figure, Alphabetic Writing, Digit Symbol Substitute, and Trails Making

## Links and References

- [Child Mind Institute](https://childmind.org/)
- [Project Documentation](https://childmindresearch.github.io/graphomotor)
