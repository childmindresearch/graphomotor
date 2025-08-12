"""Utility functions and mathematical operations for graphomotor data processing.

This module provides essential computational and visualization support functions:

## Reference Spiral Generation

- **Archimedean Spiral Mathematics**: Implements r(θ) = a + b·θ with precise
  arc length calculations using numerical integration (scipy.integrate.quad)
- **Equidistant Point Distribution**: Uses root-finding algorithms to place
  points at equal arc length intervals along the spiral curve
- **LRU Caching**: Memoizes expensive spiral computations for performance
  optimization across repeated operations
- **Configurable Parameters**: Supports custom center coordinates, growth rates,
  angle ranges, and point density through SpiralConfig

## Geometric Transformations

- **Spiral Centering**: Type-overloaded function supporting both Spiral objects
  and numpy arrays with automatic coordinate translation to origin
- **Coordinate System Normalization**: Translates spiral data to standardized
  reference frame for consistent feature extraction

## Plotting Infrastructure

- **Data Validation Pipeline**: Comprehensive DataFrame validation using Spiral
  model constraints (7-digit participant IDs, Dom/NonDom hands, valid tasks)
- **Task Metadata Enhancement**: Adds task ordering and categorization
  (trace/recall) for structured visualization
- **Dynamic Subplot Generation**: Adaptive grid layouts based on feature count
  with responsive sizing algorithms
- **Feature Name Formatting**: Intelligent text wrapping for readable plot labels
- **High-Quality Export**: PNG output with 300 DPI resolution
  and timestamp-based naming

These utilities ensure mathematical precision and computational efficiency
throughout the graphomotor analysis pipeline.
"""
