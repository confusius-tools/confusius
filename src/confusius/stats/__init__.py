"""Statistical inference utilities for fUSI analysis.

This package provides shared statistical utilities that apply across multiple analysis
workflows, such as statistical-map thresholding with multiple-comparison correction.
"""

from confusius.stats.thresholding import apply_statistical_threshold

__all__ = ["apply_statistical_threshold"]
