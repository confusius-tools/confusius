"""Multi-pose data processing utilities.

This module provides functions for processing multi-pose fUSI data, including
consolidating multiple poses into a single volume, slice timing correction, and other
multi-pose specific operations.
"""

from confusius.multipose.consolidate import consolidate_poses
from confusius.multipose.slice_timing import correct_slice_timings

__all__ = ["consolidate_poses", "correct_slice_timings"]
