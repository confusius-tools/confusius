"""Multi-pose data processing utilities.

This module provides functions for processing multi-pose fUSI data, including
assembling independently loaded poses into one array, consolidating multiple poses
into a single volume, slice timing correction, and other multi-pose specific
operations.
"""

from confusius.multipose.consolidate import consolidate_poses
from confusius.multipose.slice_timing import correct_slice_timings
from confusius.multipose.stack import stack_poses

__all__ = ["consolidate_poses", "correct_slice_timings", "stack_poses"]
