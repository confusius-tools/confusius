"""VoxelData dimension conventions shared by ConfUSIus consumers."""

from voxeldata import TIME_DIM, VOXEL_DIMS, WORLD_DIMS

GRID_DIM = "grid"
"""Probe acquisition dimension used by ConfUSIus."""

CORE_DIMS = (GRID_DIM, TIME_DIM, *VOXEL_DIMS)
"""ConfUSIus core dimensions in canonical order."""

__all__ = ["CORE_DIMS", "GRID_DIM", "TIME_DIM", "VOXEL_DIMS", "WORLD_DIMS"]
