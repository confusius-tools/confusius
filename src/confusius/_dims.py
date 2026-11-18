"""VoxelData dimension conventions shared by ConfUSIus consumers."""

from voxeldata import TIME_DIM, VOXEL_DIMS, WORLD_DIMS

POSE_DIM: str = "pose"
"""Probe acquisition dimension used by ConfUSIus."""

CORE_DIMS: tuple[str, ...] = (POSE_DIM, TIME_DIM, *VOXEL_DIMS)
"""ConfUSIus core dimensions in canonical order."""

__all__ = ["CORE_DIMS", "POSE_DIM", "TIME_DIM", "VOXEL_DIMS", "WORLD_DIMS"]
