"""Affine operations re-exported from VoxelData.

Use the flat `data.vd` accessor for geometry. ConfUSIus's `data.fusi` accessor
contains only fUSI analysis, plotting, scaling, and I/O operations.
"""

from voxeldata import (
    apply_affine,
    get_bounding_box,
    get_relative_affine,
    reindex_voxels,
    reindex_voxels_like,
)

__all__ = [
    "apply_affine",
    "get_bounding_box",
    "get_relative_affine",
    "reindex_voxels",
    "reindex_voxels_like",
]
