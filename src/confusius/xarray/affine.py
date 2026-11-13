"""World-space transforms and geometry summaries re-exported from VoxelData.

Use the flat `data.vd` accessor for geometry. ConfUSIus's `data.fusi` accessor
contains only fUSI analysis, plotting, scaling, and I/O operations.
"""

from voxeldata import (
    TransformCenter,
    apply_affine,
    apply_flip,
    apply_rotation,
    apply_translation,
    apply_zoom,
    get_bounding_box,
    get_geometry_center,
    get_relative_affine,
    reindex_voxels,
    reindex_voxels_like,
)

__all__ = [
    "TransformCenter",
    "apply_affine",
    "apply_flip",
    "apply_rotation",
    "apply_translation",
    "apply_zoom",
    "get_bounding_box",
    "get_geometry_center",
    "get_relative_affine",
    "reindex_voxels",
    "reindex_voxels_like",
]
