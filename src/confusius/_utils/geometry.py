"""VoxelData geometry re-exports for ConfUSIus's internal consumers.

Geometry is owned by the standalone `voxeldata` package; do not add a second
implementation here. Probe-specific defaults belong at the I/O boundary.
"""

from voxeldata.geometry import (
    VoxelToWorldIndex,
    VoxelToWorldTransform,
    attach_voxel_to_world_index,
    convert_length_units,
    get_affine_axis_scalings,
    get_affine_direction_matrix,
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spacings_from_coords,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_single_geometry_affine,
    restore_voxel_to_world_index,
)

__all__ = [
    "VoxelToWorldIndex",
    "VoxelToWorldTransform",
    "attach_voxel_to_world_index",
    "convert_length_units",
    "get_affine_axis_scalings",
    "get_affine_direction_matrix",
    "get_voxel_to_world_affine",
    "get_voxel_to_world_direction_matrix",
    "get_voxel_to_world_index",
    "get_voxel_to_world_index_origin",
    "get_voxel_to_world_index_spacing",
    "get_voxel_to_world_spacings_from_coords",
    "get_voxel_to_world_spatial_dims",
    "get_voxel_to_world_units",
    "has_axis_aligned_voxel_to_world_index",
    "has_voxel_to_world_index",
    "require_single_geometry_affine",
    "restore_voxel_to_world_index",
]
