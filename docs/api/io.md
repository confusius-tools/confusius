---
icon: lucide/file-input
---

# `confusius.io`

## Zarr persistence

The generic `load` and `save` functions delegate Zarr array persistence to
VoxelData, sharing its geometry, native voxel labels, units, and named-frame
representation. ConfUSIus retains acquisition-aware validation before saving and
atlas colormap restoration after loading. NIfTI and acquisition formats remain
ConfUSIus implementations; NetCDF is available through VoxelData directly, not
through this dispatcher.

ConfUSIus keeps Dask-backed Zarr loading by default (`chunks="auto"`). Pass
`chunks=None` for backend lazy indexing without Dask arrays. Stores containing
multiple data variables require an explicit `variable` selection. Incremental
Zarr writes (`append_dim`, `region`, or modes `a`, `a-`, `r+`) are rejected by
VoxelData to avoid inconsistent geometry metadata; save the complete array.

Atlas Dataset persistence and streaming AUTC conversion retain their specialized
storage layouts and are not routed through VoxelData's single-array writer.

::: confusius.io
