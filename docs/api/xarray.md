---
icon: lucide/brackets
---

# `confusius.xarray`

::: confusius.xarray

## Geometry: VoxelData

Geometry is implemented by the external
[`voxeldata`](https://github.com/confusius-tools/voxeldata) package. Its flat
`.vd` accessor replaces `.fusi.affine` and the former `.fusi` geometry properties.
ConfUSIus re-exports module-level affine functions but does not implement them.

::: voxeldata.accessors.VoxelDataAccessor

### Module-level geometry operations

::: voxeldata.affine.apply_affine

::: voxeldata.affine.apply_translation

::: voxeldata.affine.apply_rotation

::: voxeldata.affine.apply_zoom

::: voxeldata.affine.apply_flip

::: voxeldata.affine.get_bounding_box

::: voxeldata.affine.get_geometry_center

::: voxeldata.affine.get_relative_affine

::: voxeldata.affine.reindex_voxels

::: voxeldata.affine.reindex_voxels_like

::: voxeldata.geometry.get_array_origin

::: voxeldata.geometry.get_array_spacing

### Generic validation

Unlike `confusius.validation`, these functions do not require fUSI acquisition
metadata on the time coordinate.

::: voxeldata.validation.validate_voxeldata

::: voxeldata.validation.canonicalize_voxeldata
