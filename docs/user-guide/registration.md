---
icon: lucide/images
---

# Registration

!!! info "Coming soon"
    This page is currently under construction. The `registration` module provides tools
    for motion correction and spatial alignment:

    **Motion correction:**
    
    - [`register_volumewise`][confusius.registration.register_volumewise]: Register each frame in a 4D series to a reference volume.
    - [`extract_motion_parameters`][confusius.registration.extract_motion_parameters]: Extract rigid-body motion parameters from transformation matrices.
    - [`compute_framewise_displacement`][confusius.registration.compute_framewise_displacement]: Compute frame-to-frame displacement from motion parameters.
    - [`create_motion_dataframe`][confusius.registration.create_motion_dataframe]: Create a DataFrame with motion parameters for visualization.

    **Volume registration:**
    
    - [`register_volume`][confusius.registration.register_volume]: Register a 3D volume to a reference using rigid, affine, or deformable transforms.
    - [`resample_volume`][confusius.registration.resample_volume]: Resample a volume to a target affine and shape.
    - [`resample_like`][confusius.registration.resample_like]: Resample a volume to match the affine and shape of a reference.

    **Affine utilities:**
    
    - [`compose_affine`][confusius.registration.compose_affine]: Compose multiple affine transformations into one.
    - [`decompose_affine`][confusius.registration.decompose_affine]: Decompose an affine matrix into translation, rotation, and scaling components.

    Please refer to the [API Reference](../api/registration.md)
    for more information.

## SyN volume registration

[`register_volume`][confusius.registration.register_volume] supports symmetric
normalization (SyN) with `transform_type="syn"`. Use `metric="ants_cc"` for ANTs
neighborhood correlation in same-modality registration. Both inputs must be
spatial-only VoxelData arrays.

For large global misalignment, estimate a rigid or affine initialization first:

```python
from confusius.registration import register_volume, resample_like

_, affine, _ = register_volume(moving, fixed, transform_type="rigid", resample=False)
registered, field, diagnostics = register_volume(
    moving,
    fixed,
    transform_type="syn",
    metric="ants_cc",
    initialization=affine,
    learning_rate=0.1,
    number_of_iterations=30,
    update_field_variance=3.0,
    total_field_variance=1.0,
)

reapplied = resample_like(moving, fixed, field)
```

The returned `field` is a VoxelData displacement field on the original fixed grid.
It maps fixed-world points to moving-world points (the same pull convention as
other registration methods), and already includes affine initialization. Do not
apply the affine a second time. The original moving signal is resampled only
once; intensity scaling affects optimizer inputs, not the returned signal.

`correlation_radius` controls the neighborhood radius in voxels. Gaussian
update/total-field variances are in squared voxel units, not squared world units.
For SyN, `learning_rate="auto"` selects a fixed maximum update length of `0.1`
voxels. Progress reporters and cooperative cancellation are supported.

Single-slice inputs use the existing 3D thin-axis expansion. Smoothing acts on
that expanded grid, so settings are not interchangeable with a native 2D SyN
pipeline. As with other discretized deformable methods, inspect the estimated
field and Jacobians rather than assuming global invertibility from the algorithm's
name alone.

This first implementation supports **single-resolution volume registration**.
`use_multi_resolution=True` raises because the ITK wheels do not expose the
required displacement-field parameter adaptors. Optimizer weights are also
unsupported. The xarray equivalent is
`moving.fusi.register.to_volume(fixed, transform="syn", metric="ants_cc")`.
Volumewise SyN and napari method-selection/transform-file UI support are not yet
available.
