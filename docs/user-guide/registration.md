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
