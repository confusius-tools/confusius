---
icon: lucide/waypoints
---

# Multi-Pose Data

## What is Multi-Pose Imaging?

In **multi-pose fUSI**, a probe is physically stepped to a series of positions along one
spatial axis. At each position (a **pose**), one or more volumes are acquired. Stacking
the poses together extends the field of view beyond what a single probe position can
cover.

The probe at each pose can image a 2D plane or a 3D volume, depending on the probe type:

- **Linear probes** (e.g., standard linear probes): each pose yields a single 2D image
  (one elevation slice). Stepping across *N* poses and stacking gives a 3D volume of *N*
  elevation slices.
- **2D probes** (e.g., matrix, RCA, or stacked linear probes):
  each pose already yields a 3D volume. Stepping across *N* poses concatenates these
  volumes into a larger 3D volume.

Multiple fUSI systems support this approach, including Iconeus, EchoFrame, and AUTC.
ConfUSIus represents multi-pose data with a `pose` dimension and per-pose affine
transformations that record the physical position of each pose.

!!! warning "Rotational sweeps are not yet supported"
    [`consolidate_poses`][confusius.multipose.consolidate_poses] requires a
    **purely translational** sweep, where the probe is shifted along one axis without
    rotating. Rotational sweeps (so-called tomographic acquisitions) are not yet
    supported and will raise a `ValueError`.

## Loading Multi-Pose Data

### Iconeus SCAN Files

Iconeus IcoScan stores recordings in **SCAN files** (`.scan`, `.source.scan`). Three
acquisition modes are supported by ConfUSIus:

| Mode | Dimensions | Typical use |
|------|------------|-------------|
| `2Dscan` | `(time, z, y, x)` | Single-pose fUSI time-series |
| `3Dscan` | `(pose, z, y, x)` | Multi-pose anatomical volume |
| `4Dscan` | `(time, pose, z, y, x)` | Multi-pose fUSI time-series (3D+t fUSI) |

Use [`load_scan`][confusius.io.load_scan] to load SCAN files. This page focuses on
**3Dscan** and **4Dscan**. See the [I/O guide](io.md#loading-iconeus-scan-files) for a
general overview of SCAN file loading.

The examples below illustrate a recording from a mouse acquired with an **IcoPrime-4D
MultiArray probe**—four linear probes stacked along the elevation axis, giving 4
elevation slices per pose—translated across multiple regularly spaced positions.

=== "3Dscan (anatomical)"

    ```python
    from confusius.io import load_scan

    anat = load_scan("sub-01_acq-anat_pwd.scan")
    print(anat)
    ```

    ```text
    <xarray.DataArray 'scan_data' (pose: 15, z: 4, y: 72, x: 64)> Size: 2MB
    dask.array<transpose, shape=(15, 4, 72, 64), dtype=float64, chunksize=(15, 4, 72, 64), chunktype=numpy.ndarray>
    Coordinates:
      * pose     (pose) int64 120B 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
      * z        (z) float64 32B 0.0 2.1 4.2 6.3
      * y        (y) float64 576B 2.0 2.099 2.197 2.296 ... 8.702 8.801 8.899 8.998
      * x        (x) float64 512B -3.465 -3.355 -3.245 -3.135 ... 3.245 3.455 3.465
    Attributes:
        affines:            {'physical_to_lab': ...}  # shape (15, 4, 4)
        scan_mode:          3Dscan
        ...
    ```

    The probe was stepped across 15 positions, each contributing 4 elevation slices —
    a total of 60 slices once consolidated.

=== "4Dscan (functional)"

    ```python
    from confusius.io import load_scan

    fus = load_scan("sub-01_task-awake_pwd.scan")
    print(fus)
    ```

    ```text
    <xarray.DataArray 'scan_data' (time: 750, pose: 4, z: 4, y: 72, x: 64)> Size: 442MB
    dask.array<transpose, shape=(750, 4, 4, 72, 64), dtype=float64, chunksize=(227, 4, 4, 72, 64), chunktype=numpy.ndarray>
    Coordinates:
      * time       (time) float64 6kB 0.4 2.8 5.2 ... 1.793e+03 1.796e+03 1.798e+03
      * pose       (pose) int64 32B 0 1 2 3
        pose_time  (time, pose) float64 24kB 0.4 2.2 1.0 ... 1.799e+03 1.799e+03
      * z          (z) float64 32B 0.0 2.1 4.2 6.3
      * y          (y) float64 576B 2.0 2.099 2.197 2.296 ... 8.801 8.899 8.998
      * x          (x) float64 512B -3.465 -3.355 -3.245 ... 3.245 3.355 3.465
    Attributes:
        affines:            {'physical_to_lab': ...}  # shape (4, 4, 4)
        scan_mode:          4Dscan
        ...
    ```

    The probe was stepped across 4 positions, each contributing 4 elevation slices —
    a total of 16 slices once consolidated.

### Other Systems

For other fUSI systems, multi-pose data must be assembled manually: load or construct
one DataArray per pose, stack them along a new `pose` dimension, and populate
`da.attrs["affines"]` with a `(npose, 4, 4)` array of per-pose affines.

## Physical Coordinates and Affines

Spatial coordinates in a multi-pose DataArray are **pose-relative**: the `z` coordinate
(or whichever dimension is being swept) is defined in the probe frame and is the same for
every pose. The per-pose affines stored in `da.attrs["affines"]` map these probe-relative
coordinates to a common world space and record how each pose is positioned in that space.

For Iconeus SCAN files, [`load_scan`][confusius.io.load_scan] automatically stores a
`physical_to_lab` affine of shape `(npose, 4, 4)`—one matrix per pose.

## The `pose_time` Coordinate

When poses are acquired sequentially, each pose is captured at a slightly different
time. The `pose_time` non-dimension coordinate of shape `(time, pose)` records the exact
per-pose acquisition timestamp:

```python
fus.coords["pose_time"]  # (time, pose) in seconds.
```

This is important for slice timing correction, which accounts for the fact that different
poses were not acquired simultaneously.

## Pose Consolidation

[`consolidate_poses`][confusius.multipose.consolidate_poses] merges the `pose` dimension
and the sweep spatial dimension into a single axis with physically meaningful
coordinates, producing a standard ConfUSIus DataArray.
[`consolidate_poses`][confusius.multipose.consolidate_poses] performs the following
steps:

1. Read the per-pose affines to compute the world position of every `(pose, sweep_dim)`
   voxel.
2. Find the primary sweep direction via SVD of all voxel positions.
3. Project each voxel onto that axis and check that the resulting positions form a
   regular grid.
4. Reindex the data in ascending position order, replacing `pose` and `sweep_dim` with
   a single consolidated coordinate in world space.

=== "3Dscan (anatomical)"

    ```python
    import confusius as cf

    anat = cf.load("sub-01_acq-anat_pwd.scan")
    volume = cf.multipose.consolidate_poses(anat)
    print(volume)
    ```

    ```text
    <xarray.DataArray 'scan_data' (z: 60, y: 72, x: 64)> Size: 2MB
    array([...])
    Coordinates:
      * z        (z) float64 480B -21.38 -21.24 -21.1 -20.96 ... -13.4 -13.26 -13.12
      * y        (y) float64 576B 2.0 2.099 2.197 2.296 ... 8.702 8.801 8.899 8.998
      * x        (x) float64 512B -3.465 -3.355 -3.245 -3.135 ... 3.245 3.355 3.465
    Attributes:
        affines:            {'physical_to_lab': ...}  # shape (4, 4)
        scan_mode:          3Dscan
        ...
    ```

    15 poses × 4 slices = 60 consolidated z positions, spanning −21.4 to −13.1 mm in
    lab coordinates.

=== "4Dscan (functional)"

    ```python
    import confusius as cf

    fus = cf.load("sub-01_task-awake_pwd.scan")
    volume = cf.multipose.consolidate_poses(fus)
    print(volume)
    ```

    ```text
    <xarray.DataArray 'scan_data' (time: 750, z: 16, y: 72, x: 64)> Size: 442MB
    array([...])
    Coordinates:
      * time       (time) float64 6kB 0.4 2.8 5.2 ... 1.793e+03 1.796e+03 1.798e+03
      * z          (z) float64 128B -21.38 -20.86 -20.33 ... -14.56 -14.03 -13.51
        pose_time  (time, z) float64 96kB 0.4 2.2 1.0 ... 1.799e+03 1.799e+03
      * y          (y) float64 576B 2.0 2.099 2.197 2.296 ... 8.801 8.899 8.998
      * x          (x) float64 512B -3.465 -3.355 -3.245 ... 3.245 3.355 3.465
    Attributes:
        affines:            {'physical_to_lab': ...}  # shape (4, 4)
        scan_mode:          4Dscan
        ...
    ```

    4 poses × 4 slices = 16 consolidated z positions. The `pose_time` coordinate is
    preserved with dims `(time, z)`: each slice retains the timestamp of the pose it
    came from.

After consolidation, the per-pose affine stack is reduced to a single `(4, 4)` matrix
representing the consolidated volume's orientation in world space.

### Parameters

[`consolidate_poses`][confusius.multipose.consolidate_poses] accepts two parameters that
may need adjusting depending on your setup:

- **`sweep_dim`** (default: `"z"`): the spatial dimension being swept across poses.
  Change this if your sweep is along a different axis.
- **`affines_key`** (default: `"physical_to_lab"`): the key into `da.attrs["affines"]`
  that holds the per-pose affine stack. Change this if your affines are stored under a
  different key.

```python
# Example: sweeping along x using affines stored under a custom key.
volume = cf.multipose.consolidate_poses(
    da,
    sweep_dim="x",
    affines_key="physical_to_scanner",
)
```

!!! warning "Regularity requirement"
    [`consolidate_poses`][confusius.multipose.consolidate_poses] will raise a
    `ValueError` if the consolidated positions are not regularly spaced within a relative
    tolerance of 1% (default `rtol=0.01`). This check ensures uniform voxel spacing,
    which is required for registration and NIfTI export. Non-uniform spacing typically
    indicates a misconfigured sweep.

## Saving

### After Consolidation

Once consolidated, a multi-pose DataArray is a standard ConfUSIus DataArray and can be
saved to any format:

```python
import confusius as cf

anat = cf.load("sub-01_acq-anat_pwd.scan")
volume = cf.multipose.consolidate_poses(anat)

# Save to NIfTI (creates .nii.gz and a JSON sidecar).
volume.fusi.save("sub-01_acq-anat_pwd.nii.gz")

# Or to Zarr.
volume.to_zarr("sub-01_acq-anat_pwd.zarr")
```

### Without Consolidation

Non-consolidated data can be saved to **Zarr** directly, preserving the `pose` dimension
and all per-pose affines:

```python
data.to_zarr("sub-01_acq-anat_pwd_multipose.zarr")
```

Saving non-consolidated data to **NIfTI** is not straightforward because NIfTI stores a
single affine per file. If you need NIfTI output before consolidating (e.g., for per-pose
slice timing correction), save each pose as a separate file:

```python
for i, pose in enumerate(anat.pose.values):
    # The pose entity is defined in the fUSI-BIDS specification.
    anat.sel(pose=pose).fusi.save(f"sub-01_acq-anat_pose-{i:02d}_pwd.nii.gz")
```
