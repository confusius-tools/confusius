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
transformations that record the world position of each pose.

!!! warning "Rotational sweeps are not yet supported"
    [`consolidate_poses`][confusius.multipose.consolidate_poses] requires a
    **purely translational** sweep, where the probe is shifted along one axis without
    rotating. Rotational sweeps (so-called tomographic acquisitions) are not yet
    supported and will raise a `ValueError`.

## The Multi-Pose Data Model

A multi-pose VoxelData array follows the same dimension order as any other VoxelData
array, `(..., time, pose, k, j, i)`, with `pose` sitting between `time` and the native
voxel dimensions `k`/`j`/`i`. This reflects the fact that probe poses are typically
acquired sequentially during a single recording.

`pose` is not just an extra axis: it changes what the `z`/`y`/`x` world coordinates and
the `time` coordinate mean, because both are per-pose rather than per-array.

- **Geometry is per-pose.** Each pose has its own voxel-to-world affine, so the world
  coordinates `z`/`y`/`x` are pose-dependent with shape `(pose, k, j, i)`. Resolving a
  voxel to a single world position requires selecting a scalar `pose` first, e.g.
  `data.isel(pose=0).sel(z=1.2)`. See [Spatial Conventions](spatial-conventions.md#the-voxeldata-model)
  for the full explanation of pose-dependent voxel-to-world geometry.
- **Timing can be per-pose.** When poses are acquired sequentially rather than
  simultaneously, each pose is captured at a slightly different time. The `time`
  coordinate then holds each pose's own acquisition timestamp directly, shaped
  `(time, pose)` instead of the usual 1D `(time,)`—mirroring how `z`/`y`/`x` require a
  scalar `pose` selection before they resolve to a single value:

    ```python
    pwd.time  # (time, pose) in seconds.
    pwd.isel(pose=0).time  # (time,) in seconds, for pose 0 only.
    ```

    This matters for slice timing correction (see [Slice Timing
    Correction](#slice-timing-correction) below), which accounts for the fact that
    different poses were not acquired simultaneously.

These two rules—pose-dependent world coordinates, pose-dependent timing—drive
everything else in this guide: how multi-pose data is loaded and built, what you can
and cannot do with it directly, and what consolidation resolves.

## Loading and Constructing Multi-Pose Data

### Iconeus SCAN Files

When loading Iconeus SCAN files containing multi-pose data (`3Dscan` or `4Dscan`
acquisition modes), the resulting VoxelData array has a `pose` dimension. The example
below illustrates a recording from a mouse acquired with an **IcoPrime-4D MultiArray
probe**—four linear probes stacked along the elevation axis, giving 4 elevation slices
per pose—translated across multiple regularly spaced positions:

```pycon
>>> from confusius.io import load_scan
>>> pwd = load_scan("sub-01_task-awake_pwd.scan")
>>> pwd
<xarray.DataArray 'scan_data' (time: 409, pose: 4, k: 4, j: 92, i: 118)> Size: 568MB
dask.array<transpose, shape=(409, 4, 4, 92, 118), dtype=float64, chunksize=(106, 4, 4, 92, 106), chunktype=numpy.ndarray>
Coordinates:
    time       (time, pose) float64 13kB 0.4 2.05 0.95 1.5 ... 899.7 898.6 899.1
  * pose       (pose) int64 32B 0 1 2 3
  * k          (k) int64 32B 0 1 2 3
  * j          (j) int64 736B 0 1 2 3 ... 89 90 91
  * i          (i) int64 944B 0 1 2 3 ... 116 117
    z          (pose, k, j, i) float64 1MB ...
    y          (pose, k, j, i) float64 1MB ...
    x          (pose, k, j, i) float64 1MB ...
Attributes:
    affines:               {}
    device_serial_number:  ASAO0000
    software_version:      IcoScan v.1.9.0
    iconeus_scan_mode:     4Dscan
    iconeus_subject:       sub-01
    iconeus_session:       task-awake
    iconeus_scan:          sub-01_task-awake_pwd
    ...
```

The probe was stepped across 4 positions, each contributing 4 elevation slices—a total
of 16 slices once consolidated.

### Reloading from Zarr

Saving unconsolidated multi-pose data to **Zarr** preserves the `pose` dimension and the
per-pose affine stack (see [Saving](#saving) below). [`cf.load`][confusius.load]
round-trips this transparently: reloading a Zarr store written by
[`cf.save`][confusius.save] rebuilds the pose-dependent `VoxelToWorldIndex` from
`attrs["voxel_to_world"]`, so a reloaded array is indistinguishable from the one that
was saved.

```python
import confusius as cf

pwd = cf.load("sub-01_task-awake_pwd_multipose.zarr")  # dims: (time, pose, k, j, i)
```

### Other Systems

For fUSI systems without a dedicated loader, multi-pose data must be assembled manually,
in one of two ways:

- **From a raw array**, using [`create_voxeldata`][confusius.xarray.create_voxeldata]
  with a `pose` dimension, an `(npose, 4, 4)` `voxel_to_world` affine stack, and, if
  poses were acquired sequentially, a 1D `t0` array giving each pose's own time origin:

    ```python
    import confusius as cf

    multipose = cf.create_voxeldata(
        raw_power,  # shape: (i, j, k, pose, time)
        dims=("i", "j", "k", "pose", "time"),
        dt=0.6,  # seconds, shared across poses.
        t0=[0.0, 0.15, 0.3, 0.45],  # per-pose time origin, in seconds.
        voxel_to_world=pose_affines,  # shape: (4, 4, 4)
    )
    ```

- **From independently loaded single-pose arrays**, using
  [`stack_poses`][confusius.multipose.stack_poses] to combine them into one
  pose-dependent array.

    ```python
    from confusius.multipose import stack_poses

    poses = [cf.load(f"sub-01_pose-{i:02d}_pwd.nii.gz") for i in range(4)]
    multipose = stack_poses(poses)  # dims: (time, pose, k, j, i)
    ```

    If the individual poses were acquired sequentially rather than simultaneously and
    carry different `time` values, [`stack_poses`][confusius.multipose.stack_poses]
    automatically produces a pose-dependent `(time, pose)` coordinate instead of a
    shared 1D `time`.

## Working with Multi-Pose Arrays

Most Xarray operations that don't touch world coordinates or timing work on multi-pose
arrays exactly as on any other VoxelData array—reductions, arithmetic, `.isel`, mask
extraction, and so on all treat `pose` as an ordinary dimension.

Operations that resolve *positions* or *timestamps*, however, need a scalar `pose`
first, per [The Multi-Pose Data Model](#the-multi-pose-data-model) above:

```python
pose0 = pwd.sel(pose=0)
shallow = pose0.sel(z=slice(0, 2.5))       # world-coordinate selection now works.
subset = pose0.sel(time=slice(10, 60))     # time is now 1D, .sel(time=...) works too.
```

A number of ConfUSIus functions go further and require multi-pose data to be reduced
to a single spatial grid entirely, since they operate on one voxel-to-world affine:

| Operation | Requires |
|---|---|
| Registration ([registration guide](registration.md)) | Single pose, or [consolidated](#pose-consolidation) data |
| Saving to NIfTI (single affine per file) | Single pose, or consolidated data—see [Saving](#saving) |

## Slice Timing Correction

Because poses in a sequential acquisition are not captured simultaneously, comparing
voxels across poses at the same nominal `time` index compares samples taken at slightly
different real times. [`correct_slice_timings`][confusius.multipose.correct_slice_timings]
resamples each pose's time series onto a shared, per-volume reference time, removing
this offset. It accepts either unconsolidated data with a pose-dependent `(time, pose)`
`time` coordinate, or already-[consolidated](#pose-consolidation) data with a
`slice_time` coordinate:

```python
from confusius.multipose import correct_slice_timings

corrected = correct_slice_timings(pwd)  # time coordinate becomes a plain 1D index.
```

## Pose Consolidation

[`consolidate_poses`][confusius.multipose.consolidate_poses] merges the `pose` dimension
and the sweep spatial dimension into a single axis with physically meaningful
coordinates, producing a VoxelData array with a single, non-pose-dependent
voxel-to-world affine. [`consolidate_poses`][confusius.multipose.consolidate_poses]
performs the following steps:

1. Detect the swept voxel dimension (`sweep_dim`) from the per-pose affines: match
   the pose-translation direction against each voxel dimension's world-space
   direction, and take the best-aligned one.
2. Read the per-pose affines to compute the world position of every `(pose,
   sweep_dim)` voxel.
3. Find the precise sweep direction via SVD of all these voxel positions, and
   project each voxel onto that axis.
4. Check that the resulting positions form a regular grid.
5. Reindex the data in ascending position order, replacing `pose` and `sweep_dim`
   with a single consolidated coordinate in world space.

```pycon
>>> import confusius as cf
>>> pwd = cf.load("sub-01_task-awake_pwd.scan")
>>> volume = cf.multipose.consolidate_poses(pwd)
>>> volume
<xarray.DataArray 'scan_data' (time: 409, k: 16, j: 92, i: 118)> Size: 568MB
array([...])
Coordinates:
  * time        (time) float64 3kB 2.05 4.25 6.45 8.65 ... 895.2 897.5 899.7
  * k           (k) int64 128B 0 1 2 3 ... 13 14 15
    slice_time  (time, k) float64 52kB 0.4 2.05 0.95 1.5 ... 899.7 898.6 899.1
  * j           (j) int64 736B 0 1 2 3 ... 89 90 91
  * i           (i) int64 944B 0 1 2 3 ... 116 117
    z           (k, j, i) float64 1MB ...
    y           (k, j, i) float64 1MB ...
    x           (k, j, i) float64 1MB ...
Attributes:
    affines:               {}
    device_serial_number:  ASAO0000
    software_version:      IcoScan v.1.9.0
    iconeus_scan_mode:     4Dscan
    iconeus_subject:       sub-01
    iconeus_session:       task-awake
    iconeus_scan:          sub-01_task-awake_pwd
    ...
```

4 poses × 4 slices = 16 consolidated z positions, spanning −4.95 to 2.93 mm in world
space. The pose-dependent `time` coordinate is reduced to a single whole-volume `time`
and a `slice_time` coordinate with dims `(time, k)`: each slice retains the timestamp
of the pose it came from.

After consolidation, the per-pose affine stack is reduced to a single `(4, 4)` matrix
representing the consolidated volume's orientation in world space.

!!! warning "Regularity requirement"
    [`consolidate_poses`][confusius.multipose.consolidate_poses] will raise a
    `ValueError` if the consolidated positions are not regularly spaced within a
    relative tolerance of 1% (default `rtol=0.01`). This check ensures uniform voxel
    spacing, which is required for registration and NIfTI export. Non-uniform spacing
    typically indicates motor positioning errors.

## Saving

### After Consolidation

Once consolidated, a multi-pose array is still a VoxelData array and can be
saved to any format:

```python
import confusius as cf

anat = cf.load("sub-01_acq-anat_pwd.scan")
volume = cf.multipose.consolidate_poses(anat)

# Save to NIfTI (creates .nii.gz and a matching fUSI-BIDS JSON sidecar).
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
