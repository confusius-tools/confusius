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

## Loading Multi-Pose Data

### Iconeus SCAN Files

Iconeus IcoScan stores recordings in **SCAN files** (`.scan`, `.source.scan`). Three
acquisition modes are supported by ConfUSIus:

| Mode | Dimensions | Typical use |
|------|------------|-------------|
| `2Dscan` | `(time, k, j, i)` | Single-pose fUSI time-series |
| `3Dscan` | `(pose, k, j, i)` | Multi-pose anatomical volume |
| `4Dscan` | `(time, pose, k, j, i)` | Multi-pose fUSI time-series (3D+t fUSI) |

Use [`load_scan`][confusius.io.load_scan] to load SCAN files. This page focuses on
**3Dscan** and **4Dscan**. See the [I/O guide](io.md#loading-iconeus-scan-files) for a
general overview of SCAN file loading.

The examples below illustrate a recording from a mouse acquired with an **IcoPrime-4D
MultiArray probe**—four linear probes stacked along the elevation axis, giving 4
elevation slices per pose—translated across multiple regularly spaced positions.

=== "3Dscan (anatomical)"

    ```pycon
    >>> from confusius.io import load_scan
    >>> anat = load_scan("sub-01_acq-anat_pwd.scan")
    >>> anat
    <xarray.DataArray 'scan_data' (pose: 15, k: 4, j: 72, i: 64)> Size: 2MB
    dask.array<transpose, shape=(15, 4, 72, 64), dtype=float64, chunksize=(15, 4, 72, 64), chunktype=numpy.ndarray>
    Coordinates:
      * pose     (pose) int64 120B 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
      * k        (k) int64 32B 0 1 2 3
      * j        (j) int64 576B 0 1 2 3 ... 68 69 70 71
      * i        (i) int64 512B 0 1 2 3 ... 60 61 62 63
        z        (pose, k, j, i) float64 ...
        y        (pose, k, j, i) float64 ...
        x        (pose, k, j, i) float64 ...
    Attributes:
        affines:            {'world_to_lab': ...}  # shape (15, 4, 4)
        scan_mode:          3Dscan
        ...
    ```

    The probe was stepped across 15 positions, each contributing 4 elevation slices —
    a total of 60 slices once consolidated.

=== "4Dscan (functional)"

    ```pycon
    >>> from confusius.io import load_scan
    >>> fus = load_scan("sub-01_task-awake_pwd.scan")
    >>> fus
    <xarray.DataArray 'scan_data' (time: 750, pose: 4, k: 4, j: 72, i: 64)> Size: 442MB
    dask.array<transpose, shape=(750, 4, 4, 72, 64), dtype=float64, chunksize=(227, 4, 4, 72, 64), chunktype=numpy.ndarray>
    Coordinates:
        time       (time, pose) float64 24kB 0.4 2.2 1.0 ... 1.799e+03 1.799e+03
      * pose       (pose) int64 32B 0 1 2 3
      * k          (k) int64 32B 0 1 2 3
      * j          (j) int64 576B 0 1 2 3 ... 68 69 70 71
      * i          (i) int64 512B 0 1 2 3 ... 60 61 62 63
        z          (pose, k, j, i) float64 ...
        y          (pose, k, j, i) float64 ...
        x          (pose, k, j, i) float64 ...
    Attributes:
        affines:            {'world_to_lab': ...}  # shape (4, 4, 4)
        scan_mode:          4Dscan
        ...
    ```

    The probe was stepped across 4 positions, each contributing 4 elevation slices —
    a total of 16 slices once consolidated.

### Other Systems

For other fUSI systems, multi-pose data must be assembled manually: load or construct
one VoxelData array per pose, stack them along a new `pose` dimension,
and attach pose-dependent VoxelData geometry with a `(npose, 4, 4)` voxel-to-world
affine stack.

## World Coordinates and Affines

Native voxel dimensions in a multi-pose VoxelData array remain `k`, `j`, and `i`.
Their derived world coordinates `z`, `y`, and `x` are pose-dependent: selecting a
single `pose` resolves each voxel to a position in the common world space. This
pose-dependent mapping lives in the `VoxelToWorldIndex`, not in separate spatial
dimensions.

For Iconeus SCAN files, [`load_scan`][confusius.io.load_scan] automatically attaches
pose-dependent VoxelData geometry with one voxel-to-world affine per pose.

## Pose-Dependent Timing

When poses are acquired sequentially, each pose is captured at a slightly different
time. Unconsolidated multi-pose data therefore has a pose-dependent `time` coordinate,
shaped `(time, pose)` and holding each pose's own acquisition timestamp directly (mirroring
how `z`/`y`/`x` require a scalar `pose` selection before they resolve to a single position):

```python
fus.coords["time"]  # (time, pose) in seconds.
fus.isel(pose=0).set_xindex("time")  # promote back to a selectable 1D time index.
```

This is important for slice timing correction, which accounts for the fact that different
poses were not acquired simultaneously.

## Pose Consolidation

[`consolidate_poses`][confusius.multipose.consolidate_poses] merges the `pose` dimension
and the sweep spatial dimension into a single axis with physically meaningful
coordinates, producing a VoxelData array.
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

    ```pycon
    >>> import confusius as cf
    >>> anat = cf.load("sub-01_acq-anat_pwd.scan")
    >>> volume = cf.multipose.consolidate_poses(anat)
    >>> volume
    <xarray.DataArray 'scan_data' (k: 60, j: 72, i: 64)> Size: 2MB
    array([...])
    Coordinates:
      * k        (k) int64 480B 0 1 2 3 ... 56 57 58 59
      * j        (j) int64 576B 0 1 2 3 ... 68 69 70 71
      * i        (i) int64 512B 0 1 2 3 ... 60 61 62 63
        z        (k, j, i) float64 ...
        y        (k, j, i) float64 ...
        x        (k, j, i) float64 ...
    Attributes:
        affines:            {'world_to_lab': ...}  # shape (4, 4)
        scan_mode:          3Dscan
        ...
    ```

    15 poses × 4 slices = 60 consolidated z positions, spanning −21.4 to −13.1 mm in
    lab coordinates.

=== "4Dscan (functional)"

    ```pycon
    >>> import confusius as cf
    >>> fus = cf.load("sub-01_task-awake_pwd.scan")
    >>> volume = cf.multipose.consolidate_poses(fus)
    >>> volume
    <xarray.DataArray 'scan_data' (time: 750, k: 16, j: 72, i: 64)> Size: 442MB
    array([...])
    Coordinates:
      * time       (time) float64 6kB 0.4 2.8 5.2 ... 1.793e+03 1.796e+03 1.798e+03
      * k          (k) int64 128B 0 1 2 3 ... 12 13 14 15
        slice_time (time, k) float64 96kB 0.4 2.2 1.0 ... 1.799e+03 1.799e+03
      * j          (j) int64 576B 0 1 2 3 ... 68 69 70 71
      * i          (i) int64 512B 0 1 2 3 ... 60 61 62 63
        z          (k, j, i) float64 ...
        y          (k, j, i) float64 ...
        x          (k, j, i) float64 ...
    Attributes:
        affines:            {'world_to_lab': ...}  # shape (4, 4)
        scan_mode:          4Dscan
        ...
    ```

    4 poses × 4 slices = 16 consolidated z positions. The pose-dependent `time`
    coordinate is reduced to a single whole-volume `time` and a `slice_time`
    coordinate with dims `(time, k)`: each slice retains the timestamp of the pose it
    came from.

After consolidation, the per-pose affine stack is reduced to a single `(4, 4)` matrix
representing the consolidated volume's orientation in world space.

### Parameters

[`consolidate_poses`][confusius.multipose.consolidate_poses] always reads per-pose
positions from `da`'s primary voxel-to-world geometry, which must therefore itself be
pose-dependent (a `(npose, 4, 4)` affine stack — see
[VoxelToWorldIndex.is_pose_dependent][confusius._utils.geometry.VoxelToWorldIndex.is_pose_dependent]).
If you instead want to consolidate around a different, secondary affine linked in
`da.attrs["affines"]` (e.g. `world_to_brain` alongside a `world_to_lab`-equivalent
primary), rebase onto it first with
[`.fusi.affine.apply`][confusius.xarray.FUSIAffineAccessor.apply]:

```python
# Example: sweeping along native voxel dimension i, consolidating around a secondary affine.
volume = cf.multipose.consolidate_poses(
    da.fusi.affine.apply("world_to_scanner"),
    sweep_dim="i",
)
```

Adjust **`sweep_dim`** (default: `"k"`) if your sweep is along a different voxel
dimension.

!!! warning "Regularity requirement"
    [`consolidate_poses`][confusius.multipose.consolidate_poses] will raise a
    `ValueError` if the consolidated positions are not regularly spaced within a relative
    tolerance of 1% (default `rtol=0.01`). This check ensures uniform voxel spacing,
    which is required for registration and NIfTI export. Non-uniform spacing typically
    indicates a misconfigured sweep.

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
