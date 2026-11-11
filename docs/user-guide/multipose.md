---
icon: lucide/waypoints
---

# Multi-Pose Data

## What is Multi-Pose Imaging?

In **multi-pose fUSI**, the ultrasound probe moves through a series of positions (or
**poses**). At each pose, it acquires one or more images. Combining these images extends
the field of view beyond a single probe position.

Multiple fUSI systems support this approach, including Iconeus, EchoFrame, and AUTC.
ConfUSIus represents multi-pose data with a `pose` dimension, each pose having its own
time coordinate and spatial geometry.

## The Multi-Pose Data Model

Multi-pose arrays follow the usual VoxelData dimension order, with pose between time and
the voxel dimensions: `(..., time, pose, k, j, i)`. Unlike an ordinary dimension, pose
affects both spatial coordinates and acquisition times:

- **Geometry**: Each pose has its own voxel-to-world affine. World coordinates `(z, y,
  x)` therefore have dimensions `(pose, k, j, i)`. Select a pose before selecting by
  world position, for example `data.isel(pose=0).sel(z=1.2)`. See [The VoxelData
  Model](voxeldata.md#the-voxeldata-model).
- **Timing**: When poses are acquired sequentially, their timestamps differ. The time
  coordinate then has dimensions `(time, pose)` rather than `(time,)`. Selecting one
  pose gives its 1D time coordinate:

    ```python
    pwd.time  # (time, pose) in seconds.
    pwd.isel(pose=0).time  # (time,) in seconds, for pose 0 only.
    ```

These differences matter when selecting data by position or time and when processing the
recording as a single volume. We'll cover these topics in the sections below.

## Loading and Constructing Multi-Pose Data

### Iconeus SCAN Files

ConfUSIus supports both SCAN v1 (HDF5) and SCAN v2 (binary). They store multi-pose
recordings differently.

#### SCAN v1 (HDF5)

Loading a multi-pose `3Dscan` or `4Dscan` recording retains the `pose` dimension.
The following example recording was acquired with the IcoPrime-4D MultiArray probe,
which consists of four stacked linear arrays. This recording used a sequential
acquisition, moving the probe to four different positions.

```pycon
>>> import confusius as cf
>>> pwd = cf.load("sub-01_task-awake_pwd.scan")
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
    affines:               {'world_to_lab': array([[1., 0., 0., 0.],
       ...
    device_serial_number:  ASAO0000
    software_version:      IcoScan v.1.9.0
    iconeus_scan_mode:     4Dscan
    iconeus_subject:       sub-01
    iconeus_session:       task-awake
    iconeus_scan:          sub-01_task-awake_pwd
    ...
```

[Consolidating](#pose-consolidation) the four poses would therefore produce a 3D+t
recording with **16 elevation slices**.

#### SCAN v2 (binary)

IcoScan generally writes multi-pose SCAN v2 recordings as already-consolidated volumes,
when possible. [`confusius.load`][confusius.load] then returns an array **without
`pose`**:

```pycon
>>> import confusius as cf
>>> pwd = cf.load("sub-02_task-awake_pwd.scan")
>>> pwd
<xarray.DataArray 'sub-02_task-awake_pwd' (time: 1000, k: 16, j: 92, i: 102)> Size: 1GB
dask.array<getitem, shape=(1000, 16, 92, 102), dtype=float64, chunksize=(111, 16, 92, 102), chunktype=numpy.ndarray>
Coordinates:
  * time        (time) float64 8kB 2.2 4.6 7.0 ... 2.395e+03 2.397e+03 2.4e+03
  * k           (k) int64 128B 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    slice_time  (time, k) float64 128kB 0.4 2.2 1.0 ... 2.399e+03 2.399e+03
  * j           (j) int64 736B 0 1 2 3 4 5 6 7 8 ... 83 84 85 86 87 88 89 90 91
  * i           (i) int64 816B 0 1 2 3 4 5 6 7 8 ... 94 95 96 97 98 99 100 101
  * z           (k, j, i) float64 1MB ...
  * y           (k, j, i) float64 1MB ...
  * x           (k, j, i) float64 1MB ...
Attributes:
    probe_model:                         IcoPrime 4D MultiArray
    iconeus_scan_format:                 v2
    iconeus_acquisition_mode:            4Dscan
    ...
```

Here, the 16 slices come from the four linear arrays spaced 2.1 mm apart and four motor
steps spaced 0.525 mm apart. A full volume takes 2.4 s to acquire.

- `time` gives the end time of each complete volume (as is the convention in Iconeus
  SCAN v2).
- `slice_time`, with dimensions `(time, k)`, gives the end time of each slice.

The four arrays acquire simultaneously, so slices separated by four positions share a
timestamp. In the first volume, the timestamps are 0.4, 2.2, 1.0, and 1.6 s, repeated
four times. [Slice timing correction](#slice-timing-correction) can align these slices
in time.

### Reloading from Zarr

Zarr stores preserve the `pose` dimension, per-pose geometry, and timing.
[`confusius.load`][confusius.load] restores the pose-dependent voxel-to-world mapping when
loading data saved with [`confusius.save`][confusius.save]:

```python
import confusius as cf

pwd = cf.load("sub-01_task-awake_pwd_multipose.zarr")
# Dimensions: (time, pose, k, j, i)
```

### Multi-Pose NIfTI Files

Recordings stored as separate NIfTI files can be loaded as a multi-pose array using the
`{chunk}` filename pattern. This is particularly useful for fUSI-BIDS datasets, where
multi-pose recordings are often split into one NIfTI file per pose, distinguished by the
`chunk` entity.

```python
import confusius as cf

multipose = cf.load("sub-01_acq-anat_chunk-{chunk}_pwd.nii.gz")
```

Matching integer chunks are sorted numerically and used as `pose` labels. Files must
have equal numbers of time points and compatible voxel grids. See the [Load and save
multi-pose recordings
example](../examples/_built/multipose/load_multipose_recordings.md) for a complete
workflow.

### Other Formats

For other formats, you can build a multi-pose array in either of two ways.

- **From a raw array:** Use [`create_voxeldata`][confusius.xarray.create_voxeldata] with
  a `pose` dimension, one `(4, 4)` affine per pose, and optional per-pose time origins
  (`t0`):

    ```python
    multipose = cf.create_voxeldata(
        raw_power,  # Shape: (i, j, k, pose, time)
        dims=("i", "j", "k", "pose", "time"),
        dt=0.6,  # Seconds between samples
        t0=[0.0, 0.15, 0.3, 0.45],  # Time origin for each pose, in seconds
        voxel_to_world=pose_affines,  # Shape: (4, 4, 4)
    )
    ```

- **From a list of single-pose arrays:** Use
  [`stack_poses`][confusius.multipose.stack_poses]:

    ```python
    from confusius.multipose import stack_poses

    poses = [cf.load(f"pose-{p}.nii.gz") for p in ("one", "two", "three", "four")]
    multipose = stack_poses(poses)
    # Dimensions: (time, pose, k, j, i)
    ```

  If the poses have different timestamps, `stack_poses` automatically creates a `(time,
  pose)` time coordinate instead of a shared 1D coordinate.

## Working with Multi-Pose Arrays

Most Xarray operations, including arithmetic, reductions, indexing with `.isel`, and
mask extraction, work directly on multi-pose arrays.

For [napari visualization](visualization.md#interactive-exploration-with-napari), each
pose appears as a separate layer, positioned using its own affine:

```python
viewer, layers = pwd.fusi.plot.napari()
```

Layers are named `<data name> [pose <value>]`. The CLI behaves the same way.

Operations that resolve *positions* or *timestamps*, however, need a scalar `pose`
first, per [The Multi-Pose Data Model](#the-multi-pose-data-model) above:

```python
pose0 = pwd.sel(pose=0)
shallow = pose0.sel(z=slice(0, 2.5))
subset = pose0.sel(time=slice(10, 60))
```

Operations requiring a single spatial grid, such as [registration](registration.md),
need either a single pose or a [consolidated volume](#pose-consolidation). Similarly, a
single NIfTI file cannot store multiple pose-dependent affines; use separate `{chunk}`
files or consolidate first.

## Slice Timing Correction

In sequential acquisitions, different poses (or slices) are measured at different times.
Comparing them at the same volume index can therefore mix signals acquired at different
moments.

[`correct_slice_timings`][confusius.multipose.correct_slice_timings] resamples these
time series onto a common time coordinate. It works with either:

- **Unconsolidated arrays** with a `(time, pose)` `time` coordinate.
- **Consolidated arrays** with a `(time, k)` `slice_time` coordinate.

```python
from confusius.multipose import correct_slice_timings

corrected = correct_slice_timings(pwd)
# Time is now a shared 1D coordinate.
```

## Pose Consolidation

When poses are acquired sequentially and along a single translational axis, they can be
combined into a single volume.
[`consolidate_poses`][confusius.multipose.consolidate_poses] combines the `pose`
dimension with the spatial dimension along which the probe moved. The result is a single
regular voxel grid with one voxel-to-world affine.

!!! warning "Rotational sweeps are not yet supported"
    [`consolidate_poses`][confusius.multipose.consolidate_poses] requires a
    **purely translational** sweep, where the probe is shifted along one axis without
    rotating. Rotational sweeps (e.g., tomographic acquisitions) are not yet supported
    and will raise a `ValueError`.

It does this by:

1. Identifying the sweep dimension from the probe translations and voxel-axis
   directions.
2. Computing the world positions of the slices from the per-pose affines.
3. Estimating the sweep direction using SVD, then projecting slice positions onto it.
4. Checking that the positions form a regularly spaced grid.
5. Sorting the slices by position and combining `pose` and the sweep dimension into one
   spatial axis.

For the SCAN v1 example above:

```pycon
>>> import confusius as cf
>>> pwd = cf.load("sub-01_task-awake_pwd.scan")
>>> volume = cf.multipose.consolidate_poses(pwd)
>>> volume
<xarray.DataArray 'scan_data' (time: 409, k: 16, j: 92, i: 118)>
Coordinates:
  * time        (time) ...
  * k           (k) 0 1 2 ... 15
    slice_time  (time, k) ...
    z           (k, j, i) ...
    y           (k, j, i) ...
    x           (k, j, i) ...
```

The four poses, each with four slices, become **16 elevation positions spanning −3.85 to
4.02 mm**. The new 1D `time` coordinate records the time of each complete volume,
while `slice_time` retains each slice's original acquisition time. The per-pose affine
stack becomes one `(4, 4)` affine for the combined volume.

!!! warning "Regular spacing is required"
    Consolidation raises a `ValueError` if slice positions are not regularly spaced
    within the default relative tolerance of 1% (`rtol=0.01`). A regular grid is
    required for registration and NIfTI export. Irregular spacing can indicate motor
    positioning errors.

## Saving

### After Consolidation

Consolidated data can be saved like any other VoxelData array:

```python
import confusius as cf

anat = cf.load("sub-01_acq-anat_pwd.scan")
volume = cf.multipose.consolidate_poses(anat)

# NIfTI, with a matching fUSI-BIDS JSON sidecar.
volume.fusi.save("sub-01_acq-anat_pwd.nii.gz")

# Or Zarr.
volume.to_zarr("sub-01_acq-anat_pwd.zarr")
```

### Without Consolidation

**Zarr** preserves the `pose` dimension and its per-pose geometry:

```python
anat.to_zarr("sub-01_acq-anat_pwd_multipose.zarr")
```

For **NIfTI**, include `{chunk}` in the filename to save one NIfTI file and JSON sidecar
per pose:

```python
anat.fusi.save("sub-01_acq-anat_chunk-{chunk}_pwd.nii.gz")

# Reload all poses together
anat = cf.load("sub-01_acq-anat_chunk-{chunk}_pwd.nii.gz")
```

The `pose` labels must be unique, non-negative integers. The `{chunk}` placeholder must
appear exactly once in the filename (not in a directory name). Without it, saving to
NIfTI requires selecting one pose or consolidating the array first.
