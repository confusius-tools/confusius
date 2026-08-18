---
icon: lucide/brackets
---

# Working with Xarray

## Why Xarray?

A typical fUSI recording is a 4D array indexed by time and native voxel dimensions (`i`,
`j`, `k`). World coordinates corresponding to the voxel dimensions are (`x`, `y`, `z`).
Storing this as a plain NumPy array means losing all of that structure: axes become
anonymous integers, world coordinates must be tracked separately, and keeping them in
sync with the data after slicing or averaging is error-prone. A custom wrapper class
would address the labeling and coordinate tracking, but at the cost of needing a complex
reimplementation of many array operations and losing access to the broader scientific
Python ecosystem.

[Xarray](https://xarray.dev/) solves this by wrapping arrays with named dimensions and
world coordinates, while [maintaining compatibility](https://xarray.dev/#ecosystem)
with the Python scientific ecosystem. With Xarray, operations become self-documenting:

```python
# NumPy: what does axis 0 mean here?
mean_volume = pwd.mean(axis=0)

# Xarray: unambiguous.
mean_volume = pwd.mean("time")
```

Coordinates also carry metadata through transformations, so voxel sizes, timestamps, and
acquisition parameters travel with the data rather than being stored separately.

```python
# Select a depth range by world coordinate.
shallow = pwd.sel(z=slice(1, 2.5))

# Coordinates are updated automatically, no manual bookkeeping needed.
shallow.y
```

ConfUSIus leverages Xarray to provide a robust data model for fUSI recordings while
staying interoperable with other scientific Python libraries. Thus, ConfUSIus can add
fUSI-specific conventions and functionality without trapping the data in a
ConfUSIus-specific object.

## DataArrays and Datasets

Xarray has two core data structures: **DataArrays** and **Datasets**.

A **[DataArray][xarray.DataArray]** is a single array with its own dimensions,
coordinates, and attributes. You get a DataArray whenever you load a single fUSI
recording, for example a power Doppler NIfTI file from the [Nunez-Elizalde 2022
dataset][confusius.datasets.fetch_nunez_elizalde_2022] via
[`confusius.load`][confusius.load]:

```pycon
>>> import confusius as cf
>>>
>>> pwd = cf.load("sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd.nii.gz")
>>> pwd
<xarray.DataArray 'sub-CR022_ses-20201011_task-spontaneous_acq-slice03_pwd'
    (time: 751, k: 1, j: 114, i: 80)> Size: 27MB
dask.array<transpose, shape=(751, 1, 114, 80), dtype=float32, chunksize=(751, 1, 114, 80)>
Coordinates:
  * time     (time) float64 6kB 10.61 10.91 11.21 ... 235.1 235.4 235.7
  * k        (k) float64 8B 0.0
  * j        (j) float64 912B 0.0 1.0 2.0 ... 112.0 113.0
  * i        (i) float64 640B 0.0 1.0 2.0 ... 78.0 79.0
  * z        (k, j, i) float64 73kB 1.0 1.0 1.0 ... 1.0 1.0
  * y        (k, j, i) float64 73kB 2.73 2.73 2.73 ... 8.19 8.19
  * x        (k, j, i) float64 73kB -3.95 -3.85 -3.75 ... 3.85 3.95
Indexes:
  ┌ z        VoxelToWorldIndex
  │ y
  └ x
Attributes: (12/24)
    qform_code:                1
    manufacturer:              Verasonics
    manufacturers_model_name:  Vantage 128
    software_version:          Alan Urban Technology & Consulting (AUTC)
    probe_manufacturer:        Vermon
    probe_type:                linear
    ...                        ...
    task_description:          Spontaneous activity without explicit visual stimulation.
    depth:                     [0.0, 5.46016]
    transmit_frequency:        15625000.0
    compound_sampling_frequency: 500.0
    plane_wave_angles:         [-10.0, -7.9, -5.8, -3.7, -1.6, 0.5, 2.6, 4.7, 6.8, 8.9]
    probe_voltage:             25.0
```

Reading the output from top to bottom, a DataArray has four components:

- **Dimensions** `(time, k, j, i)`: native voxel axes in the order they appear in the
  underlying array. `time` is the temporal axis; `i`, `j`, and `k` index the voxel grid.

    !!! question "Why `(time, k, j, i)` instead of `(i, j, k, time)`?"
        Neuroimaging formats such as NIfTI conventionally order axes `(x, y, z, time,
        ...)`, with the first axis varying fastest in storage. NumPy uses the opposite
        convention: its default memory layout has the last axis varying fastest.
        ConfUSIus therefore uses `(..., time, k, j, i)`, which maps NIfTI data naturally
        onto the memory layout used throughout the Python scientific ecosystem, often
        without copying or rearranging the data. In practice, this ordering is usually
        transparent because Xarray operations refer to dimensions by name rather than by
        position. See [Spatial Conventions](spatial-conventions.md) for the full
        explanation.

- **Data**: the underlying array. ConfUSIus loaders return
  [Dask](https://www.dask.org/)-backed data, meaning values are not loaded into memory
  until you explicitly request them (e.g., by calling `.compute()` or accessing
  `.values`).
- **Coordinates**: timestamps and world positions, typically in seconds and millimeters.
  The world coordinates `x`, `y`, and `z` are derived from the voxel-to-world geometry
  (via the `VoxelToWorldIndex` shown under `Indexes`) and enable slicing to work in
  physical units rather than array indices.
- **Attributes**: acquisition metadata as a flat key-value dictionary. Attributes are
  preserved through most ConfUSIus operations, and some are required for certain
  functions (for example, `transmit_frequency` is needed for velocity calculations).

A **[Dataset][xarray.Dataset]** is a dictionary-like container of multiple DataArrrays
that share some dimensions and coordinates. It shows up when a source naturally groups
several co-registered variables together, rather than one array per file. An atlas is a
good example: loading one with
[`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas] gives you a
Dataset with `reference`, `annotation`, and `hemispheres` variables sharing the same
voxel grid and world coordinates:

```pycon
>>> atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um")
>>> atlas
<xarray.Dataset> Size: 40MB
Dimensions:      (k: 132, j: 80, i: 114)
Coordinates:
  * k            (k) float64 1kB 0.0 1.0 2.0 ... 130.0 131.0
  * j            (j) float64 640B 0.0 1.0 2.0 ... 78.0 79.0
  * i            (i) float64 912B 0.0 1.0 2.0 ... 112.0 113.0
  * z            (k, j, i) float64 10MB 0.0 0.0 0.0 ... 13.1 13.1
  * y            (k, j, i) float64 10MB 0.0 0.0 0.0 ... 7.9 7.9
  * x            (k, j, i) float64 10MB 0.0 0.1 0.2 ... 11.2 11.3
Data variables:
    reference    (k, j, i) float32 5MB 0.0 0.0 0.0 ... 1.0 1.0
    annotation   (k, j, i) int32 5MB 0 0 0 0 0 ... 0 0 0 0 0
    hemispheres  (k, j, i) int8 1MB 2 2 2 2 2 ... 1 1 1 1 1
Indexes:
  ┌ z        VoxelToWorldIndex
  │ y
  └ x
Attributes:
    name:        allen_mouse
    citation:    Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007
    species:     Mus musculus
    orientation: asr
```

ConfUSIus mostly operates on DataArray objects. Datasets are a convenient way to group
several DataArrays together when they naturally belong together, as with the atlas
above—pull out a single variable (e.g. `atlas["reference"]`) to get a DataArray back.

## Basic Operations

!!! question "New to Xarray?"
    If you are not yet familiar with Xarray, the [Xarray quick
    overview](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html)
    is the best place to start. Understanding indexing, selection, and broadcasting will
    make working with ConfUSIus much easier.

A DataArray behaves like a NumPy array in most respects. Arithmetic and broadcasting
work as usual, and thanks to NumPy's [array
protocol](https://numpy.org/doc/stable/reference/arrays.classes.html), NumPy functions
can be called directly on a DataArray and return a DataArray back:

```python
pwd_sqrt = np.sqrt(pwd)  # np.sqrt dispatches to Xarray, returns a DataArray.
```

The main difference is that reductions and indexing use dimension names instead of
axis positions:

```python
mean_volume = pwd.mean("time")  # reduction by name, not axis=0.
```

Indexing comes in two flavors: [`.isel`][xarray.DataArray.isel] indexes by integer
position, like plain NumPy indexing, while [`.sel`][xarray.DataArray.sel] indexes by
coordinate value:

```python
first_50_volumes = pwd.isel(time=slice(0, 50))  # first 50 volumes, by position.
shallow = pwd.sel(y=slice(0, 2.5))              # depth 0-2.5 mm, by coordinate.
```

Scalar indexing (e.g. `.isel(k=0)`) drops the indexed dimension but keeps its
coordinate as a scalar:

```python
slice_movie = pwd.isel(k=0)
slice_movie.dims
# ('time', 'j', 'i')

slice_movie.coords["z"]
# scalar world coordinate: z = 0.0 mm, with the original coordinate metadata
```

Most geometry-sensitive ConfUSIus functions automatically restore such scalar-indexed
spatial coordinates as singleton dimensions before validating the data. For example,
`slice_movie` is treated as `(time, k, j, i)` with `k=1` when passed to registration or
resampling APIs. Dimension-generic operations such as smoothing preserve the indexed
shape.

## Creating VoxelData-compatible DataArrays from Raw Arrays

Use [`create_fusi_dataarray`][confusius.xarray.create_fusi_dataarray] when you already
have a NumPy, Dask, or array-like object and want to create a VoxelData-compatible
DataArray. This is useful when you have raw data from a custom acquisition system or a
non-standard file format. The function will attach VoxelData dimensions, coordinates,
and metadata. Dimensions can be supplied in any order; the result is canonicalized to
native `(time, k, j, i)` order:

```python
import confusius as cf

recording = cf.create_fusi_dataarray(
    raw_power,  # shape: (time, k, j, i)
    dims=("time", "k", "j", "i"),
    dt=0.6,  # seconds
    spacing=(0.4, 0.05, 0.1),  # world spacing in z/y/x order, in mm.
    attrs={"description": "Power Doppler from my system"},
)
```

Single-slice recordings can be provided by omitting the relevant voxel dimension in the
`dims` argument. ConfUSIus will automatically add the missing singleton dimension and
its corresponding world coordinate. Note that you must still provide the world spacing
for the missing dimension in `spacing`, since a fUSI slice still has a physical
thickness in the missing dimension.

```python
single_slice = cf.create_fusi_dataarray(
    raw_power,  # shape: (time, j, i)
    dims=("time", "j", "i"),
    dt=0.6,
    spacing=(0.4, 0.05, 0.1),  # world spacing in z/y/x order, in mm.
)
```

Acquisition metadata that describes the whole recording belongs in the DataArray
`attrs`. Coordinate metadata such as `units` and `voxdim` is added automatically.

For beamformed IQ data, use
[`create_iq_dataarray`][confusius.xarray.create_iq_dataarray] instead. It use
[`create_fusi_dataarray`][confusius.xarray.create_fusi_dataarray] under the hood, but
also adds IQ-specific metadata such as `transmit_frequency` and
`beamforming_sound_velocity`:

```python
iq = cf.create_iq_dataarray(
    raw_iq,  # shape: (time, j, i)
    dims=("time", "j", "i"),
    dt=1 / 500,
    spacing=(0.4, 0.05, 0.1),  # world spacing in z/y/x order, in mm.
    transmit_frequency=15.625e6,
    beamforming_sound_velocity=1540.0,
)
```

## The `.fusi` Accessor

Most of the functions you have seen so far (`cf.load`, `create_fusi_dataarray`, etc.)
are module-level, imported explicitly from `confusius`. An **accessor** is Xarray's
mechanism for attaching a custom namespace directly to a DataArray or Dataset instead,
so related functionality is reachable straight off the data—for example,
`pwd.fusi.scale.db()`. This also keeps the boundary between Xarray's own API and
library-specific functionality explicit; see [Xarray's guide to extending
Xarray](https://docs.xarray.dev/en/stable/internals/extending-xarray.html) for the
general mechanism.

ConfUSIus registers two such accessors: `.fusi` on DataArrays, and `.atlas` on Datasets.
`.atlas` is useful on atlas Datasets such as the one returned by
[`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas], and is covered
separately in the [Atlases guide](atlas.md). The rest of this section covers `.fusi`.

Importing ConfUSIus registers the accessors automatically:

```python
import xarray as xr
import confusius as cf  # Registers the .fusi and .atlas accessors.
```

The `.fusi` accessor is organized into focused sub-accessors, plus a set of global
helper properties:

| Accessor | Description |
|---|---|
| [`.fusi.save`][confusius.xarray.FUSIAccessor.save] | Save data to file (NIfTI or Zarr), dispatching by extension. |
| [`.fusi.iq`][confusius.xarray.FUSIIQAccessor] | Process beamformed IQ into power Doppler or axial velocity volumes. |
| [`.fusi.scale`][confusius.xarray.FUSIScaleAccessor] | Scaling transformations: decibel, log, and power scaling. |
| [`.fusi.affine`][confusius.xarray.FUSIAffineAccessor] | Inspect and apply voxel-to-world and world-to-reference affine transforms. |
| [`.fusi.register`][confusius.xarray.FUSIRegistrationAccessor] | Motion correction via volumewise image registration. |
| [`.fusi.extract`][confusius.xarray.FUSIExtractAccessor] | Extract and reconstruct signals using spatial masks. |
| [`.fusi.plot`][confusius.xarray.FUSIPlotAccessor] | Visualization with napari and carpet plots. |
| [`.fusi.connectivity`][confusius.xarray.FUSIConnectivityAccessor] | Seed-based functional connectivity maps. |

The sub-accessors offer the same functions as the module-level API, but with an
intuitive syntax that allows quick operations directly on DataArray objects. They are
designed to be used for easy exploration and quick analyses, while the module-level
functions are available for more complex workflows where you might prefer explicit
function calls for readability.

### Global Helpers

Currently, three global helpers are available:

- [`.fusi.spacing`][confusius.xarray.FUSIAccessor.spacing], which returns the step size
  along each dimension as a dictionary:

  ```pycon
  >>> pwd.fusi.spacing
  {'time': 0.6, 'k': 0.4, 'j': 0.049, 'i': 0.091}
  ```

  This is particularly useful for sanity-checking voxel sizes or sampling periods before
  passing data to functions that require regular spacing (e.g., temporal filters, affine
  registration). Spatial dimensions (`k`/`j`/`i`) always have their spacing derived
  from the voxel-to-world affine, even for a singleton dimension. A singleton `time`
  dimension
  falls back to the `volume_acquisition_duration` attribute when present. Otherwise,
  `None` is returned with a warning when spacing cannot be determined: non-uniform
  coordinates, a single coordinate point, or no coordinate at all.

- [`.fusi.origin`][confusius.xarray.FUSIAccessor.origin], which returns the coordinate
  values at the origin along each dimension as a dictionary:

  ```pycon
  >>> pwd.fusi.origin
  {'time': 0.299, 'z': 0.0, 'y': 5.664, 'x': -3.583}
  ```

  This is typically used for computing the affine transformation corresponding to the
  world coordinates of the DataArray, for example when saving to NIfTI.

- [`.fusi.direction`][confusius.xarray.FUSIAccessor.direction], which returns the
  world-space direction matrix for the spatial dimensions:

  ```pycon
  >>> pwd.fusi.direction
  array([[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]])
  ```

  This is the identity for axis-aligned data (the common case). For oblique data, the
  columns are the unit world-space directions of the voxel axes.

### IQ Processing

The [`.fusi.iq`][confusius.xarray.FUSIIQAccessor] accessor lets you access the
[`process_iq_to_power_doppler`][confusius.iq.process_iq_to_power_doppler] and
[`process_iq_to_axial_velocity`][confusius.iq.process_iq_to_axial_velocity] functions
directly on a DataArray containing beamformed IQ data. Refer to the [Beamformed IQ
guide](beamformed-iq.md) for background IQ processing.

```python
import dask
import xarray as xr

import confusius  # Registers the .fusi accessor.

iq = cf.load("iq.zarr")

# Power Doppler with SVD clutter filtering (default).
pwd = iq.fusi.iq.process_to_power_doppler(
    clutter_window_width=200,
    doppler_window_width=100,
    low_cutoff=40,
)

# Axial velocity (in m/s).
velocity = iq.fusi.iq.process_to_axial_velocity(
    clutter_window_width=200,
    velocity_window_width=100,
)

(pwd, velocity) = dask.compute(pwd, velocity)  # Compute both in a single pass.
```

### Scaling

The [`.fusi.scale`][confusius.xarray.FUSIScaleAccessor] accessor provides common
scaling transformations: decibel, natural log, and power scaling.

```python
import numpy as np

pwd_db = pwd.fusi.scale.db()  # Default factor=10 for power quantities.
iq_db = np.abs(iq).fusi.scale.db(factor=20)  # Use factor=20 for amplitude quantities.

pwd_log = pwd.fusi.scale.log()

pwd_sqrt = pwd.fusi.scale.power(exponent=0.5)
```

Because the accessor returns a DataArray, it chains naturally with standard Xarray
operations:

```python
pwd_db = pwd.where(pwd > 0).fusi.scale.db()
```

### Registration

The [`.fusi.register`][confusius.xarray.FUSIRegistrationAccessor] accessor provides easy
access to the [`register_volumewise`][confusius.registration.register_volumewise]
function for motion correction.

```python
registered = pwd.fusi.register.volumewise(reference_time=0)
```

By default, rigid registration allows translation and rotation. Pass
`transform="translation"` for translation-only correction. For rigid registration,
set the first three `optimizer_weights` values to `0` to freeze rotation.

### Affine Transforms

The [`.fusi.affine`][confusius.xarray.FUSIAffineAccessor] accessor inspects and applies
the voxel-to-world and world-to-reference affines described in [Spatial
Conventions](spatial-conventions.md).

#### Reading and applying affines

Read the DataArray's
[`voxel_to_world`][confusius.xarray.FUSIAffineAccessor.voxel_to_world] affine, or
apply a world-space affine to its coordinates with
[`apply`][confusius.xarray.FUSIAffineAccessor.apply]—either a `(4, 4)` array directly,
or a string key naming a reference frame already stored in `attrs["affines"]`:

```python
voxel_to_world = pwd.fusi.affine.voxel_to_world  # (4, 4) array.

# By key: re-express world coordinates in the "world_to_qform" reference frame; the
# full affine (including any rotation) is absorbed into the DataArray's
# VoxelToWorldIndex.
registered_to_qform = pwd.fusi.affine.apply("world_to_qform")

# By array: apply an arbitrary (4, 4) world-space affine directly.
shifted = pwd.fusi.affine.apply(my_affine)
```

To replace a DataArray's voxel-to-world geometry outright (e.g. after computing a new
affine by hand), use
[`set_voxel_to_world`][confusius.xarray.FUSIAffineAccessor.set_voxel_to_world]:

```python
pwd = pwd.fusi.affine.set_voxel_to_world(new_voxel_to_world)
```

#### Relating two DataArrays

To compute the affine relating two DataArrays' world spaces through a named affine
they both carry in `attrs["affines"]` (e.g. a shared `"world_to_lab"` key from two
poses of the same acquisition), use
[`to`][confusius.xarray.FUSIAffineAccessor.to]:

```python
moving_to_fixed = moving.fusi.affine.to(fixed, via="world_to_lab")
```

#### Rebasing voxel coordinates to dense positions

Cropping or striding a DataArray doesn't renumber its `k`/`j`/`i` coordinates—slicing
`i=slice(3, 6)` keeps the labels `[3, 4, 5]`, not `[0, 1, 2]`:

```pycon
>>> cropped = pwd.isel(i=slice(3, 6))
>>> cropped.coords["i"].values
array([3, 4, 5])
>>> cropped.coords["x"].isel(k=0, j=0).values
array([3., 4., 5.])
```

That's correct as long as you keep indexing by *label*. But tools that assume dense,
zero-based voxel indices (ITK, nilearn, ...) read array *position* 0 as voxel 0—which
would silently place this cropped array at the wrong spot in world space.
[`reindex_voxels`][confusius.xarray.FUSIAffineAccessor.reindex_voxels] relabels `k`/`j`/`i`
back to `0, 1, ..., dim - 1` and adjusts the affine to compensate, so the world
coordinates stay exactly the same:

```pycon
>>> dense = cropped.fusi.affine.reindex_voxels()
>>> dense.coords["i"].values
array([0, 1, 2])
>>> dense.coords["x"].isel(k=0, j=0).values  # unchanged
array([3., 4., 5.])
```

The reverse problem shows up when two DataArrays occupy the exact same world grid but
carry different voxel labels—for example one was cropped from a larger array and the
other was freshly built with dense labels. Since `.sel()`, arithmetic, and
[`xarray.align`][xarray.align] all match by coordinate *label*, two such arrays won't
align automatically despite describing the same physical space.
[`reindex_voxels_like`][confusius.xarray.FUSIAffineAccessor.reindex_voxels_like] first
verifies the two DataArrays already occupy the same world grid, then relabels one
DataArray's voxel coordinates and affine to match the other's:

```python
aligned = data.fusi.affine.reindex_voxels_like(reference)
```

### Signal Extraction

The [`.fusi.extract`][confusius.xarray.FUSIExtractAccessor] accessor provides access to
signal extraction and reconstruction functions, making it easy to pass fUSI data to
scikit-learn, pandas, or other tools that expect a 2D matrix of shape
`(samples, features)`.

#### Mask-based extraction

[`extract_with_mask`][confusius.extract.extract_with_mask] flattens all voxels selected
by a boolean (or single-label integer) mask into a `space` dimension:

```python
mask = cf.load("brain_mask.zarr")

# signals has dims (time, space).
signals = registered.fusi.extract.with_mask(mask)
```

For a quick round-trip,
[`.unstack("space")`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.unstack.html)
reconstructs the spatial dimensions within the bounding box of the mask. To reconstruct
the full spatial volume, use
[`.fusi.extract.unmask()`][confusius.xarray.FUSIExtractAccessor.unmask] with the
original mask:

```python
# reconstructed is a VoxelData array.
reconstructed = signals.fusi.extract.unmask(mask)
```

#### Label-based extraction

[`extract_with_labels`][confusius.extract.extract_with_labels] aggregates signals by
brain region using an integer label map. It accepts two label formats:

- **Flat label map** `(k, j, i)`: each unique non-zero integer identifies a distinct,
  non-overlapping region (e.g., from an atlas annotation volume).
- **Stacked mask format** `(mask, k, j, i)`: one layer per region, with values in
  `{0, region_id}`. Regions may overlap. This is the format returned by
  [`get_masks`][confusius.atlas.AtlasAccessor.get_masks].

```python
# Using a flat label map (e.g., atlas annotations).
label_map = cf.load("atlas_labels.zarr")

# region_signals has dims (time, region).
region_signals = registered.fusi.extract.with_labels(label_map)

# Use a different aggregation (default is "mean").
region_signals = registered.fusi.extract.with_labels(label_map, reduction="sum")
```

### Functional Connectivity

The [`.fusi.connectivity`][confusius.xarray.FUSIConnectivityAccessor] accessor fits
seed-based correlation maps, wrapping
[`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps]. Provide either a seed mask
(voxels averaged into a seed signal) or a pre-computed seed signal directly:

```python
seed_masks = cf.load("seed_masks.zarr")

mapper = registered.fusi.connectivity.seed_map(seed_masks=seed_masks)

# spatial correlation maps, one per seed, and the seed signals used.
correlation_maps = mapper.maps_
seed_signals = mapper.seed_signals_
```

### Visualization

The [`.fusi.plot`][confusius.xarray.FUSIPlotAccessor] accessor provides easy access to
visualization functions for quick data inspection and quality control.

For example, to display data in [napari](https://napari.org/):

```python
viewer, layer = registered.fusi.plot.napari(gamma=0.5)
```

Or to show standardized time series in a carpet plot, useful for quality control:

```python
fig, ax = registered.fusi.plot.carpet(mask=mask)
```

### Saving

The [`.fusi.save`][confusius.xarray.FUSIAccessor.save] accessor allows saving a
DataArray to NIfTI or Zarr. For NIfTI, an accompanying fUSI-BIDS JSON sidecar is always
written alongside, storing converted metadata fields, custom attributes, and timing
fields derived from the `time` coordinate when available:

```python
# Creates sub-01_task-awake_pwd.nii.gz and sub-01_task-awake_pwd.json
registered.fusi.save("sub-01_task-awake_pwd.nii.gz")
```

## Complete Workflow Example

The following example shows a typical fUSI analysis from raw IQ to saved results:

```python
import confusius as cf
from confusius.decomposition import PCA

# 1. Load beamformed IQ data and corresponding brain mask.
iq = cf.load("iq.zarr")
brain_mask = cf.load("brain_mask.zarr")

# 2. Process IQ into power Doppler.
pwd = iq.fusi.iq.process_to_power_doppler(
    clutter_window_width=200,
    doppler_window_width=100,
    clutter_mask=brain_mask,
    low_cutoff=40,
)

# 3. Inspect in napari.
viewer, layer = pwd.fusi.plot.napari(gamma=0.5)

# 4. Motion correction.
registered = pwd.fusi.register.volumewise()

# 5. Quick quality check with a carpet plot.
fig, ax = registered.fusi.plot.carpet(mask=brain_mask)

# 6. Save registered power Doppler to NIfTI with a fUSI-BIDS JSON sidecar.
registered.fusi.save("sub-01_task-awake_pwd.nii.gz")

# 7. Extract global signal using the (boolean) brain mask.
global_signal = registered.fusi.extract.with_mask(brain_mask).mean("space")

# 8. Denoise and standardize power Doppler signals.
pwd_denoised = cf.signal.clean(
    registered, low_cutoff=0.01, confounds=global_signal, standardize_method="zscore"
)

# 9. Decompose brain signals with PCA.
pca = PCA(n_components=5, random_state=0)
component_signals = pca.fit_transform(pwd_denoised)
spatial_maps = pca.maps_
```

## API Reference

For full parameter documentation, see the [Xarray Integration API
reference](../api/xarray.md).
