---
icon: lucide/move-3d
---

# Spatial Conventions

ConfUSIus works with three kinds of coordinate systems:

- the **voxel space**, linked to the underlying array storage and indexed by integer
  voxel coordinates,
- the **physical space** embedded in every DataArray's coordinates,
- and any number of **reference spaces** (atlas, scanner, etc.) linked to the physical
  space through affine transforms stored in `attrs["affines"]`.

Understanding these three spaces and the axis-ordering convention used throughout
ConfUSIus makes it much easier to reason about I/O, registration, and downstream
statistical analysis.

```mermaid
---
config:
  layout: elk
---
flowchart LR
    V["<b>Voxel space</b><br>(integer indices)"]
    P["<b>Physical space</b><br>(probe-relative)"]
    W1["<b>Scanner space</b>"]
    ellipsis{{"..."}}
    W2["<b>Atlas space</b>"]

    V -->|".coords"| P
    P -->|".attrs[affines]"| W1
    P -->|".attrs[affines]"| W2
    P --> ellipsis

    ellipsis@{ shape: text }
```

## Axis Ordering: `(time, z, y, x)`

Every ConfUSIus DataArray that represents a fUSI recording uses the dimension order
`(time, z, y, x)`, where:

| Dimension | Physical axis | Typical size |
|---|---|---|
| `time` | Acquisition time | Thousands |
| `z` | Elevation (stacking direction) | 1 for 2D acquisitions |
| `y` | Axial / depth | Tens to hundreds |
| `x` | Lateral | Tens to hundreds |

!!! tip "Dimension ordering is mostly transparent in Xarray"
    Users familiar with neuroimaging may be more accustomed to spatiotemporal
    conventions like `(x, y, z, t)`. Thankfully, Xarray makes dimension ordering largely
    transparent in practice: you can always refer to dimensions by name and in any
    order (e.g. `data.mean("time")`, `data.sel(x=4.54, y=-2.48, z=0.0)`) rather than by
    axis index, so you won't have to remember the order of the dimensions.

This ordering is motivated by several considerations.

- **Equivalence with NIfTI:** NIfTI stores arrays in column-major (Fortran) order as
  `(x, y, z, time)`. Transposing to the more Pythonic row-major (C) order is a zero-copy
  operation that yields `(time, z, y, x)`.
- **Memory layout for volume-wise processing:** In row-major order the last axes are
  contiguous in memory, so `data[t]`—a single spatial volume—is a contiguous block,
  which is the natural unit of work for IQ processing, motion correction, and similar
  operations.
- **Statistical analysis convention:** After spatial processing, fUSI data is typically
  reshaped to `(time, voxels)` for GLMs and dimensionality reduction. This is
  `data.stack(voxels=["z", "y", "x"])` in Xarray, matching the standard
  `(samples, features)` convention of
  [scikit-learn](https://scikit-learn.org/stable/) and
  [statsmodels](https://www.statsmodels.org/stable/index.html).
- **Alignment with neuroanatomical atlases:** For coronal preclinical fUSI,
  `(z, y, x) = (elevation, axial/depth, lateral)` maps to
  `(antero-posterior, superior-inferior, left-right)`, sharing the first two axes with
  [BrainGlobe](https://brainglobe.info) atlases (e.g. Allen CCFv3). The physical →
  reference affine captures any remaining orientation difference (e.g. a lateral
  mirror).
- **Visualization:** Most visualization tools (e.g. napari) expect the last two axes to
  be the display axes of a 2D image. `data.sel(time=t, z=z)` yields a `(y, x)` array
  that plots correctly without transposing.

## Coordinate Systems

### Voxel Space

Voxel space has its origin at voxel `(0, 0, 0)` and integer indices along each
spatial axis. It is the natural indexing space of the underlying array: DataArrays can
be indexed in voxel space using the standard Xarray integer-location indexer (`.isel`).

### Physical Space

The physical space is the coordinate system embedded in the DataArray's dimension
coordinates. Its axes are `(z, y, x)` corresponding to `(elevation, axial/depth,
lateral)`. The unit of the coordinates is determined by the `units` attribute of each
coordinate array and is not fixed by ConfUSIus — millimeters are typical for fUSI, but
any consistent unit can be used.

The origin is typically the center of the probe surface, but users are free to define
any physical space they find convenient. What matters is that the coordinate values
are internally consistent and carry a meaningful physical scale.

Physical coordinates are set at data-loading time and depend on the source format:

- **EchoFrame**: Lateral and axial coordinates are read from the acquisition metadata
  file.
- **AUTC**: Lateral and axial coordinates are supplied by the user as parameters to the
  conversion function. If coordinates are omitted, ConfUSIus falls back to bare voxel
  indices and emits a warning.
- **Iconeus SCAN**: Coordinates are derived from the `voxelsToProbe` affine embedded in
  the SCAN file. The axial coordinate (`y`) is sign-flipped so that it is always
  positive and increases with depth.
- **NIfTI**: Coordinates are derived from the translation and scale components of the
  "best" affine transformation found in the file header.
- **Hand-constructed DataArrays**: The physical space is whatever the user assigns to
  the dimension coordinates.

!!! tip "The "best" NIfTI affine"
    NIfTI files can store two affine transforms in their header: an `sform` and a
    `qform`, each with an associated integer code indicating whether the affine is
    valid (`code > 0`) and which space it points to. ConfUSIus follows the NIfTI
    specification: if `sform_code > 0` the `sform` is used; otherwise, if
    `qform_code > 0` the `qform` is used. If both codes are zero a warning is emitted
    and coordinates fall back to a diagonal affine built from the `pixdim` field.

Each spatial coordinate also carries a `voxdim` attribute that records the native voxel
size along that dimension:

```python
da.coords["y"].attrs["voxdim"]  # native axial voxel size.
```

This is set at load time alongside the physical coordinates and is preserved through
any Xarray operation that propagates coordinate attributes. It is particularly useful
after downsampling: the coordinate values themselves reflect the new, coarser spacing,
but `voxdim` retains the original acquisition resolution. It is used wherever native
voxel dimensions are needed — for example, for aspect-ratio-correct visualization,
NIfTI export, and as a fallback when a dimension is later reduced to a single point
(e.g. after `.isel(z=0)`):

```python
da_down = da.isel(y=slice(None, None, 2), x=slice(None, None, 2))
# da_down.coords["y"].attrs["voxdim"] still holds the native voxel size.

da_slice = da_down.isel(z=0)
# da_slice.coords["z"].attrs["voxdim"] is now the only record of the z voxel size,
# since the coordinate has been collapsed to a scalar.
```

### Reference Spaces

ConfUSIus stores affine transformations between the DataArray's physical space and any
other reference physical space in `da.attrs["affines"]`, a dictionary keyed by affine
name. These target spaces can be external coordinate systems defined by other tools or
standards, for example an atlas space (Allen CCFv3), a scanner space, or a user-defined
stereotactic space. Each value is a `(4, 4)` homogeneous matrix in `(z, y, x)`
convention that maps a physical-space point to the corresponding point in the reference
space:

```python
A @ [pz, py, px, 1] = [rz, ry, rx, 1]
```

where `(pz, py, px)` are the physical coordinates stored in `da.coords`.

Several loaders populate `da.attrs["affines"]` automatically:

- **NIfTI**: NIfTI files store `qform` and `sform` affines in their header that map
  voxel indices to reference coordinates. [`load_nifti`][confusius.io.load_nifti] reads
  the relevant affine(s), converts them from voxel → reference to physical → reference
  form, and stores them under the keys `"physical_to_sform"` and/or
  `"physical_to_qform"` depending on which codes are valid in the header. When a file
  declares both forms, the physical coordinates come from the primary (usually `sform`)
  affine, and `physical_to_{secondary}` (usually `physical_to_qform`) is anchored to
  that same physical space, so both keys map the one set of coordinates in `da.coords`,
  just to different reference spaces. When saving back to NIfTI,
  [`save_nifti`][confusius.io.save_nifti] can write any named affine in
  `attrs["affines"]` to the header via its `qform=` and `sform=` arguments, with
  `"physical_to_qform"` and `"physical_to_sform"` used as convenience fallbacks.
- **Iconeus SCAN**: [`load_scan`][confusius.io.load_scan] stores a
  `"physical_to_lab"` affine mapping ConfUSIus physical coordinates `(z, y, x)` to the
  Iconeus lab coordinate system. For multi-pose acquisitions (`3Dscan`, `4Dscan`),
  one affine per pose is stored, with shape `(npose, 4, 4)`.

Registration transforms are handled separately from `attrs["affines"]`.
[`register_volume`][confusius.registration.register_volume] returns the estimated
transform explicitly; for linear registration it follows the SimpleITK pull convention
and maps fixed physical coordinates to moving physical coordinates. That pull transform
is useful for resampling, but it is not stored automatically in `attrs["affines"]`.

When registration is run with `resample=True`, the returned DataArray already lives on
the fixed/reference grid. In that case ConfUSIus copies `fixed.attrs["affines"]` onto
the result, so any named physical-space relationships already attached to the
fixed/reference DataArray are preserved automatically.

!!! question "Why physical → reference, not voxel → reference?"
    The standard NIfTI affine maps **voxel indices → reference coordinates**. ConfUSIus
    uses a **physical → reference** affine instead, for one practical reason: it is
    **invariant to slicing and downsampling**.

    A voxel → reference affine encodes the origin as the reference position of voxel
    `(0,0,0)` specifically. The moment you crop or subsample the DataArray—even a simple
    `.isel(y=slice(10, 50))`—voxel `(0,0,0)` is no longer in the array, and the affine
    silently points to the wrong location.

    A physical → reference affine operates on the coordinate values already stored in
    `da.coords`. Those values travel with the data through any Xarray operation that
    preserves coordinates, so the affine remains valid without any adjustment.

    [`save_nifti`][confusius.io.save_nifti] reconstructs the full voxel → reference
    NIfTI affine internally by composing the physical → reference affine with the
    voxel → physical map built from the dimension coordinate origin and spacing. Using
    the full affine (not just its orientation) means each affine keeps its own origin, so
    a file with distinct `sform` and `qform` round-trips faithfully.

## Switching Physical Spaces

The physical space is a *choice of frame*, not a fixed property of the data. Whenever a
DataArray carries an affine to a reference space in `attrs["affines"]`, you can
**re-anchor its coordinates onto that frame** with
[`.fusi.affine.apply`][confusius.xarray.FUSIAffineAccessor.apply], making the reference
space the new physical space. This is handy for visualization (so coordinate ticks and
napari's world axes read in the target frame) and for initializing a registration
(switching two scans into a common frame lines up their coordinates, leaving a small,
optimizer-friendly transform between them).

Take the `sform`/`qform` example from [Reference Spaces](#reference-spaces). A NIfTI
file with both affines anchors its physical space to the axis-aligned `sform` frame (the
primary affine when `sform_code > 0`). The remaining orientation component of the
`sform` is carried on the `physical_to_sform` affine attribute, while
`physical_to_qform` records the mapping of the current physical space to the `qform`
frame.

The example below shows an `sform` with no rotation or shear, resulting in a
`physical_to_sform` that is the identity. The `qform` has a shift in `(z, y)` of (+10,
+5) with respect to the `sform`, so `physical_to_qform` encodes that shift:

```python
da.coords["z"].values  # array([0., 1., 2.])
da.coords["y"].values  # array([0., 1., 2., 3.])

da.attrs["affines"]["physical_to_sform"]  # identity
# array([[ 1.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])

da.attrs["affines"]["physical_to_qform"]  # here a (+10, +5) shift in (z, y):
# array([[ 1.,  0.,  0., 10.],
#        [ 0.,  1.,  0.,  5.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])
```


Applying `physical_to_qform` moves the coordinates into the `qform` frame, collapses
`physical_to_qform` to the identity (the data is *in* that frame now), and re-expresses
`physical_to_sform` to point back the way you came:

=== "`confusius.xarray.apply_affine`"

    ```python
    da_q, orientation = cf.xarray.apply_affine(da, da.attrs["affines"]["physical_to_qform"])
    ```

=== "Xarray accessor"

    ```python
    da_q, orientation = da.fusi.affine.apply(da.attrs["affines"]["physical_to_qform"])
    ```

```python
da_q.coords["z"].values  # array([10., 11., 12.])   <- shifted by +10
da_q.coords["y"].values  # array([5., 6., 7., 8.])  <- shifted by +5

da_q.attrs["affines"]["physical_to_sform"]  # the (-10, -5) shift back to sform
# array([[ 1.,  0.,  0., -10.],
#        [ 0.,  1.,  0.,  -5.],
#        [ 0.,  0.,  1.,   0.],
#        [ 0.,  0.,  0.,   1.]])

da_q.attrs["affines"]["physical_to_qform"]  # identity, the new physical frame
# array([[ 1.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])
orientation                                 # identity, nothing left over
```

Every other affine in `attrs["affines"]` (atlas, lab, ...) is re-expressed the same way,
like `physical_to_sform` above, so it stays valid.

### Rotations and Shears

As explained above, ConfUSIus physical coordinates are three independent 1D arrays, so
they can encode scaling and translation but **not a rotation or shear that mixes axes**.
If `physical_to_qform` were instead only a 90° rotation in the `(z, y)` plane, none of
it could go into the coordinates—they come back unchanged from `apply_affine`—and the
whole rotation is returned as `orientation`:

```python
da.coords["z"].values  # array([0., 1., 2.])
da.coords["y"].values  # array([0., 1., 2., 3.])

da.attrs["affines"]["physical_to_sform"]  # identity
# array([[ 1.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])

da.attrs["affines"]["physical_to_qform"]  # here a 90° rotation in (z, y) plane:
# array([[ 0., -1.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])
```

=== "`confusius.xarray.apply_affine`"

    ```python
    da_q, orientation = cf.xarray.apply_affine(da, da.attrs["affines"]["physical_to_qform"])
    ```

=== "Xarray accessor"

    ```python
    da_q, orientation = da.fusi.affine.apply(da.attrs["affines"]["physical_to_qform"])
    ```

```python
da_q.coords["z"].values  # array([0., 1., 2.])      <- unchanged
da_q.coords["y"].values  # array([0., 1., 2., 3.])  <- unchanged

orientation  # the 90° rotation the coordinates could not absorb:
# array([[ 0., -1.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])

da_q.attrs["affines"]["physical_to_sform"]  # identity
# array([[ 1.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])

da_q.attrs["affines"]["physical_to_qform"]  # here a 90° rotation in (z, y) plane:
# array([[ 0., -1.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  1.]])
```

In general `apply` absorbs the axis-aligned part (zoom + translation) into the
coordinates and returns the residual rotation, with
`orientation @ new_physical == affine @ old_physical`. You can leave it in
`physical_to_qform`, compose it, or store it elsewhere.
