# VoxelData geometry reference

## Spatial coordinates

`VoxelData` is an `xarray.DataArray` with canonical dimensions `(...extra, time, pose,
k, j, i)` and a `VoxelToWorldIndex` defining world coordinates `z/y/x`. `time` and
`pose` are optional; all three spatial dimensions are required, including for slices and
lines.

Distinguish three coordinate spaces:

- **Array space:** Dense zero-based positions, selected with `isel`.
- **Voxel space:** Integer `k/j/i` labels, selected with `sel`. Initially
  `arange(size)`, but preserved through cropping and striding.
- **World space:** Physical `z/y/x` coordinates derived from voxel labels
  through the affine. They are not independent input coordinates.

World coordinates may depend on all three voxel axes for oblique grids.
A non-spatial `(time, region)` signals table is not `VoxelData`.

## Affine convention

The voxel-to-world affine maps **voxel labels to world coordinates**:

```text
[z, y, x, 1]^T = A @ [k, j, i, 1]^T

A[:3, :3] = direction @ diag(spacing)
A[:3, 3] = origin
A[3, :] = [0, 0, 0, 1]
```

- `spacing=(dz, dy, dx)` and `origin=(z0, y0, x0)` follow ConfUSIus
  `(z, y, x)` conventions, regardless of raw array order.
- `direction` has world `(z, y, x)` rows and voxel `(k, j, i)` columns.
  Its columns specify the physical directions of the voxel axes.
- Spacing must be positive; represent axis reversals in the direction matrix. Geometry
  must be finite and invertible.

The origin is the world position of **voxel label zero**, not necessarily the first
stored voxel after cropping. Geometry refers to voxel centers; convert corner-based
coordinates with the appropriate half-voxel offset.

When `origin` is omitted, ConfUSIus uses probe-centered defaults:

```text
z0 = -(nk - 1) * dz / 2
y0 = dy / 2
x0 = -(ni - 1) * dx / 2
```

These center the elevation/lateral axes and place the first depth voxel half a voxel
below the probe surface in the standard axis-aligned case. Prefer calibrated origins for
custom imports, especially anatomical data.

### Converting external geometry

Pass raw arrays unchanged, with `dims` describing their stored axis order. Convert the
source geometry separately.

For axis-aligned data with `i → x`, `j → y`, `k → z`, metadata stored in `(i, j, k)`
order must be reversed:

```python
spacing_ijk = (0.1, 0.05, 0.4)
spacing = spacing_ijk[::-1]  # (dz, dy, dx)
```

For a general external affine `B` mapping `[i, j, k, 1]` to `[x, y, z, 1]`, permute both
its input and output coordinate orders:

```python
P = np.eye(4)[[2, 1, 0, 3]]
A = P @ B @ P
```

This changes coordinate conventions, **not the array data**. It does not perform
registration, spatial flipping, or RAS↔LPS conversion. Invert an external affine first
if it maps world to voxel coordinates.

Use ConfUSIus's NIfTI loader rather than manually interpreting NIfTI header affines.

### Cropping, striding, and flipping

Spatial selections preserve voxel labels, so the affine continues to map them correctly.
Do not reset labels without updating the affine.

Use `confusius.xarray.reindex_voxels(data)` to restore dense zero-based labels while
preserving world positions, particularly before exporting array-space affines to other
packages.

If a **raw array** is reversed along `i` and assigned new zero-based labels, update its
affine accordingly:

```python
F = np.eye(4)
F[2, 2] = -1
F[2, 3] = ni - 1
A_flipped = A @ F
```

When reversing an **existing VoxelData** with `data.isel(i=slice(None, None, -1))`,
labels already track the reversal; do not modify the affine again.

Read geometry through `data.fusi.affine.voxel_to_world` and `data.fusi.affine.units`;
modify it using `set_voxel_to_world` and `set_units`. Do not edit `z/y/x` coordinate
attributes directly. Changing units metadata does not rescale coordinates.

`data.fusi.origin`, `.spacing`, and `.direction` describe the sampled **array-space
grid**, accounting for voxel-label offsets, strides, and reversals. They may differ from
the original affine parameters.

## Multiple probe poses

A `pose` dimension represents distinct probe positions, not components or repetitions.
It requires one affine per pose, with `voxel_to_world.shape == (npose, 4, 4)`, even for
a single pose. Spatial scale magnitudes must agree across poses; origins and
orientations may differ. Do not silently resample mismatched grids.

For recordings, timestamps must be pose-dependent with shape `(ntime, npose)`:

- Provide measured `time` for irregular acquisitions.
- For regular acquisitions, provide `dt` (repeat interval of the **same pose**) and one
  `t0` per pose, giving `time[t, p] = t0[p] + t * dt`.
- For simultaneous poses, repeat equal timestamps across pose columns.

`volume_acquisition_duration` describes each pose's acquisition window, not the entire
sweep. Determine it from the acquisition protocol rather than relying on the fallback
inferred from inter-pose timing.

World coordinates depend on `(pose, k, j, i)`. Select one pose before world-coordinate
selection. For time-label selection:

```python
one = data.isel(pose=0).set_xindex("time")
one.sel(time=...)
```

Do not combine separate probe positions into a spatial dimension without representing
their actual geometry.

## Timing and validation

Time coordinates require `units`, `volume_acquisition_reference` (`start`, `center`,
`end`), and positive `volume_acquisition_duration`. Use seconds consistently.

Acquisition duration is distinct from the sampling interval (`dt`). For power Doppler,
it typically represents the integration window, which may overlap between consecutive
frames. Inferred durations may not reflect the actual acquisition or processing
protocol.

Run `validate_voxeldata(data)` to check dimensions, coordinates, geometry, index
structure, and metadata. It cannot verify physical calibration or orientation against
the source.

Use `require_time=True` when multiple timepoints are required; static volumes and
single-timepoint arrays can otherwise be valid `VoxelData`. Dtype and IQ-velocity
metadata requirements depend on the downstream operation.
