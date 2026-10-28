# VoxelData geometry reference

## One spatial model

A VoxelData array is an `xarray.DataArray` with canonical dimensions
`(...extra, time, pose, k, j, i)` and a `VoxelToWorldIndex` attached to `z/y/x`.
`time` and `pose` are optional; all three spatial dimensions remain present even
for a slice or line. An already-reduced `(time, region)` signals table is not a
spatial array and should not be wrapped as one.

`k/j/i` are integer voxel **labels**, initially `arange(size)`; they are not mm.
`isel` selects dense array positions; `sel(k=..., j=..., i=...)` selects labels.
World coordinates are lazily derived by the index, not independent input arrays.
Oblique affines can make every world coordinate depend on all three voxel axes.

## Affine convention

For a single grid:

```text
[z, y, x, 1]^T = A @ [k_label, j_label, i_label, 1]^T
A[:3, :3] = direction @ diag(spacing)
A[:3, 3] = origin
A[3, :] = [0, 0, 0, 1]
```

Rows are world `z/y/x`; columns are voxel `k/j/i`. Direction columns specify the
world direction of each voxel axis. Spacing is positive; encode reflection in the
direction/affine, not negative spacing. Require finite, physically meaningful,
invertible geometry; a homogeneous bottom row alone is not sufficient calibration.
An oblique grid cannot be represented by three independent 1D world axes.

The origin is the position of **voxel label zero**, which equals the first stored
voxel only for zero-based labels. Geometry maps voxel centers. If metadata describes
an outer corner, translate by half a voxel along each relevant direction column.
Do not apply a half-voxel correction unless the source convention warrants it.

With omitted `origin`, probe defaults for dense zero-based labels are:

```text
z0 = -(nk - 1) * dz / 2
y0 = dy / 2
x0 = -(ni - 1) * dx / 2
```

Thus depth starts half a voxel below the probe surface, and elevation/lateral are
centered. These defaults are not universal scanner or atlas origins. Prefer explicit
calibrated origins for custom imports.

### Converting an external xyz/ijk affine

If `B` maps `[i, j, k, 1]` to `[x, y, z, 1]`, reverse **both** orders:

```python
P = np.eye(4)[[2, 1, 0, 3]]
A = P @ B @ P
```

This is a basis permutation only, not an RAS↔LPS conversion, registration, or an
instruction to flip the data. Determine physical handedness separately. Use the
built-in NIfTI reader for NIfTI files rather than reproducing header/sform/qform
handling. An affine mapping the opposite direction must first be inverted.

### Reordering, cropping, and flipping

Naming `dims` in raw order lets the constructor reorder storage correctly. It does
not infer the physical meaning of an axis. If metadata uses a different voxel-axis
order, its affine columns need the corresponding basis conversion as well.

After a VoxelData crop or stride, labels retain their original values and the affine
still maps those labels. Do not reset labels to zero without updating the affine.
`confusius.xarray.reindex_voxels(data)` returns dense zero-based labels with an updated
affine preserving world locations. This is useful before exporting an array-space
affine to another package.

If a raw array is flipped and then assigned fresh dense labels, compose the affine
with the index transform. For a reversed `i` axis of length `ni`:

```python
F = np.eye(4)
F[2, 2] = -1
F[2, 3] = ni - 1
A_flipped = A @ F
```

Do not copy another dataset's display flip without checking orientation. When
flipping an existing VoxelData array with `isel(i=slice(None, None, -1))`, the index
already tracks the reversed labels; do not apply this correction a second time.
Use length-one slices such as `isel(k=slice(0, 1))` to retain all spatial dims, or
`ensure_voxeldata` after scalar selection.

Read geometry from `data.fusi.affine.voxel_to_world` and
`data.fusi.affine.units`. Change it through `set_voxel_to_world`/`set_units`, not by
editing `z/y/x` coordinate attributes or adding a geometry-looking array attribute.
Changing the units string does not rescale numbers. `data.fusi.origin`, `.spacing`,
and `.direction` describe the sampled array grid, taking voxel-label strides and
reversals into account; they need not equal the original affine parameters.

## Multiple probe poses

A `pose` dimension means distinct probe positions, not components or repetitions.
It **always** requires `voxel_to_world` with shape `(npose, 4, 4)`, even for one
pose. A shared 4×4 affine or `spacing` alone is invalid with `pose`. Without `pose`,
use a single 4×4 affine. Pose labels are increasing integers. Pose affines must have
equal spatial scale magnitudes; origins/orientations can differ. Do not silently
resample differing grids to make them fit this model.

For recordings with time, use pose-dependent `time` of shape `(ntime, npose)`:

- Supply measured `time=timestamps_seconds` for irregular acquisition.
- Or supply shared `dt` (repeat period of the **same pose**) and one `t0` per pose.
  This generates `time[t, p] = t0[p] + t * dt`.
- Explicitly repeat equal timestamps across pose columns for simultaneous poses.
- `volume_acquisition_duration` describes one pose's acquisition, not the full sweep.
  If not supplied, duration is inferred from the smallest nonzero inter-pose gap
  where possible. Set the measured duration instead of relying on that inference.

World coordinates then depend on `(pose, k, j, i)`. Select a scalar pose before
world-coordinate selection. For time-label lookup on a pose's 2D-derived time,
use `one = data.isel(pose=0).set_xindex("time")` before `one.sel(time=...)`.
Do not confuse `pose` with an extra component axis or concatenate positions into
`k` without representing their true geometry.

## Time and validation

Time coordinates need `units`, `volume_acquisition_reference` (`start`, `center`,
`end`), and positive `volume_acquisition_duration`. Use seconds consistently;
constructor-generated time has `units="s"`. For plain explicit timestamps, an
interval may be inferred from their differences, including the median for jittered
times, but that is not necessarily the actual acquisition duration.

Use `validate_voxeldata(data)` as the universal check. It validates structure, index,
coordinates, canonical ordering, and metadata; it does not inspect the experiment's
true orientation. `require_time=True` additionally requires more than one timepoint;
a static map or a single timestamp can still be valid VoxelData without that flag.
Dtype and IQ-velocity metadata checks are optional, consumer-specific requirements.
