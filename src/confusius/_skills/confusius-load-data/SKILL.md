---
name: confusius-load-data
description: Load any spatially referenced voxel array into ConfUSIus, including fUSI recordings, derived maps, anatomical images, and atlases. Use when importing data from files or arrays, choosing loaders, or constructing VoxelData with correct dimensions, coordinates, timing, and spatial geometry.
compatibility: Requires Python and ConfUSIus; custom readers may need SciPy, h5py, or other libraries.
---

# Load spatial data into ConfUSIus

Prefer the installed ConfUSIus API and version-matched documentation.

## 1. Choose an existing loader or custom import

Use existing ConfUSIus loaders whenever possible:

| Source | Entry point |
|---|---|
| NIfTI, Iconeus SCAN, ConfUSIus Zarr, EchoFrame DAT | `confusius.load(path)` |
| AUTC DAT folder | `confusius.io.convert_autc_dats_to_zarr(...)`, then `confusius.load(...)` |
| BrainGlobe atlas | `confusius.datasets.fetch_brainglobe_atlas(name)` |
| Saved ConfUSIus atlas | `confusius.io.load_atlas(path)` |

For EchoFrame DAT files, `confusius.load` uses the adjacent `ScanParameters.mat` by
default, or accepts an explicit `meta_path`. AUTC DAT files require conversion and
spatial calibration. Do not infer the DAT format from its extension alone. Check
`confusius.io` for additional loaders.

For **custom formats** (MATLAB, NPY, HDF5, foreign Zarr, etc.), read the array with an
appropriate library, then use `confusius.create_voxeldata`. Use `scipy.io.loadmat` for
pre-v7.3 MAT files, `h5py` for v7.3/HDF5, or `np.load(..., allow_pickle=False)` for
numeric NPY.

## 2. Help the user identify dimensions and metadata

`VoxelData` represents any spatially referenced voxel array, including fUSI recordings
(IQ, power Doppler, velocity), derived maps (correlation, statistics, decomposition
components), anatomical images, displacement fields, masks, and atlas labels.

A complete brain atlas is instead an `xarray.Dataset` containing anatomical reference
and annotation volumes, hemisphere labels, and region metadata. Preserve this structure
when loading atlases.

Inspect source metadata, sidecars, acquisition settings, and export code. Confirm
missing or ambiguous information with the user rather than guessing:

- **Dimensions:** Identify the stored axis order and pass `dims` accordingly. Typically
  `i` is lateral, `j` is depth, and `k` is elevation, but conventions vary. Distinguish
  probe poses from extra dimensions (components, seeds, channels, etc.).
- **Geometry:** Determine voxel spacing, origin, orientation, units, and coordinate
  frame. Preserve existing affines rather than approximating geometry.
- **Timing (if applicable):** Use `dt` and `t0` for regular sampling, or `time` for
  irregular timestamps (all in seconds). Determine `volume_acquisition_duration`
  (typically the integration window for power Doppler) and
  `volume_acquisition_reference` (`start`, `center`, `end`) from the acquisition or
  processing protocol. Duration may differ from `dt`.
- **Data semantics:** Preserve complex IQ, signed velocities, integer atlas labels, and
  boolean masks. Use `extra_coords` for extra-axis labels and `attrs` for metadata.

Static images and maps do not require a `time` dimension. Do not add one artificially.
Region-by-time signal tables without spatial dimensions are not `VoxelData`.

## 3. Construct and validate VoxelData

Pass the raw array unchanged and specify its axis order with `dims`. Geometry parameters
follow ConfUSIus conventions, regardless of the input array's order.

Choose **one** geometry representation:

- `spacing=(dz, dy, dx)`, optionally `origin=(z0, y0, x0)` and a 3×3 `direction`. These
  follow world `(z, y, x)` order; direction rows correspond to world `(z, y, x)` and
  columns to voxel `(k, j, i)` axes.
- `voxel_to_world=A`, a 4×4 affine mapping voxel `[k, j, i, 1]` to world `[z, y, x, 1]`.
  Do not combine it with `spacing`, `origin`, or `direction`.

If source metadata follows the raw `(i, j, k)` order, convert it to ConfUSIus
conventions. For axis-aligned data with `i → x`, `j → y`, `k → z`, this
simply means reversing the spacing and origin tuples. For other orientations,
convert the full affine instead.

Spacing must be positive. Units label coordinates but do not convert them. See [geometry
details](references/geometry.md) for affine conversions, rotations, cropping, defaults,
and multi-pose acquisitions. A `pose` dimension requires one affine per pose and, for
recordings, pose-dependent timestamps.

For example, suppose two NPY exports share a spatial grid stored in
`(i, j, k)` order, with spacing and origin metadata in `(x, y, z)` order:

```python
import numpy as np
import confusius as cf
from confusius.validation import validate_voxeldata

# Source calibration in (x, y, z) order, with i → x, j → y, k → z.
spacing_xyz = (0.1, 0.05, 0.4)  # mm
origin_xyz = (1.0, 2.0, 3.0)   # mm, first voxel center

geometry = {
    "spacing": spacing_xyz[::-1],  # (dz, dy, dx)
    "origin": origin_xyz[::-1],    # (z0, y0, x0)
    "units": "mm",
}

# Time-series recording: (i, j, k, time)
power = cf.create_voxeldata(
    np.load("power.npy", allow_pickle=False),
    dims=("i", "j", "k", "time"),
    dt=0.4,
    **geometry,
)

# Static seed-correlation map: (i, j, k)
seed_map = cf.create_voxeldata(
    np.load("seed_map.npy", allow_pickle=False),
    dims=("i", "j", "k"),
    **geometry,
)

validate_voxeldata(power)
validate_voxeldata(seed_map)
```

The constructor reorders dimensions to `(...extra, time, pose, k, j, i)`,
omitting unused `time`/`pose` axes and adding missing singleton spatial axes.
World coordinates `z/y/x` are derived from the affine, not used as dimension names.

Verify array values, dimensions, timing (if present), and physical positions
against the source. Structural validation cannot prove physical calibration.
Report unresolved metadata rather than inventing values.

See [runnable examples](scripts/examples.py) for irregular timing, affine
conversions, multiple poses, complex IQ, and file round-trips. Bundled paths
are relative to this skill's directory.
