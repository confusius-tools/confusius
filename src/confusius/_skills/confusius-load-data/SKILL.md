---
name: confusius-load-data
description: Load custom or supported ultrasound data into ConfUSIus. Use for file imports, VoxelData dimensions, acquisition timing, and voxel-to-world geometry.
compatibility: Requires Python and ConfUSIus; custom readers may need NumPy, SciPy, or h5py.
---

# Load data into ConfUSIus

## 1. Choose the existing loader or a custom import

For **supported formats, use ConfUSIus I/O**, not `create_voxeldata`:

| Source | Entry point |
|---|---|
| NIfTI, Iconeus SCAN, VoxelData-convention Zarr | `confusius.load(path)` |
| EchoFrame DAT + metadata MAT | `confusius.io.load_echoframe_dat(dat_path, meta_path=...)` |
| AUTC DAT folder | `confusius.io.convert_autc_dats_to_zarr(...)`, then `confusius.load(...)` |

AUTC conversion requires spatial calibration from the user. Do not infer the vendor
from `.dat` alone: the generic DAT loader handles EchoFrame, not AUTC.
For other supported formats, check `confusius.io` before implementing an import.

For **custom formats** (lab-specific MATLAB, NPY, HDF5, foreign Zarr, etc.), read the
array with the appropriate library, then use `confusius.create_voxeldata`.
Use `scipy.io.loadmat` for pre-v7.3 MAT, `h5py` for v7.3/HDF5, or
`np.load(path, allow_pickle=False)` for numeric NPY.

## 2. Help the user establish the metadata

Inspect file variables, sidecars, acquisition settings, and export code first.
Ask the user to confirm anything missing or ambiguous; do not guess from shape:

- **Axis order:** What does each stored axis represent? `(i, j, k, time)` is common,
  but not universal. Pass `dims` in the actual raw order. Typically `i` is lateral,
  `j` is depth, and `k` is elevation; confirm this for their export.
- **Spatial calibration:** What are the voxel spacings, units, and first voxel-center
  position? Are rotations, direction vectors, or a full affine already available?
  Establish their axis order and physical frame rather than discarding orientation.
- **Timing:** Is sampling regular? Use `dt` in seconds (or `1 / rate_hz`) and `t0`.
  For irregular sampling, pass exact increasing timestamps through `time`; preserve
  jitter and convert to seconds. If available, supply `volume_acquisition_reference`
  (`start`, `center`, `end`) and `volume_acquisition_duration`; duration need not equal `dt`.
- **Other axes/data:** Identify probe poses versus extra dimensions; preserve complex
  IQ values. Use `extra_coords` for extra-axis labels and `attrs` for data metadata.

## 3. Construct custom VoxelData with the correct geometry

Choose **one** geometry representation:

- `spacing=(dz, dy, dx)`, optionally `origin=(z0, y0, x0)` and a 3×3 `direction`;
  direction rows are world `z/y/x`, columns voxel `k/j/i`. Use positive spacing.
- `voxel_to_world=A`, mapping `[k_label, j_label, i_label, 1]` to `[z, y, x, 1]`;
  do not also pass `spacing`, `origin`, or `direction`.

Units label geometry; they do not convert it. Do not copy an external `ijk → xyz`
affine unchanged. Read [geometry details](references/geometry.md) for basis conversion,
rotations, cropping, defaults, or multiple poses. A `pose` dimension requires one
4×4 affine per pose, even for a single pose; recordings also need pose-dependent timestamps.

Example **after confirming** an NPY export is `(i, j, k, time)`, sampled every 0.4 s,
with spacings `(dz, dy, dx) = (0.4, 0.05, 0.1)` mm and voxel-zero center at the origin:

```python
import numpy as np
import confusius as cf
from confusius.validation import validate_voxeldata

raw = np.load("recording.npy", allow_pickle=False)
data = cf.create_voxeldata(
    raw,
    dims=("i", "j", "k", "time"),
    dt=0.4,  # For irregular sampling, replace with time=timestamps_seconds.
    spacing=(0.4, 0.05, 0.1),  # Or use voxel_to_world=calibrated_affine.
    origin=(0.0, 0.0, 0.0),
    units="mm",
)
validate_voxeldata(data)
```

The constructor returns `(...extra, time, pose, k, j, i)`, omitting unused `time`/`pose`
and adding missing singleton spatial axes. `z/y/x` are derived world coordinates,
never dimension names. Do not manually attach world-coordinate grids.

Verify a raw sample and its physical position against the result; structural
validation cannot prove calibration. Report unresolved metadata, not invented values.
Run [tested examples](scripts/examples.py) for regular/irregular timing, affines, poses,
IQ, and file round-trips. Bundled paths are relative to this skill's directory.
