---
name: confusius-load-data
description: Load custom functional ultrasound, power Doppler, beamformed IQ, atlas, or other voxel arrays into ConfUSIus. Use when importing NumPy, MATLAB MAT, HDF5, binary, or lab-specific exports; assigning time and voxel dimensions; converting voxel-to-world affines; or debugging VoxelData geometry and validation.
compatibility: Requires Python with ConfUSIus and NumPy. Custom file readers may require SciPy or h5py. Instructions are agent-neutral; no MCP server, network access, or editor-specific tools required.
---

# Load custom data into ConfUSIus

Turn the source array and its acquisition metadata into a **VoxelData array** using
`confusius.xarray.create_voxeldata` (also exported as `confusius.create_voxeldata`).
Do not write a new loader when an existing ConfUSIus reader handles the format.
Do not guess physical calibration from array shape or silently accept a plain
DataArray as spatial data.

Resolve bundled paths relative to this skill's directory, not the working directory.
Read [the geometry reference](references/geometry.md) before adapting an affine,
flipping axes, handling cropped data, or importing multiple probe poses.
See [installation and verification](README.md) for setup and supporting sources.

## 1. Establish the source contract

Inspect the file with its appropriate reader and record:

- Dataset/variable name, shape, dtype, and the meaning and order of **every** axis.
  Check acquisition metadata or export code; MATLAB/HDF5 layouts are not universal.
- Which axes are elevation, depth, lateral, time, probe pose, or extra features.
- Spatial spacing, origin, orientation, units, and whether positions describe voxel
  centers or edges. If an affine is supplied, establish its input/output axis order,
  handedness, coordinate frame, and whether it maps voxel labels or array positions.
- Frame timestamps or sampling rate, their units and reference (start/center/end),
  and actual acquisition duration. A frame period need not equal acquisition duration.
- Whether IQ is already complex or stored as separate real/imaginary channels.

Ask for missing calibration or explicitly state an uncalibrated assumption before
proceeding. Never invent atlas alignment. Do not overwrite the source file.

Prefer `cf.load(path)` for supported NIfTI, Zarr, and SCAN inputs; use the appropriate
`confusius.io` reader/converter for supported vendor exports. For unsupported inputs,
read only the needed array: `np.load(path, allow_pickle=False)` for numeric NPY,
`scipy.io.loadmat(path)[variable]` for pre-v7.3 MAT, or `h5py.File(path, "r")` for
HDF5/v7.3 MAT. Inspect keys and metadata instead of assuming a variable name. Keep
lazy HDF5-backed readers open until computation finishes; `dataset[()]` is eager.
Never enable pickle or execute code embedded in an untrusted file.

## 2. Name the input axes; let the constructor canonicalize

VoxelData always has dimensions `(...extra, time, pose, k, j, i)` in that order.
`time` and `pose` are optional; `k`, `j`, and `i` are always present in the result.
Typical probe axes are `k` = elevation, `j` = depth, `i` = lateral, but verify the
source. Extra dimensions (e.g. `component`) precede the core dimensions.

Pass `dims` in the **actual raw array order**. The constructor transposes by name
and inserts missing spatial dimensions as singletons. Do not relabel a reversed
axis or reshape to pretend its samples are in a different order. `z`, `y`, and `x`
are derived world coordinates, **never dimension names**.

## 3. Supply geometry and time explicitly

Choose exactly one geometry representation:

- `spacing=(dz, dy, dx)`, optionally `origin=(z0, y0, x0)` and a 3×3 `direction`.
  Supply all three positive spacings even for a 2D slice. Any singleton-axis spacing
  must be documented if it is merely a placeholder, not measured slice thickness.
- `voxel_to_world=A`, a 4×4 homogeneous affine mapping
  `[k_label, j_label, i_label, 1]` to `[z, y, x, 1]`.
  Do **not** also supply `spacing`, `origin`, or `direction`.

Set `units="mm"` only after converting the numeric geometry to millimeters.
Units are labels, not conversions. The constructor creates the lazy
`VoxelToWorldIndex` on `z/y/x`; do not manually store world coordinate grids or call
private index-attachment helpers.

For time, use `dt=1 / frame_rate_hz` and `t0`, or actual increasing `time=` values
in seconds. Preserve jitter; do not replace measured timestamps with a synthetic
regular grid. Supply `volume_acquisition_reference` and
`volume_acquisition_duration` from metadata when known. These become attributes of
`time`, not the array. Defaults are `"start"` and an inferred sample interval;
these are conveniences, not evidence of acquisition timing. For one timestamp,
supply a duration explicitly (or a known `dt`). See the reference for per-pose time.

## 4. Build and verify

Example source contract: a calibrated 2D recording stored as `(lateral, depth, time)`,
2.5 Hz sampling, first voxel center at `(z, y, x) = (0, 0.025, -0.15)` mm.
The singleton elevation spacing below is a **placeholder**, not slice thickness.

```python
import numpy as np
import confusius as cf
from confusius.validation import validate_voxeldata

raw = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
power = cf.create_voxeldata(
    raw,
    dims=("i", "j", "time"),
    dt=0.4,
    spacing=(1.0, 0.05, 0.1),
    origin=(0.0, 0.025, -0.15),
    units="mm",
    volume_acquisition_reference="start",
    volume_acquisition_duration=0.2,
    name="power_doppler",
)
validate_voxeldata(power, require_time=True)
np.testing.assert_array_equal(power.data[:, 0], raw.transpose(2, 1, 0))
np.testing.assert_allclose(power.time, np.arange(5) * 0.4)
point = power.isel(time=2, k=0, j=1, i=3)
np.testing.assert_allclose(
    [point.z.item(), point.y.item(), point.x.item()], [0.0, 0.075, 0.15]
)
assert point.item() == raw[3, 1, 2]
```

For a static volume omit `time`/`dt`. For extra dimensions use
`extra_coords={"component": ["a", "b"]}` and include `component` in `dims`.
For beamformed IQ preserve the complex dtype and use this same constructor; put
`transmit_frequency` (Hz) and `beamforming_sound_velocity` (m/s) in `attrs`.
Validate those only when needed with `require_velocity_attrs=True`, and check
complex data with `require_dtype=np.complexfloating`. There is no separate IQ
geometry model.

Run [the executable examples](scripts/examples.py) using the Python environment
where ConfUSIus is installed. They include the example above, NPY/MAT/HDF5 imports,
oblique affines, axis-order conversion, cropping/flips, IQ, multiple poses, and a
Zarr round-trip. They use temporary synthetic files and need no downloads.

## 5. Finish with evidence

- Call `validate_voxeldata` on the result. Add flags such as
  `require_uniform_time` or `require_unchunked_time` only for a consumer that needs
  them. Validation does not prove that the physical calibration is correct.
- Check a known raw sample against its VoxelData sample and verify world positions
  at the first voxel and at least one other landmark against acquisition metadata.
- If a prior scalar selection removed a spatial dimension, use
  `ensure_voxeldata` to canonicalize and validate. It cannot turn an unindexed plain
  `z/y/x` array into VoxelData or recover unknown geometry.
- For large arrays keep Dask data lazy; avoid `.values`, eager casts, and materializing
  full world grids. Choose chunks for the intended algorithm, not merely the import.
- If persistence is requested, use `cf.save(data, new_path)` and `cf.load(new_path)`
  for supported formats. Verify values, timestamps, units, and geometry after reload;
  generic xarray serialization alone is not an index-restoration workflow.
- Report axis mapping, units, geometry/timing source, assumptions, validation, and
  any unverified physical landmarks. Stop at loading; do not add registration,
  filtering, resampling, or analysis unless requested.
