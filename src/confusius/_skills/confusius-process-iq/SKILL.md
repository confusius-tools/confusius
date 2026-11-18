---
name: confusius-process-iq
description: Process beamformed complex IQ with ConfUSIus into B-mode, power Doppler, axial velocity, or custom windowed measurements and filters. Use for IQ timing/windows, clutter filtering, SVD cutoffs, custom block processing, and Dask execution.
compatibility: Requires Python and ConfUSIus; Dask and threadpoolctl for threaded examples.
---

# Process beamformed IQ with ConfUSIus

Prefer the installed API, source, and tests over published docs for unreleased changes.
Use `confusius-load-data` for file imports or constructing VoxelData. This skill starts
with **beamformed complex IQ**, not raw RF or per-channel signals.

## Validate input

- Require complex VoxelData with `(time, k, j, i)` dimensions and voxel-to-world
  geometry; singleton spatial axes are allowed. Process poses separately; built-in
  wrappers reject extra dimensions. Never discard the imaginary part.
- Check timestamps, acquisition duration/reference, compound IQ rate, and chunks.
  **Use compound IQ timing, not transmit PRF.** Do not let windows cross acquisition gaps.
- Velocity additionally needs `iq.attrs["transmit_frequency"]` (actual transmit
  frequency, Hz) and `iq.attrs["beamforming_sound_velocity"]` (m/s). Inspect acquisition
  metadata and ask about unresolved values instead of guessing. These attrs are not
  required for B-mode or power Doppler.

## Choose a measurement

| Goal | `cf.iq` function | Result |
| --- | --- | --- |
| Tissue anatomy | `process_iq_to_bmode` | Mean magnitude; linear arbitrary units, no clutter filter. |
| Blood signal power | `process_iq_to_power_doppler` | Mean squared magnitude after filtering, not absolute blood volume. |
| Blood/tissue motion | `process_iq_to_axial_velocity` | Signed beam-axis velocity (m/s), not total flow speed. |
| Custom processing | `process_iq_blocks` | Bare Dask array; see [custom processing](references/custom-processing.md). |

The built-ins also have `iq.fusi.iq.process_to_*` accessor forms.

## Set windows and filters

Set widths explicitly: they count compound IQ **frames**, not seconds. Doppler/velocity
clutter filter outer windows, then integrate inner windows; B-mode doesn't use clutter
filters, so only one window level. See [temporal
windows](references/temporal-windows.md).

For blood power, `filter_method="svd_indices"` with a tuned `low_cutoff` is a starting
approach. `low_cutoff=N` removes the `N` strongest components; omitting both cutoffs
removes none. See [clutter filtering](references/clutter-filtering.md) for adaptive
alternatives, masks, and motion-artifact trade-offs.

## Process and save

```python
import confusius as cf

# Parameters are illustrative.
pwd = cf.iq.process_iq_to_power_doppler(iq, clutter_window_width=200, low_cutoff=40)
cf.save(pwd, "power_doppler.zarr")
```

Test representative windows first. Keep large recordings Dask-backed and save without
first computing the full output in RAM. See [performance](references/performance.md)
for bounded threaded execution, BLAS limits, and distributed resources.

## Check results

- Inspect quiet and motion-rich maps/time courses for residual clutter and lost blood
  signal; involve the user when the scientific trade-off is unclear.
- Check timestamps, integration durations, gaps, dropped/overlapping windows, and
  warnings about approximate timing metadata.
- Calibrate velocity sign against known motion/flow. Keep `0 < lag < velocity_window_width`;
  larger lags reduce unambiguous velocity range. `spatial_kernel=1` is the current
  default and disables spatial median filtering.
- Keep linear values for analysis. For display use `.fusi.scale.db(factor=20)` for
  B-mode and `.fusi.scale.db()` for power Doppler; never log signed measurements.
