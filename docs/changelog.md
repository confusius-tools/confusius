---
hide:
    - navigation
icon: lucide/history
---

# Changelog

## 0.3.0.dev0

Current development version for the next ConfUSIus release.

### :boom: Breaking changes

- `register_volume` now also returns a
  [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics] dataclass
  with the per-iteration metric values, final metric value, iteration count, optimizer
  stop condition, and the metric name. `register_volumewise` always adds per-frame
  `final_metric_value` and `n_iterations` columns to `motion_params`, and exposes the
  full per-frame diagnostics list under `attrs["registration_diagnostics"]` only when
  called with `keep_diagnostics=True` to avoid retaining the full optimizer metric
  trace by default ([#139](https://github.com/confusius-tools/confusius/pull/139)).

### :sparkles: Enhancements

- Added `plot_composite`, `VolumePlotter.add_composite`, and a matching
  `data.fusi.plot.composite` accessor that render two volumes as a red/cyan
  RGB overlay ([#145](https://github.com/confusius-tools/confusius/pull/145)).
- Added `datatypes` filter to `fetch_cybis_pereira_2026`, allowing downloads to be
  scoped to specific BIDS datatype directories (`"fusi"`, `"angio"`, `"motion"`)
  ([#141](https://github.com/confusius-tools/confusius/pull/141)).
- Added `show_progress` to volumewise registration so joblib progress output can be
  disabled in scripted or quiet workflows
  ([#126](https://github.com/confusius-tools/confusius/pull/126)).
- Added shared `fontsize` parameter to `plot_volume`, `plot_contours`, and carpet
  plotting entry points so text sizing is consistent across all plotting APIs
  ([#128](https://github.com/confusius-tools/confusius/pull/128)).
- Replaced plotting `black_bg` with explicit `bg_color` and `fg_color` controls for
  clearer visual customization ([#124](https://github.com/confusius-tools/confusius/pull/124)).
- Added [`FastICA`][confusius.decomposition.FastICA] transformer for independent
  component analysis of fUSI recordings, with the same xarray-aware `fit` /
  `transform` / `inverse_transform` API as [`PCA`][confusius.decomposition.PCA]
  ([#118](https://github.com/confusius-tools/confusius/pull/118)).
- Added example gallery helper utilities to streamline writing and maintaining docs
  examples ([#102](https://github.com/confusius-tools/confusius/pull/102)).

### :books: Documentation

- Added a [Registering two
  acquisitions](examples/registration/register_volume_two_acquisitions.py) example
  demonstrating `register_volume`, the new diagnostics, and confusius's
  [`plot_volume`][confusius.plotting.plot_volume] overlay pattern for inspecting
  alignment before and after registration
  ([#139](https://github.com/confusius-tools/confusius/pull/139)).

### :bug: Fixes

- Fixed `resample_like` and `resample_volume` filling out-of-FOV voxels with `0.0`
  when resampling onto a larger grid. This caused a bright background artifact for
  dB-scaled data (where 0 is maximum intensity). The `default_value` parameter now
  defaults to `float(moving.min())` instead of `0.0`. `register_volume` gains a
  `fill_value` parameter that overrides the default for both the final resampled output
  and the live progress composite overlay
  ([#138](https://github.com/confusius-tools/confusius/pull/138)).
- Fixed plotting hover information silently disappearing when the returned
  `VolumePlotter` was not held in a variable (e.g. `obj.fusi.plot.volume().show()`). The
  hover manager is now kept alive until the figure is closed
  ([#148](https://github.com/confusius-tools/confusius/pull/148)).
- Fixed napari x-axis extent computation to ignore the interactive cursor guide line,
  preventing incorrect plot bounds
  ([#111](https://github.com/confusius-tools/confusius/pull/111)).

### :zap: Performance

- Top-level `confusius` and `confusius.xarray` namespaces now use
  [SPEC-0001](https://scientific-python.org/specs/spec-0001/) PEP 562 lazy loading.
  Submodules and exported functions are only imported on first access, reducing `import
  confusius` overhead for workflows that use a subset of the package.

### :wrench: Maintenance

- Switched documentation hosting to GitHub Pages with `mike` versioning and automatic
  PR preview deployments
  ([#134](https://github.com/confusius-tools/confusius/pull/134)).

## 0.2.0

Released 2026-05-05.

First official public beta release of ConfUSIus.

### :sparkles: Highlights

- ConfUSIus now covers the core alpha roadmap, including I/O, beamformed IQ processing,
  registration, quality control, atlas integration, signal processing, decomposition,
  functional connectivity, and general linear model workflows.
- The package provides both a Python API and a napari plugin for interactive data
  loading, visualization, signal inspection, and quality control.

### :memo: Notes

- `0.1.0` was used only to reserve the `confusius` project name on PyPI and is not a
  supported public release. `0.2.0` is therefore the first official public release
  series for ConfUSIus.
