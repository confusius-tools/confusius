---
hide:
    - navigation
icon: lucide/history
---

# Changelog

## 0.3.0.dev0

Current development version for the next ConfUSIus release.

### :boom: Breaking changes

- `register_volume` now returns a 3-tuple
  `(registered, transform, diagnostics)` instead of a 2-tuple. Existing call sites
  that unpack the two-element return value need to add a third variable (or
  `_`)
  ([#139](https://github.com/confusius-tools/confusius/pull/139)).

### :sparkles: Enhancements

- `register_volume` now also returns a
  [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics] dataclass
  with the per-iteration metric values, final metric value, iteration count, optimizer
  stop condition, and the metric name. `register_volumewise` propagates the per-frame
  diagnostics list under `attrs["registration_diagnostics"]` and adds
  `final_metric_value` and `n_iterations` columns to `motion_params`
  ([#139](https://github.com/confusius-tools/confusius/pull/139)).
- Added `show_progress` to volumewise registration so joblib progress output can be
  disabled in scripted or quiet workflows
  ([#126](https://github.com/confusius-tools/confusius/pull/126)).
- Replaced plotting `black_bg` with explicit `bg_color` and `fg_color` controls for
  clearer visual customization ([#124](https://github.com/confusius-tools/confusius/pull/124)).
- Added example gallery helper utilities to streamline writing and maintaining docs
  examples ([#102](https://github.com/confusius-tools/confusius/pull/102)).

### :books: Documentation

- Added a [Registering two acquisitions](examples/_built/registration/register_volume_two_acquisitions.md)
  example demonstrating `register_volume`, the new diagnostics, and confusius's
  [`plot_volume`][confusius.plotting.plot_volume] overlay pattern for inspecting
  alignment before and after registration
  ([#139](https://github.com/confusius-tools/confusius/pull/139)).

### :bug: Fixes

- Fixed napari x-axis extent computation to ignore the interactive cursor guide line,
  preventing incorrect plot bounds
  ([#111](https://github.com/confusius-tools/confusius/pull/111)).

### :zap: Performance

- Top-level `confusius` and `confusius.xarray` namespaces now use
  [SPEC-0001](https://scientific-python.org/specs/spec-0001/) PEP 562 lazy loading.
  Submodules and exported functions are only imported on first access, reducing `import
  confusius` overhead for workflows that use a subset of the package.

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
