---
hide:
    - navigation
icon: lucide/history
---

# Changelog

## 0.5.0.dev0

Current development version for the next ConfUSIus release.

### :boom: Breaking changes

- Registration now takes a single `initialization` parameter in place of
  `centering_initialization` and `initial_transform`. `initialization` accepts
  `"center_geometry"`, `"center_moments"`, a homogeneous affine matrix or `None` for an
  identity initialization. Affects
  [`register_volume`][confusius.registration.register_volume],
  [`register_volumewise`][confusius.registration.register_volumewise], and the
  `data.fusi.register` accessor
  ([#215](https://github.com/confusius-tools/confusius/pull/215)).

### :sparkles: Enhancements

- Added [`NMF`][confusius.decomposition.NMF] for non-negative matrix factorization
  of fUSI time series, wrapping `sklearn.decomposition.NMF` with the same
  xarray-aware `fit`/`transform`/`inverse_transform` interface as
  [`PCA`][confusius.decomposition.PCA] and [`FastICA`][confusius.decomposition.FastICA].
  Both `mode='temporal'` and `mode='spatial'` are supported
  ([#117](https://github.com/confusius-tools/confusius/issues/117)).
- Added [`adjust_pvalues`][confusius.stats.adjust_pvalues] for generic
  multiple-comparison correction of p-value maps, and
  [`apply_statistical_threshold`][confusius.stats.apply_statistical_threshold] to
  threshold z-scaled statistical DataArrays with the same family-wise-error
  (Bonferroni, Šidák, Holm, Holm-Šidák, Simes-Hochberg, Hommel) or
  false-discovery-rate (Benjamini-Hochberg, Benjamini-Yekutieli) corrections, plus an
  optional cluster-extent threshold
  ([#204](https://github.com/confusius-tools/confusius/pull/204)).

### :bug: Fixes

- Image plotting functions now leave `alpha` unset by default (`None`), so a
  colormap's built-in alpha channel is respected
  ([#225](https://github.com/confusius-tools/confusius/pull/225)).
- `load_nifti` no longer drops affines loaded from the JSON sidecar (e.g.
  `bspline_initialization` written by the registration pipeline) when merging in the
  NIfTI qform/sform affines
  ([#222](https://github.com/confusius-tools/confusius/pull/222)).
- `save_nifti` no longer maps non-time additional axes to the NIfTI 4th slot. When
  additional axes are present in the DataArray, a degenerate length-1 `time` axis is
  inserted at NIfTI axis 4 (NIfTI's conventional time slot) so non-time additional axes
  always land at NIfTI axes 5, 6, 7. The original dim name for each additional axis is
  always written to the sidecar as `ConfUSIusDim{N}Name` (with `N` in 4, 5, 6, matching
  the 0-based NIfTI axis of the extra dim). The matching `ConfUSIusDim{N}Coordinates`
  entry is only written when the coord cannot be reconstructed from `pixdim` (i.e. when
  the coord does not start at 0 with regular spacing); otherwise the spacing is stored
  in `pixdim` and the coord is rebuilt as `step * arange(size)` on load. Attributes are
  preserved in `ConfUSIusDim{N}Attributes` entries.
  ([#223](htttps://github.com/confusius-tools/confusius/pull/223)).

### :books: Documentation

- Add an [NMF example](examples/_built/decomposition/nmf_single_recording.md) to the
  gallery, demonstrating the z-score + absolute-value standardization that makes
  signed fUSI signals NMF-compatible.

## 0.4.0

*Released 2026-06-25.*

### :sparkles: Enhancements

- The `confusius` CLI now accepts multiple fUSI data files in a single
  invocation (e.g. `confusius fixed.nii moving.nii`). Each file is added as its
  own image layer, named after the file's basename
  ([#205](https://github.com/confusius-tools/confusius/issues/205)).
- `data.fusi.affine.apply` now accepts affines with rotation and shear. The
  axis-aligned part updates the 1D `z`/`y`/`x` coordinates and the method returns
  the residual orientation as a 4x4 affine (the identity for diagonal affines)
  for the caller to use as they wish
  ([#188](https://github.com/confusius-tools/confusius/pull/188)).
- Add `smoothing_fwhm` parameter to [`FirstLevelModel`][confusius.glm.FirstLevelModel].
  Smoothing is applied to each run before model fitting
  ([#201](https://github.com/confusius-tools/confusius/pull/201)).

### :zap: Performance

- [`process_iq_blocks`][confusius.iq.process.process_iq_blocks] now uses
  `dask.array.map_blocks` for non-overlapping outer IQ windows and batches
  overlapping windows with explicit overlap before mapping blocks, reducing Dask
  overhead in common blockwise processing workflows
  ([#190](https://github.com/confusius-tools/confusius/pull/190)).

### :bug: Fixes

- Masks are now coerced to boolean by `validate_mask` (added `return_dtype_as_bool`
  parameter that defaults to `True`) to avoid DataArrays using *positional indexing*.
  Previously these masks could select the wrong voxels or, for `register_volume`,
  silently disable the metric mask
  ([#197](https://github.com/confusius-tools/confusius/pull/197)).
- `process_iq_blocks` now handles strongly overlapping IQ windows without corrupting
  the output time dimension, so power Doppler and related IQ reducers work when
  `window_stride < window_width / 2`
  ([#192](https://github.com/confusius-tools/confusius/pull/192)).
- `load_nifti` now anchors `physical_to_qform` to the same physical frame as the
  primary (sform) coordinates, so the stored qform affine maps the array's
  physical coordinates to qform world space
  ([#187](https://github.com/confusius-tools/confusius/pull/187)).
- `save_nifti` now preserves each affine's own translation, so a NIfTI file with sform
  and qform round-trips through `load_nifti`/`save_nifti` without corrupting the qform
  ([#187](https://github.com/confusius-tools/confusius/pull/187)).
- **[Napari plugin]** Fixed the Signals plot x-axis for volumes without a time
  dimension. It now follows the slider axis world coordinates, with a matching label and
  dropdown option ([#180](https://github.com/confusius-tools/confusius/pull/180)).

## 0.3.0

*Released 2026-05-27.*

### :boom: Breaking changes

- `register_volume` now also returns a
  [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics] dataclass
  with the per-iteration metric values, final metric value, iteration count, optimizer
  stop condition, and the metric name. `register_volumewise` always adds per-frame
  `final_metric_value` and `n_iterations` columns to `motion_params`, and exposes the
  full per-frame diagnostics list under `attrs["registration_diagnostics"]` only when
  called with `keep_diagnostics=True` to avoid retaining the full optimizer metric
  trace by default ([#139](https://github.com/confusius-tools/confusius/pull/139)).
- Renamed `validate_iq` to
  [`validate_iq_dataarray`][confusius.validation.validate_iq_dataarray]
  ([#153](https://github.com/confusius-tools/confusius/pull/153)).

### :sparkles: Enhancements

- Added a `mask` argument to the [`PCA`][confusius.decomposition.PCA],
  [`FastICA`][confusius.decomposition.FastICA],
  [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps], and
  [`FirstLevelModel`][confusius.glm.FirstLevelModel] estimators, restricting fitting
  and projection to the selected voxels. Output maps retain the full spatial geometry,
  with voxels outside the mask set to `0`
  ([#155](https://github.com/confusius-tools/confusius/pull/155)).
- Added `plot_composite`, `VolumePlotter.add_composite`, and a matching
  `data.fusi.plot.composite` accessor that render two volumes as a red/cyan
  RGB overlay ([#145](https://github.com/confusius-tools/confusius/pull/145)).
- Added `datatypes` filter to `fetch_cybis_pereira_2026`, allowing downloads to be
  scoped to specific BIDS datatype directories (`"fusi"`, `"angio"`, `"motion"`)
  ([#141](https://github.com/confusius-tools/confusius/pull/141)).
- Added `fetch_template_huang_2025` for downloading and loading the Huang et al.
  vascular mouse template from OSF, with cache/refresh behavior matching existing
  template fetchers ([#162](https://github.com/confusius-tools/confusius/pull/162)).
- Added `show_progress` to volumewise registration so joblib progress output can be
  disabled in scripted or quiet workflows
  ([#126](https://github.com/confusius-tools/confusius/pull/126)).
- Added a reusable [`validate_fusi_dataarray`][confusius.validation.validate_fusi_dataarray]
  validator and refactored IQ/registration validation to use it. Core dimension
  coordinates are now validated as 1D, numeric, finite, and strictly increasing,
  while extra/non-dimension coordinates remain allowed
  ([#153](https://github.com/confusius-tools/confusius/pull/153)).
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

### :zap: Performance

- Top-level `confusius` and `confusius.xarray` namespaces now use
  [SPEC-0001](https://scientific-python.org/specs/spec-0001/) PEP 562 lazy loading.
  Submodules and exported functions are only imported on first access, reducing `import
  confusius` overhead for workflows that use a subset of the package.

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

### :books: Documentation

- Added a [Registration of two sessions from the same
  subject](examples/registration/register_volume_same_subject.py) example
  demonstrating `register_volume`, the new diagnostics, and confusius's
  [`plot_composite`][confusius.plotting.plot_composite] overlay pattern for
  inspecting alignment before and after registration
  ([#139](https://github.com/confusius-tools/confusius/pull/139)).

### :wrench: Maintenance

- Switched documentation hosting to GitHub Pages with `mike` versioning and automatic
  PR preview deployments
  ([#134](https://github.com/confusius-tools/confusius/pull/134)).

## 0.2.0

*Released 2026-05-05.*

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
