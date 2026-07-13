---
hide:
    - navigation
icon: lucide/history
---

# Changelog

## 0.6.0.dev0

Current development version for the next ConfUSIus release.

### :boom: Breaking changes

- Motion diagnostics helpers now require actual affine matrices: [`extract_motion_parameters`][confusius.registration.extract_motion_parameters], [`compute_framewise_displacement`][confusius.registration.compute_framewise_displacement], and [`create_motion_dataframe`][confusius.registration.create_motion_dataframe] no longer accept `None` placeholders in their affine lists ([#302](https://github.com/confusius-tools/confusius/pull/302)).
- Renamed the public BIDS table I/O helpers to match the rest of ConfUSIus:
  [`read_events`][confusius.bids.load_events] →
  [`load_events`][confusius.bids.load_events], and
  [`write_events`][confusius.bids.save_events] →
  [`save_events`][confusius.bids.save_events]
  ([#294](https://github.com/confusius-tools/confusius/pull/294)).

### :sparkles: Enhancements

- Added [`plot_motion_diagnostics`][confusius.plotting.plot_motion_diagnostics] to visualize motion-correction summaries from `motion_params` tables returned by [`register_volumewise`][confusius.registration.register_volumewise] ([#302](https://github.com/confusius-tools/confusius/pull/302)).
- [`create_motion_dataframe`][confusius.registration.create_motion_dataframe] now always reports all named rotation / translation axes exposed by the affine dimensionality, even when one spatial axis is singleton ([#302](https://github.com/confusius-tools/confusius/pull/302)).
- Added [`load_physio`][confusius.bids.load_physio] to load BIDS physio TSV files with
  column names and metadata from the JSON sidecar, synthesizing a `time` column when
  needed; the napari plugin now uses it for imported signal tables
  ([#294](https://github.com/confusius-tools/confusius/pull/294)).

### :bug: Fixes

- NIfTI loading no longer crashes when a sidecar `VolumeTiming` length disagrees with
  the actual data. ConfUSIus now ignores the malformed sidecar timing, falls back to
  `pixdim[4]` when available, and otherwise warns before using frame indices.
  ([#304](https://github.com/confusius-tools/confusius/pull/304)).
- Motion parameter tables from
  [`create_motion_dataframe`][confusius.registration.create_motion_dataframe] now label
  rotations and translations by the coordinate names `x`/`y`/`z` instead of by raw
  transform-component order, so canonical ConfUSIus arrays stored as `(z, y, x)` no
  longer mislabel in-plane motion
  ([#301](https://github.com/confusius-tools/confusius/pull/301)).
- **[Napari plugin]** The signal import dialog now finds BIDS physio files ending in
  `.tsv.gz`, keeps the x-axis cursor visible for imported-only plots when enabled, and
  lets you import multiple signal files in one go ([#294](https://github.com/confusius-tools/confusius/pull/294)).
- Opening a `.scan` file that is not the legacy HDF5-based Iconeus format now raises a
  clear error that points users to newer SCAN v2 files and to converting them to NIfTI
  with Iconeus tools first ([#297](https://github.com/confusius-tools/confusius/pull/297)).
- Plotting functions now accept a slice dimension reduced to a scalar coordinate by a
  single-index selection, so `plot_contours(atlas.annotation.sel(z=6))` works like
  `sel(z=[6])` ([#296](https://github.com/confusius-tools/confusius/pull/296)).

### :wrench: Maintenance

- [Example Gallery]: pandas DataFrame outputs in the example gallery now render with
  clean, theme-aware notebook styling instead of a fully-bordered table
  ([#307](https://github.com/confusius-tools/confusius/pull/307)).
- [Example Gallery]: Cells can now hide their code behind a collapsed callout with a
  `collapse` cell tag, with optional custom title and type (`collapse[<type>]:
  <title>`), i.e. `# %% tags=["collapse[warning]: Collapsed warning"]`
  ([#309](https://github.com/confusius-tools/confusius/pull/309)).

## 0.5.2

Released 2026-07-10.

### :wrench: Maintenance

- Python 3.14 now keeps `xarray[accel]` everywhere except macOS Intel, where ConfUSIus
  falls back to plain `xarray` to avoid a `numba` / `llvmlite` build failure caused by
  napari's macOS Intel `numba<=0.62.1` cap.

## 0.5.1

Released 2026-07-10.

### :sparkles: Enhancements

- **[Napari plugin]** Added a `File > Open Sample` entries for a [Nunez-Elizalde
  2022](citing.md#nunez-elizalde-et-al-2022) mouse recording and for a pair of [Cybis
  Pereira 2026](citing.md#cybis-pereira-et-al-2026) rat recordings. Samples are fetched on
  demand, shows download progress with an abort button, and only downloads the matching
  raw fUSI files instead of the full dataset
  ([#273](https://github.com/confusius-tools/confusius/pull/273)).
- Dataset fetchers now print the citation to use for the fetched data and accept a
  `print_citation` argument to silence it. The template fetchers
  [`fetch_template_huang_2025`][confusius.datasets.fetch_template_huang_2025] and
  [`fetch_template_pepe_mariani_2026`][confusius.datasets.fetch_template_pepe_mariani_2026]
  also expose the citation on the returned DataArray as `da.attrs["citation"]`
  ([#279](https://github.com/confusius-tools/confusius/pull/279)).
- Dataset fetchers called with `refresh=True` now re-download cached files whose upstream
  MD5 changed, comparing the cached dataset index against the freshly fetched one instead
  of only checking whether the file exists; downloads are additionally verified against
  the index MD5. A locally cached dataset whose `dataset_index.json` predates this format
  is detected on fetch and reported with a clear error naming the directory to delete and
  re-fetch, rather than being silently mishandled. Affects
  [`fetch_cybis_pereira_2026`][confusius.datasets.fetch_cybis_pereira_2026],
  [`fetch_nunez_elizalde_2022`][confusius.datasets.fetch_nunez_elizalde_2022], and
  [`fetch_landemard_2026`][confusius.datasets.fetch_landemard_2026]
  ([#261](https://github.com/confusius-tools/confusius/pull/261)).
- Added [`sample_displacement_field`][confusius.registration.sample_displacement_field],
  [`sample_displacement_field_like`][confusius.registration.sample_displacement_field_like],
  and [`invert_displacement_field`][confusius.registration.invert_displacement_field]
  to sample a B-spline (or composite affine + B-spline) registration transform into a
  dense displacement field and invert it via SimpleITK's
  `InvertDisplacementFieldImageFilter`.
  [`resample_volume`][confusius.registration.resample_volume] and
  [`resample_like`][confusius.registration.resample_like] now also accept displacement
  fields directly, so a saved B-spline transform's inverse can be applied without a
  closed-form inverse
  ([#235](https://github.com/confusius-tools/confusius/pull/235)).

### :bug: Fixes

- Saving to Zarr (via [`save`][confusius.io.save] or `DataArray.fusi.save`) now works
  for data carrying affines or other numpy-valued attributes: nested numpy arrays are
  stored as lists and non-serializable attrs (e.g. matplotlib colormaps) are dropped
  with a warning, matching the NIfTI sidecar behaviour
  ([#284](https://github.com/confusius-tools/confusius/pull/284)).
- B-spline control-point DataArrays returned by
  [`register_volume`][confusius.registration.register_volume] no longer have their
  per-axis grid geometry (spacing, origin, domain) swapped between axes on anisotropic
  images. The bug was invisible on isotropic data, which is why it went unnoticed since
  it shipped [#235](https://github.com/confusius-tools/confusius/pull/235).
- [`plot_napari`][confusius.plotting.plot_napari] no longer sets
  `viewer.scale_bar.unit` (and the napari 0.7.0 `FutureWarning` is gone for good).
  The previous workaround in
  [#271](https://github.com/confusius-tools/confusius/pull/271) is no longer needed:
  napari ≥ 0.7.1 infers the scale bar unit from the layer's `units` attribute, which
  `plot_napari` already forwards from the spatial coordinates.
- [`plot_volume`][confusius.plotting.plot_volume] and friends no longer crash on
  matplotlib ≥ 3.11 when a `threshold` is set. `LinearSegmentedColormap.from_list`
  now requires strictly monotonic `(value, color)` pairs, and the threshold gray
  band could collide with neighbouring cmap entries at the boundary values.
- [`build_atlas_cmap_and_norm`][confusius._utils.atlas.build_atlas_cmap_and_norm]
  no longer calls the matplotlib-3.11-deprecated `set_under`/`set_over`/`set_bad`
  colormap methods, and no longer passes the deprecated `N=` argument to
  `ListedColormap`. The under colour is now passed as a constructor kwarg, the
  matplotlib-3.11-recommended way to set it.
- [`plot_volume`][confusius.plotting.plot_volume],
  [`plot_stat_map`][confusius.plotting.plot_stat_map],
  [`plot_composite`][confusius.plotting.plot_composite], and
  [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours] no
  longer silently reorder panels when `slice_mode`'s own coordinate isn't already
  sorted (e.g. a `region` dimension built from an arbitrary list of acronyms, or a
  descending `z`). Only the two display dimensions are sorted for plotting geometry now
  ([#268](https://github.com/confusius-tools/confusius/pull/268)).

### :books: Documentation

- The [same-subject registration
  example](examples/_built/registration/register_volume_same_subject.md) now follows
  the rigid registration step with a B-spline refinement, showing the extra local
  correction it adds and how its parameters differ from the rigid step's
  ([#235](https://github.com/confusius-tools/confusius/pull/235)).
- Long output in gallery examples—warnings, text reprs, tracebacks, and rich-rendered
  text such as the dataset citation banner—now wraps instead of showing a horizontal
  scrollbar ([#285](https://github.com/confusius-tools/confusius/pull/285)).

### :wrench: Maintenance

- Raised the minimum supported versions to **napari 0.7.1** and
  **matplotlib 3.11**.
- The example-gallery build tool now accepts specific example scripts as arguments
  (`uv run python tools/build_gallery.py docs/examples/01_io/01_confusius_xarray_101.py`),
  running only those; the rest of the gallery is still rendered, taken from cache if
  present or built without outputs
  ([#285](https://github.com/confusius-tools/confusius/pull/285)).

## 0.5.0

Released 2026-07-07.

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

- Added [`plot_matrix`][confusius.plotting.plot_matrix] for plotting 2D matrices
  (e.g. connectivity or correlation matrices), with optional lower/diagonal triangle
  masking, grid lines, and a `groups` parameter that annotates contiguous label runs
  with colored rectangle strips—useful for marking anatomical groupings (e.g. cortex,
  thalamus) when there are too many individual labels to read
  ([#243](https://github.com/confusius-tools/confusius/pull/243)).
- **[Napari plugin]** Integer-dtype files (e.g. atlas annotations, ROI masks) opened via
  the `confusius` CLI, the Data Panel, or the native napari file readers (drag-and-drop /
  **File > Open**) are now added as a `Labels` layer with per-label colors, instead of an
  `Image` layer with the wrong colormap
  ([#257](https://github.com/confusius-tools/confusius/pull/257)).
- **[Napari plugin]** Added **Events** panel to annotate temporal events within Napari.
  Events shade the signal plot; active event names appear in the time overlay; load
  from / save to a BIDS `.tsv`
  ([#176](https://github.com/confusius-tools/confusius/pull/176)).
- [`confusius.bids`][confusius.bids] module is now public with new
  [`load_events`][confusius.bids.load_events] and
  [`save_events`][confusius.bids.save_events]
  ([#176](https://github.com/confusius-tools/confusius/pull/176)).
- Added a `datasets` CLI namespace, listed in `confusius --help`:
  `confusius datasets --list` prints the table of available datasets, their sizes,
  and whether each is cached on disk. A bare `confusius PATH...` still launches the
  viewer ([#234](https://github.com/confusius-tools/confusius/pull/234)).
- Added [`NMF`][confusius.decomposition.NMF] for non-negative matrix factorization
  of fUSI time series, wrapping `sklearn.decomposition.NMF` with the same
  xarray-aware `fit`/`transform`/`inverse_transform` interface as
  [`PCA`][confusius.decomposition.PCA] and [`FastICA`][confusius.decomposition.FastICA].
  Both `mode='temporal'` and `mode='spatial'` are supported
  ([#211](https://github.com/confusius-tools/confusius/pull/211)).
- Added [`adjust_pvalues`][confusius.stats.adjust_pvalues] for generic
  multiple-comparison correction of p-value maps, and
  [`apply_statistical_threshold`][confusius.stats.apply_statistical_threshold] to
  threshold z-scaled statistical DataArrays with the same family-wise-error
  (Bonferroni, Šidák, Holm, Holm-Šidák, Simes-Hochberg, Hommel) or
  false-discovery-rate (Benjamini-Hochberg, Benjamini-Yekutieli) corrections, plus an
  optional cluster-extent threshold
  ([#204](https://github.com/confusius-tools/confusius/pull/204)).
- Added [`fetch_landemard_2026`][confusius.datasets.fetch_landemard_2026] for
  downloading the Landemard et al. (2026) fUSI-BIDS dataset from OSF, with
  `datasets`, `subjects`, `acqs`, and `datatypes` filters
  ([#228](https://github.com/confusius-tools/confusius/pull/230)).
- Added [`plot_stat_map`][confusius.plotting.plot_stat_map] (and the matching
  `data.fusi.plot.stat_map` accessor) for plotting statistical maps, optionally
  overlaid fully opaque on a background anatomical volume. `vmin`/`vmax` default to
  the data's actual min/max, and `auto_range=True` (default) picks both the
  colormap range and colormap from the data's sign: diverging symmetric
  `[-m, m]` with `"coolwarm"` when both signed, sequential `[0, vmax]` with
  `"viridis"` when non-negative, or `[vmin, 0]` with `"viridis_r"` when
  non-positive ([#242](https://github.com/confusius-tools/confusius/pull/242)).
- [`plot_volume`][confusius.plotting.plot_volume] and
  [`plot_stat_map`][confusius.plotting.plot_stat_map] (and their `data.fusi.plot.*`
  accessors) now accept `cbar_kwargs`, forwarded to
  [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]—useful to
  shrink a shared colorbar down to size on a multi-panel grid
  ([#242](https://github.com/confusius-tools/confusius/pull/242)).
- [`apply_affine`][confusius.xarray.affine.apply_affine] and the
  `data.fusi.affine.apply` accessor now accept a string naming a key in
  `attrs["affines"]`, instead of requiring the affine matrix itself
  ([#247](https://github.com/confusius-tools/confusius/pull/247)).
- Plotting functions that slice along `slice_mode` (
  [`plot_volume`][confusius.plotting.plot_volume],
  [`plot_contours`][confusius.plotting.plot_contours],
  [`plot_composite`][confusius.plotting.plot_composite], and the
  [`VolumePlotter`][confusius.plotting.VolumePlotter] methods) now support non-numeric
  coordinates (e.g. region/mask labels), so a single call can slice a stacked
  connectivity or ROI map by label instead of looping over `.sel()` per panel
  ([#250](https://github.com/confusius-tools/confusius/pull/250)).

### :bug: Fixes

- [`plot_volume`][confusius.plotting.plot_volume] and other image plotting functions now
  raise a clear `ValueError` when `vmin`/`vmax` (or a passed-in `norm`) resolve to a
  non-finite value, instead of crashing deep inside
  `matplotlib.colors.LinearSegmentedColormap.from_list` with an opaque `IndexError`
  ([#259](https://github.com/confusius-tools/confusius/pull/259)).
- [`save_nifti`][confusius.io.save_nifti] now drops attrs that cannot be serialized to
  JSON as-is (e.g. matplotlib `ListedColormap`/`BoundaryNorm` objects) instead of
  writing their `str()` repr into the sidecar, which could corrupt fields such as `cmap`
  on reload. A warning lists the dropped keys. [`confusius.load`][confusius.io.load] now
  rebuilds `cmap`/`norm` from `rgb_lookup` when they are missing, so atlas-derived masks
  and annotations keep their canonical colors after a save/load round-trip. **[Napari
  plugin]** The reader now falls back to the `"gray"` colormap (with a napari warning)
  instead of crashing when a layer's `cmap` attr is not a valid napari colormap name
  ([#255](https://github.com/confusius-tools/confusius/pull/255)).
- [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks] now suffixes the `mask`
  coordinate with `_L`/`_R` for `sides="left"`/`"right"`, so requesting the same region
  on both hemispheres no longer produces duplicate `mask` values.
  [`extract_with_labels`][confusius.extract.extract_with_labels] no longer requires
  unique region ids across stacked mask layers—a layer is already identified by its
  position along `mask`—so `get_masks` output can be passed straight through without
  manual relabeling ([#249](https://github.com/confusius-tools/confusius/pull/249)).
- [`apply_affine`][confusius.xarray.apply_affine] now rescales the `voxdim` attribute
  of the spatial coordinates along with the coordinate values
  ([#245](https://github.com/confusius-tools/confusius/pull/245)).
- [`clean`][confusius.signal.clean] now supports `ensure_finite=True` to repair
  non-finite `signals` and `confounds` by interpolating along time, fills censored
  boundary samples from the nearest kept sample before filtering, and accepts
  `interpolate_kwargs` for pre-scrubbing interpolation
  ([#239](https://github.com/confusius-tools/confusius/pull/239)).
- Image plotting functions now leave `alpha` unset by default (`None`), so a
  colormap's built-in alpha channel is respected
  ([#225](https://github.com/confusius-tools/confusius/pull/225)).
- [`load_nifti`][confusius.io.load_nifti] no longer drops affines loaded from the JSON
  sidecar (e.g. `bspline_initialization` written by the registration pipeline) when
  merging in the NIfTI qform/sform affines
  ([#222](https://github.com/confusius-tools/confusius/pull/222)).
- [`save_nifti`][confusius.io.save_nifti] no longer maps non-time additional axes to the
  NIfTI 4th slot. When additional axes are present in the DataArray, a degenerate
  length-1 `time` axis is inserted at NIfTI axis 4 (NIfTI's conventional time slot) so
  non-time additional axes always land at NIfTI axes 5, 6, 7. The original dim name for
  each additional axis is always written to the sidecar as `ConfUSIusDim{N}Name` (with
  `N` in 4, 5, 6, matching the 0-based NIfTI axis of the extra dim). The matching
  `ConfUSIusDim{N}Coordinates` entry is only written when the coord cannot be
  reconstructed from `pixdim` (i.e. when the coord does not start at 0 with regular
  spacing); otherwise the spacing is stored in `pixdim` and the coord is rebuilt as
  `step * arange(size)` on load. Attributes are preserved in `ConfUSIusDim{N}Attributes`
  entries. ([#223](https://github.com/confusius-tools/confusius/pull/223)).

### :books: Documentation

- Add an [NMF example](examples/_built/decomposition/nmf_single_recording.md) to the
  gallery, demonstrating the z-score + absolute-value standardization that makes
  signed fUSI signals NMF-compatible.
- Add an [atlas-based region correlation matrix
  example](examples/_built/connectivity/atlas_correlation_matrix.md) to the gallery,
  demonstrating registration to the Pepe-Mariani 2026 template, resampling the Allen
  Mouse Brain Atlas onto a recording's native grid, and plotting a region correlation
  matrix with [`plot_matrix`][confusius.plotting.plot_matrix]'s `groups` annotation
  ([#243](https://github.com/confusius-tools/confusius/pull/243)).

### :wrench: Maintenance

- Simplified the NIfTI save path: time and extra-dimension voxel spacings are now
  written directly to the header `pixdim` instead of through nibabel's `set_zooms` (that
  was overwritten anyways), dropping a redundant spatial write that the qform
  immediately overwrote. Behavior is unchanged.
  ([#253](https://github.com/confusius-tools/confusius/pull/253)).

## 0.4.0

*Released 2026-06-25.*

### :sparkles: Enhancements

- The `confusius` CLI now accepts multiple fUSI data files in a single
  invocation (e.g. `confusius fixed.nii moving.nii`). Each file is added as its
  own image layer, named after the file's basename
  ([#206](https://github.com/confusius-tools/confusius/pull/206)).
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
