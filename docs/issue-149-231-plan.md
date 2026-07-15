# Plan for issues #149 and #231

## Implementation status (branch `feat/issue-149-231-zyx`)

Done and merged into the branch (full test suite green):

- **Constructor helper (§1)** — `create_fusi_dataarray` in
  `src/confusius/xarray/create.py`, exported as `confusius.create_fusi_dataarray` and
  `confusius.xarray.create_fusi_dataarray`. Builds canonical `(…, z, y, x)` arrays with
  regularly spaced physical coordinates and `units`/`voxdim` metadata, and validates the
  result. Beyond the plan it also accepts `volume_acquisition_reference` (default
  `"start"`, validated) and `volume_acquisition_duration` (defaults to the time spacing),
  written to the `time` coordinate attrs. DataArray-level metadata such as
  `beamforming_sound_velocity` / `transmit_frequency` flows through the `attrs` argument
  (no dedicated parameters).
- **Validator (§2)** — `validate_fusi_dataarray` now requires the full `z, y, x` trio and
  the `minimum_spatial_dims` parameter was removed. The singleton axis may be any of
  `z`/`y`/`x` (the validator only checks presence, not which axis is length 1).
- **Call-site migration (§3, §4)** — every direct caller (registration ×5, IQ validation,
  smoothing) updated. `smooth_volume` is treated as a fUSI-native API (requires `z, y, x`).
- **2D removal (extends §4)** — beyond the original plan, all internal 2D SimpleITK / 2D
  affine handling was removed: `motion.py` (`_decompose_affine_2d` and the 2D
  extract/validate/column branches gone; affines are `(4, 4)`-only), `volume.py` (the
  `Euler2DTransform` branch), and `affines.py` (`Euler2DTransform` /
  `Similarity2DTransform` references). Registration is 3D-only; `register_volume`
  transforms and motion affines are always `(4, 4)`; motion params use the 3D schema
  (`rot_x/y/z`, `trans_x/y/z`).
- **Motion-correction default fix** — `register_volumewise` default `learning_rate`
  changed `0.01` → `1.0` (and the napari time-series panel to match). The old default was
  too small to recover realistic inter-frame shifts within the iteration budget — a
  general defect, **not** specific to single-slice data (verified by MWE). `register_volume`
  was already fine (`"auto"`).
- **Test migration (§9, §11)** — fUSI-semantic fixtures/tests across validation,
  registration, napari, and IQ moved to singleton-`z`; the progress-plotter tests moved to
  3D SimpleITK images. Genuinely-generic 2D tests (matrix plotting, stats, GLM maps) left
  as real 2D (§12).
- **Docs** — changelog entries (breaking change, `create_fusi_dataarray`, learning-rate
  fix); public registration docstrings updated to "3D volume / singleton-`z`".

Not yet done (tracked for follow-up):

- **§6 extraction / connectivity** — `extract/labels.py`, `extract/mask.py`,
  `connectivity/seed.py` not deliberately audited for "prefer canonical `(z, y, x)`"; the
  suite passes and they are dimension-agnostic, but no targeted review was made.
- **§7 `docs/user-guide/xarray.md`** — not yet updated to stop presenting `(y, x)` as a
  canonical fUSI recording.
- **§8 doc wording** — cosmetic leftovers only (`visualization.md` "2D+t" alt text; a
  `spatial-conventions.md` display-slice example). IO guide and examples already use
  `(time, z, y, x)`.

## Summary

We will handle these two issues together:

- **#149**: add a helper to create ConfUSIus-style fUSI `xarray.DataArray` objects from raw arrays plus higher-level metadata like `dt`, `dz`, `dy`, and `dx`.
- **#231**: make `(z, y, x)` the minimum required spatial axes for fUSI DataArrays across ConfUSIus.

The key design choice is:

- **single-slice data is still represented as 3D**, with a singleton `z` axis;
- canonical time-varying single-slice data is therefore `(time, z, y, x)` with `z=1`, not `(time, y, x)`.

## Target invariant

A ConfUSIus fUSI DataArray should:

- always contain spatial dimensions **`z`, `y`, `x`**;
- optionally contain **`time`**;
- optionally contain extra non-core dimensions when the API allows them;
- preserve spatial metadata on all three spatial axes, including `units` and `voxdim` where relevant.

This means:

- valid spatial-only data: `(z, y, x)`;
- valid time-varying data: `(time, z, y, x)`;
- valid single-slice data: `(1, y, x)` or `(time, 1, y, x)`;
- invalid fUSI data representations: `(y, x)` and `(time, y, x)`.

## Scope split

We will distinguish between two kinds of APIs.

### A. fUSI-native APIs

These should enforce the new invariant.

Examples:

- fUSI validation helpers;
- IO that produces fUSI DataArrays;
- registration / resampling / motion correction;
- fUSI creation helpers;
- other APIs whose contract is explicitly “fUSI volume / recording”.

### B. generic array / map / slice APIs

These should continue to accept real 2D arrays when they are not representing full fUSI volumes.

Examples:

- matrix plotting;
- generic statistical maps;
- already-sliced `(y, x)` images;
- coordinate/unit comparison helpers;
- other utilities whose contract is “generic DataArray”, not “fUSI recording”.

The rule is:

- if an API means **fUSI data**, require `z`, `y`, `x`;
- if an API means **generic 2D/3D array**, keep 2D support.

## Implementation plan

## 1. Add a fUSI DataArray constructor helper

Add a single public helper to build valid fUSI DataArrays from raw arrays.

Proposed shape:

```python
create_fusi_dataarray(
    data,
    *,
    dims,
    dt=None,
    dz=None,
    dy=None,
    dx=None,
    t0=0.0,
    z0=0.0,
    y0=0.0,
    x0=0.0,
    name=None,
    attrs=None,
)
```

Requirements:

- `dims` is explicit and required;
- no implicit dim inference from array rank;
- builds 1D coordinates for present dimensions;
- `time` gets `attrs={"units": "s"}`;
- `z`, `y`, `x` get `attrs={"units": "mm", "voxdim": ...}`;
- validates the result before returning it.

Validation should ensure the helper returns a canonical ConfUSIus fUSI DataArray.

## 2. Tighten `validate_fusi_dataarray`

Update `src/confusius/validation/fusi.py` so that ConfUSIus fUSI validation requires the canonical spatial trio.

Planned behavior:

- require presence of **all of `z`, `y`, `x`**;
- keep `time` optional unless requested;
- keep optional extra dims behavior;
- preserve existing metadata validation options (`units`, `voxdim`, regular spacing, canonical order, etc.).

Important detail:

- do **not** rely only on “minimum number of spatial dims”; require the actual named dims `z`, `y`, and `x`.

## 3. Audit all direct validator call sites

Direct callers already identified:

- `src/confusius/spatial/smooth.py`
- `src/confusius/registration/volume.py`
- `src/confusius/registration/resampling.py`
- `src/confusius/registration/bspline.py`
- `src/confusius/registration/volumewise.py`
- `src/confusius/registration/motion.py`
- `src/confusius/validation/iq.py`

For each caller:

- confirm whether the API is fUSI-native or generic;
- if fUSI-native, move it to the new `z/y/x` invariant;
- if generic, avoid over-tightening it.

## 4. Registration stack migration

This is the main code area affected by #231.

Files to review:

- `src/confusius/registration/volume.py`
- `src/confusius/registration/resampling.py`
- `src/confusius/registration/bspline.py`
- `src/confusius/registration/volumewise.py`
- `src/confusius/registration/motion.py`
- `src/confusius/registration/_utils.py`
- `tests/unit/test_registration/`
- relevant napari registration tests

Tasks:

- stop accepting true `(y, x)` fUSI inputs at the public API level;
- migrate single-slice registration inputs to `(1, y, x)` or `(time, 1, y, x)`;
- review whether internal SimpleITK 2D branches can stay as implementation details or should be removed;
- update docstrings and error messages that currently say “2D or 3D”.

New preferred wording:

- “3D volumes, including single-slice volumes stored with `z=1`”;
- “time-varying data should be `(time, z, y, x)`”.

## 5. Review smoothing semantics

File:

- `src/confusius/spatial/smooth.py`

Open design question:

- should `smooth_volume` remain a generic smoothing helper, or should it become strictly fUSI-volume oriented?

If it is a fUSI-volume API, it should require `z`, `y`, `x`.
If it is intentionally generic, then real 2D input may stay supported.

This decision should be made explicitly before changing behavior.

## 6. Review extraction / connectivity boundaries

Files:

- `src/confusius/extract/labels.py`
- `src/confusius/extract/mask.py`
- `src/confusius/connectivity/seed.py`
- tests in `tests/unit/test_extract/` and `tests/unit/test_connectivity/`

Planned approach:

- keep low-level extraction helpers generic if they are truly dimension-agnostic;
- update higher-level fUSI workflow APIs to prefer / require canonical `(z, y, x)`-based inputs where appropriate;
- switch examples and fUSI-oriented tests from `(y, x)` to singleton-`z`.

## 7. Review xarray accessor examples and docs

Files:

- `src/confusius/xarray/accessors.py`
- `docs/user-guide/xarray.md`
- `tests/unit/test_xarray/test_accessors.py`

Plan:

- keep generic accessors like spacing/origin working on arbitrary DataArrays;
- stop presenting `(y, x)` as the canonical representation of a fUSI recording;
- update examples to use `(z, y, x)` or `(time, z, y, x)` when describing fUSI data.

## 8. IO and user-facing examples

Files / areas to review:

- `docs/user-guide/io.md`
- `docs/user-guide/spatial-conventions.md`
- `docs/user-guide/visualization.md`
- `docs/user-guide/multipose.md`
- `docs/examples/01_io/01_confusius_xarray_101.py`
- `docs/examples/02_registration/02_volumewise_motion_correction.py`
- any other example or guide that describes “2D+t” fUSI data

Plan:

- update wording so that single-slice acquisitions are described as singleton-`z` 3D+t data;
- align the constructor helper examples with existing IO behavior;
- remove any guidance that suggests users should drop or squeeze the `z` axis for single-slice fUSI data.

## 9. Test migration strategy

We should treat this as a fixture migration, not just scattered edits.

### 9.1 Convert fUSI-semantic fixtures

These should be bumped from 2D to singleton-`z` when they represent:

- fUSI recordings;
- fUSI volumes;
- registration inputs/outputs;
- spatial maps that are supposed to retain fUSI geometry.

Typical replacements:

- `(y, x)` → `(1, y, x)`;
- `(time, y, x)` → `(time, 1, y, x)`.

### 9.2 Keep true 2D tests where appropriate

Do **not** blanket-convert every 2D test in the repo.

Keep real 2D tests for APIs that are intentionally generic, such as:

- matrix plotting;
- generic statistical maps;
- generic validators;
- already-sliced image plotting helpers;
- non-fUSI array utilities.

### 9.3 Add explicit regression tests

Add tests that ensure:

- `(y, x)` is rejected by `validate_fusi_dataarray` when validating fUSI data;
- `(time, y, x)` is likewise rejected;
- `(z, y, x)` with `z=1` validates successfully;
- `(time, z, y, x)` with `z=1` validates successfully;
- `create_fusi_dataarray(...)` returns valid canonical results.

## 10. Audit checklist by area

### Core validation

- `src/confusius/validation/fusi.py`
- `tests/unit/test_validation/test_fusi.py`

### Registration

- `src/confusius/registration/*.py`
- `tests/unit/test_registration/*`
- `tests/unit/test_napari/test_registration_*`

### Constructor helper

- new public module / export location
- helper tests
- helper docs

### Smoothing

- `src/confusius/spatial/smooth.py`
- associated tests

### Extraction / connectivity

- `src/confusius/extract/*.py`
- `src/confusius/connectivity/*.py`
- associated tests

### Xarray / docs / examples

- accessor examples
- user guide
- examples

### Generic utilities sanity pass

- confirm that truly generic 2D APIs still accept 2D arrays;
- confirm that only fUSI-native APIs are tightened.

## 11. Likely test files to migrate heavily

Based on the initial audit, the biggest fUSI-semantic 2D test usage is likely in:

- `tests/unit/test_validation/test_fusi.py`
- `tests/unit/test_registration/conftest.py`
- `tests/unit/test_registration/test_volume.py`
- `tests/unit/test_registration/test_volumewise.py`
- `tests/unit/test_registration/test_progress.py`
- `tests/unit/test_napari/test_registration_panel.py`
- `tests/unit/test_napari/test_registration_progress.py`
- `tests/unit/test_plotting/test_napari.py`
- any fixture currently named like `sample_2dt_*` that is really single-slice fUSI data

These should be reviewed first.

## 12. Likely areas to leave 2D alone

These are likely generic and should not be forced into singleton-`z` unless a closer review shows otherwise:

- `src/confusius/plotting/matrix.py`
- `tests/unit/test_stats/test_thresholding.py`
- generic GLM map tests using derived spatial maps;
- generic plotting tests operating on already-sliced `(y, x)` views;
- unit/coordinate validators that compare arbitrary arrays.

## 13. Deliverable structure

A reasonable implementation order is:

1. add the constructor helper;
2. tighten `validate_fusi_dataarray`;
3. migrate registration and its tests;
4. migrate shared fUSI fixtures;
5. update docs/examples;
6. do a final pass to make sure generic 2D APIs still behave as intended.

## 14. Main risk

The main risk is over-applying #231 and breaking APIs that are meant to work on generic 2D arrays.

Guiding rule:

- if the object is described as a **fUSI volume or recording**, require `z`, `y`, `x`;
- if the object is described as a **generic image / slice / map / matrix**, keep 2D support.

## 15. Expected outcome

After this work:

- ConfUSIus will have a single canonical representation for all fUSI recordings and volumes;
- single-slice acquisitions will preserve `z` spacing and geometry instead of collapsing to ambiguous 2D arrays;
- users will have a straightforward helper to construct valid fUSI DataArrays from scratch;
- registration and other geometry-sensitive tools will operate on a less ambiguous model.
