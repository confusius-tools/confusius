# ConfUSIus Agent Guidelines

This file guides AI coding agents working in this repository.

## What is ConfUSIus

ConfUSIus is a Python library for functional ultrasound imaging (fUSI) analysis:
loading, registering, extracting signals from, and visualizing spatially referenced
voxel data (fUSI recordings, brain atlas volumes, decomposition maps, displacement
fields, ...), built on `xarray`.

This is a **beta package** under rapid iteration. Backward compatibility is not a
concern — feel free to make breaking API changes when they improve the design.

## Data Model: VoxelData

**VoxelData** is ConfUSIus's canonical `xarray.DataArray` model for spatially referenced
voxel data. This includes fUSI recordings, brain atlas volumes, PCA/decomposition
component maps, dense displacement fields, B-spline control-point grids, and anything
else gridded in space. A VoxelData array is **always**:

- Dims `(...extra, time, pose, k, j, i)` — `time`/`pose` are optional, extra
  non-spatial dims (PCA/ICA components, displacement `component`, atlas region
  masks, ...) may precede them, but native voxel indices `k`/`j`/`i` are always
  present and always last.
- Backed by a `VoxelToWorldIndex` attached to `z`/`y`/`x`
  (`confusius._utils.geometry.VoxelToWorldIndex`/`VoxelToWorldTransform`), which
  lazily derives those world coordinates from `k`/`j`/`i` via the voxel-to-world affine:
  either one affine shared by all data or one affine per `pose`. World coordinates
  are never stored directly—always this index's output.

Every 3D spatial array in the codebase must be VoxelData. Use these terms
consistently in code, docstrings, and docs:

- **VoxelData array**: preferred term in API docs, docstrings, and prose for an
  `xarray.DataArray` that satisfies the required VoxelData structure.
- **DataArray following the VoxelData model**: optional explanatory intro text when
  defining or teaching the concept.
- **VoxelData geometry**: specifically the affine/index/spatial coordinate semantics,
  not the whole array.

Use `confusius.validation.validate_voxeldata`/`ensure_voxeldata` to check any
VoxelData array—by default it enforces the universal `k`/`j`/`i` +
`VoxelToWorldIndex` structure above (dims, the index itself, `units` coordinate attrs).
Its optional flags (`require_time`, `require_unchunked_time`, `require_dtype`,
`require_velocity_attrs`, ...) layer on stricter requirements and should only be
enabled when needed.
Use `confusius.xarray.create_voxeldata` to build VoxelData regardless of what the
array's content represents, including IQ data. IQ-specific metadata such as
`transmit_frequency` and `beamforming_sound_velocity` are plain DataArray attrs,
validated with `require_velocity_attrs=True` when needed.
`confusius._utils.geometry.attach_voxel_to_world_index`/`has_voxel_to_world_index` are
private internals used by `create_voxeldata` itself—reach for them directly only in rare
cases (e.g. genuine I/O boundary code).

**There is no second supported data shape.** Do not add "plain z/y/x dims or k/j/i with
an index" branches, or code that silently degrades to plain-coordinate handling when
`has_voxel_to_world_index(data)` is `False`. A function that receives a non-VoxelData
DataArray should raise, not fall back to an alternate path.

Two exceptions:

1. **I/O boundary code** (e.g. `io/nifti.py`, `io/scan.py`) whose job is
   constructing the voxel-to-world index from an external file format—this code
   inherently runs before the index exists. Once construction is done, the result
   must be VoxelData; nothing downstream may special-case "no index" as supported.
2. **Dual-input consumers** documented to accept VoxelData *or* an already-reduced
   signals table `(time, region)` (e.g. `compute_compcor_confounds`,
   `apply_statistical_threshold`, `plot_carpet`). A signals table isn't a degraded
   form of VoxelData—it's already reduced off the voxel grid. Such consumers branch
   explicitly on `has_voxel_to_world_index`; don't reimplement this branch per call
   site—use `confusius._utils.mask.validate_spatial_or_feature_mask` and
   `confusius._utils.mask.select_masked_features`. Both are internal, existing only
   to serve this small set of dual-input consumers. This does not loosen the
   VoxelData requirement for functions whose input is meant to be spatial
   (`extract_with_mask`, `unmask`, `ensure_mask`, `ensure_labels`, ...).
   `_BaseFUSIDecomposer` (PCA/FastICA/NMF) is VoxelData-only, not a dual-input
   consumer: decomposing an already-reduced signals table is regular tabular
   PCA/ICA/NMF with no spatial structure to track, so use scikit-learn directly for
   that instead.

## Commands

Uses [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
uv sync                  # install dependencies
just test                # run tests with coverage (pytest --mpl)
just test-verbose        # ... with verbose output
just generate-baselines  # regenerate pytest-mpl visual regression baselines
just pre-commit          # ruff check/format, ty, codespell, numpydoc-validation
just docs                # build docs (Zensical)
just serve-docs          # build + serve docs locally with live reload
just generate-doc-images # regenerate docs/images/*/generate.py outputs
```

Run a single test: `uv run pytest path/to/test_file.py::TestClass::test_method`.

Docs are built on CI and deployed to a separate `confusius-docs` repo; see
`.github/workflows/docs.yml` and `tools/prefetch_doc_datasets.py` for the pipeline
(gallery build, image generation, dataset prefetch/cache) and `docs/contributing.md`
for the human contributor workflow.

Use the `/release NEW_VERSION` skill (`.claude/skills/release/SKILL.md`) for the full
release process (version bumps, changelog, tag, push, announcements).

## Code Architecture

```
src/confusius/
├── atlas/          # Atlas class, BrainGlobe integration, region/hemisphere masks
├── bids/           # BIDS coordinate/event/physio mapping and validation
├── connectivity/   # Seed-based and matrix functional connectivity
├── datasets/       # Downloaders for public fUSI/atlas datasets
├── decoding/       # Searchlight decoding
├── decomposition/  # PCA/ICA/NMF on VoxelData (_base.py)
├── extract/        # mask.py, labels.py, reconstruction.py — signal extraction
├── glm/            # First-/second-level GLM (design, contrasts, HRF models)
├── io/             # NIfTI, AUTC, EchoFrame, SCAN readers/writers (I/O boundary)
├── iq/             # IQ data clutter filtering and power reduction
├── multipose/      # Multi-pose consolidation and slice-timing correction
├── _napari/        # napari plugin (viewer widgets, QC, registration, video)
├── plotting/       # image.py (VolumePlotter, plot_contours/volume/carpet), matrix.py
├── qc/             # DVARS, tSNR quality-control metrics
├── registration/   # Affine/B-spline volume registration, resampling, motion
├── signal/         # Confound regression, censoring, detrending, filtering
├── spatial/        # Spatial smoothing
├── stats/          # Statistical thresholding
├── validation/     # validate_voxeldata/ensure_voxeldata, mask/labels/atlas/time-series checks
├── xarray/         # FUSIAccessor (`data.fusi.<accessor>.<method>()`), create/scale
└── _utils/         # cross-module internal helpers (see below)
```

### Module organization

- **Cross-module shared utilities** live in `confusius/_utils/<topic>.py` with public
  function names (e.g. `confusius/_utils/coordinates.py` exports
  `get_coordinate_spacings`, not `_get_coordinate_spacings`) — the leading `_` on the
  package conveys "internal API", names inside it are not prefixed. Group by topic
  (`geometry.py`, `mask.py`, `coordinates.py`, `timing.py`, `io.py`, `atlas.py`,
  `plotting.py`, ...); don't pile everything into one file.
- **Module-private shared helpers** live in `<module>/_utils.py` (e.g.
  `registration/_utils.py`, `glm/_utils.py`) — for a helper shared by 2+ files inside
  one module but not used outside it.
- **Never import a `_name` across module boundaries.** Need a private function from
  another module? Either inline it, make it public in the same file, or promote it to
  `_utils/`. The underscore is a real boundary, not decoration.

## Coding Conventions

- **Imports**: absolute only (`from confusius.io import AUTCDAT`); group
  stdlib/third-party/local; use `TYPE_CHECKING` for type-only imports.
- **Naming**: `snake_case` functions/variables (prefer imperative verb phrases, e.g.
  `get_source_dataarray`, not noun phrases), `PascalCase` classes, `UPPER_CASE`
  constants, leading `_` for private.
- **Types**: comprehensive hints incl. `numpy.typing.NDArray`; `Literal` for string
  literals; `TypedDict` for structured dicts; `TypeAlias` for complex types; `py.typed`
  is enabled.
- **Errors**: specific exceptions (`ValueError`, `TypeError`, `FileNotFoundError`);
  `warnings.warn()` for non-critical issues; validate inputs early with descriptive
  messages.
- **Docstrings**: NumPy format for *all* functions/methods, public and private
  (private helpers still need full Parameters/Returns/Raises).
  - Optional params: `arg : type, optional` (default `None`) or
    `arg : type, default: value` — never `arg : type or None, default: None`.
  - Fallback behavior: "If not provided, ..." — never "If `None`, ...".
  - Booleans: start the description with "Whether to ..." — never "If `True`/`False`, ...".
  - Multiple return values: one `name : type` block per value in `Returns` — never
    `tuple[type1, type2]` as the documented return type.
  - Inline code: single backticks. Full package names in the `type` line
    (`xarray.DataArray`); no `xarray.` prefix/backticks in prose (`DataArray`).
  - Array shapes: `(X, Y, Z) numpy.ndarray` / `(X, Y, Z) xarray.DataArray`.
  - Cross-references: `[name][confusius.module.path.name]` — never Sphinx `.. [1]`.
  - Module-level constants get a triple-quoted docstring immediately after them.
- **Comments**:
  1. Comments should not duplicate code.
  2. Good comments do not excuse unclear code.
  3. If you can't write a clear comment, there may be a problem with the code.
  4. Comments should dispel confusion, not cause it.
  5. Explain unidiomatic code in comments.
  6. Provide links to the original source of copied code.
  7. Include links to external references where they will be most helpful.
  8. Add comments when fixing bugs.
  9. Use `TODO:` prefix for incomplete implementations.
  10. All comments end with a period.
  - Code adapted from another project (e.g. nilearn) needs a module-level `NOTICE`
    file reference: "Portions of this file are derived from [Project], which is
    licensed under the [License]. See `NOTICE` file for details."

## Test Conventions

- **No useless tests**: must fail if the function returns garbage — avoid tests that
  only check shape preservation or "output differs from input".
- **Test public API only**: don't test `_private` functions directly; they're covered
  via the public functions that use them.
- **No `# pragma: no cover`**, and don't force tests for branches that are unreachable
  due to an upstream library invariant.
- Cover: edge cases (empty inputs, boundaries), error validation (`pytest.raises`),
  and comparisons against reference implementations (scipy, naive implementations)
  where one exists. Reach for property-based tests only when no reference
  implementation exists (mathematical properties: idempotence, commutativity, ...).
- Use fixtures from `conftest.py` before creating new test data;
  `numpy.testing.assert_allclose`/`assert_array_equal`; seeded RNGs; small arrays.
- Visual regression: `@pytest.mark.mpl_image_compare`, run with `pytest --mpl`,
  regenerate with `just generate-baselines`.
- **napari plugin tests**: prefer small unit tests over full GUI/integration tests —
  trust napari to deliver callbacks/events, test our logic and observable
  widget/viewer state directly. Use napari's `make_napari_viewer`/
  `make_napari_viewer_proxy` fixtures rather than hand-rolled viewer setup/teardown.

## Git & Changelog

Commit messages follow [Commitizen](https://commitizen.github.io/cz-cli/):

```
<type>(<scope>): <short summary>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.

Scopes: `io`/`nifti`/`autc`/`zarr` (I/O), `signal`/`spatial`/`iq`/`reduce`/`clutter`
(signal/IQ processing), `extract`/`validation`, `atlas`/`registration`,
`connectivity`/`multipose`, `qc`, `xarray`/`io-accessor`/`plotting`/`napari` (UI),
`docs`/`mkdocs`/`api`, `tests`.

`docs/changelog.md` is grouped by version, newest first. Add entries under the
current development version's `X.Y.Z.devN` heading (never a released version), under
section labels in this order (only add the ones you need):
`:boom: Breaking changes`, `:sparkles: Enhancements`, `:zap: Performance`,
`:bug: Fixes`, `:books: Documentation`, `:wrench: Maintenance`. napari-plugin entries
may be prefixed **[Napari plugin]**. Write from the user's perspective (effect, not
implementation), ending with a PR link:
`([#123](https://github.com/confusius-tools/confusius/pull/123))`.
