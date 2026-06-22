# ConfUSIus Agent Guidelines

## Project Status

This is a **beta package** under rapid iteration. Backward compatibility is not a
concern, feel free to make breaking API changes when they improve the design.

## Release Process

Use the `/release NEW_VERSION` skill (`.claude/skills/release/SKILL.md`) to perform a
full release. It handles version bumps across all files, lock file sync, pre-commit
checks, commit + tag creation (with annotated tag message), a gated push step, and
generation of the GitHub release message and Discord announcement.

## Build/Lint/Test Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
[just](https://github.com/casey/just) as a command runner.

### Build & Environment
- `uv sync` - Install dependencies and sync the virtual environment
- `uv build` - Build the package

### Documentation
- `just docs` (or `just d`) - Build documentation using Zensical
- `just serve-docs` (or `just sd`) - Build and serve documentation locally with live reload
- `just clean-docs` (or `just cd`) - Clean documentation build directory and generated API files
- `just generate-doc-images` (or `just gdi`) - Run all documentation image generators locally

#### CI pipeline

Documentation is built entirely on GitHub Actions and deployed to
[confusius-tools/confusius-docs](https://github.com/confusius-tools/confusius-docs)
(a separate public repo served via GitHub Pages at `https://confusius.tools/`).
The workflow (`.github/workflows/docs.yml`) has three jobs:

- **`build`** (every push and PR): generates all documentation images, builds the
  example gallery, runs `zensical build`, and uploads `site/`, `docs/examples/_built/`,
  and `docs/images/` as artifacts.
- **`deploy-preview`** (PRs only): downloads the `site/` artifact and deploys it to
  `pr-preview/pr-N/` in the docs repo. Posts a sticky comment on the PR with the preview
  URL and maintains `pr-preview/versions.json` for the mike version switcher.
- **`deploy-docs`** (pushes to `main` and version tags): downloads the pre-built
  artifacts and runs `mike deploy`. `main` → `dev` + `latest` alias; `v*` tags →
  `<version>` + `stable` alias.

A companion workflow (`.github/workflows/docs-cleanup-preview.yml`) removes the preview
directory and its `versions.json` entry when a PR is closed.

Deployment requires a `DOCS_DEPLOY_TOKEN` secret: a fine-grained PAT with **Contents:
Read and write** on `confusius-tools/confusius-docs`.

#### Adding a new documentation image generator

Image generators live in `docs/images/<topic>/generate.py`. Each one is gitignored
(only `generate.py` is committed; the outputs are not). To add a new generator:

1. Create `docs/images/<topic>/generate.py` and a matching `.gitignore` (copy an
   existing one as a template).
2. Add `uv run docs/images/<topic>/generate.py` to the `just generate-doc-images`
   recipe in `justfile`.
3. Add the script to the **Generate documentation images** step in
   `.github/workflows/docs.yml`:
   ```yaml
   xvfb-run -a uv run docs/images/<topic>/generate.py
   ```
4. **Cache**: add (or update) the matching fetch call in
   `tools/prefetch_doc_datasets.py` so the new script's data is pre-warmed in CI
   and the dataset cache key — which hashes only `tools/prefetch_doc_datasets.py`
   — invalidates when the args change. If the script pulls a brand-new dataset,
   also add a dedicated `actions/cache` step in `.github/workflows/docs.yml`
   keyed off the same prefetch file (see the existing cache steps as a template).

#### Adding new example scripts

Examples live in `docs/examples/` as Jupytext notebooks (`.py` files with a
`# %% [markdown]` header). The gallery builder (`tools/build_gallery.py`) discovers
and executes them automatically. To add a new example:

1. Place the script under the appropriate subdirectory in `docs/examples/`.
2. Add the built output path (`docs/examples/_built/<subdir>/<name>.md`) to the `nav`
   in `zensical.toml`.
3. **Cache**: the gallery cache key in `.github/workflows/docs.yml` already
   uses `docs/examples/**/*.py`, so it invalidates automatically when any
   example script changes. If the example loads data, add the matching fetch
   call to `tools/prefetch_doc_datasets.py` so it is pre-warmed in CI — the
   gallery renderer enforces light/dark output parity and a cold cache there
   will fail the build.

### Linting, Formatting & Type Checking
- `just pre-commit` (or `just pc`) - Run all pre-commit hooks (recommended)
- `uv run ruff check . --fix` - Run Ruff linter with auto-fix
- `uv run ruff format .` - Format code with Ruff
- `uv run ty check src/` - Run ty type checking
- `uv run codespell` - Run spell checker

Pre-commit hooks include:
- **ruff-check**: Linting with auto-fix
- **ruff-format**: Code formatting
- **ty**: Type checking (src/ directory only)
- **codespell**: Spell checking
- **numpydoc-validation**: Docstring validation

### Testing
- `just test` (or `just t`) - Run all tests with coverage
- `just test-verbose` (or `just tv`) - Run all tests with verbose output
- `just generate-baselines` - Regenerate visual regression test baselines (pytest-mpl)
- `uv run pytest path/to/test_file.py` - Run a single test file
- `uv run pytest path/to/test_file.py::TestClass::test_method` - Run a single test

Coverage reports are generated automatically (terminal, HTML in `htmlcov/`, and XML).

## Code Style Guidelines

### Imports
- Use absolute imports: `from confusius.io import AUTCDAT`
- Group imports: standard library, third-party, local modules
- Use type-only imports when possible: `from typing import TYPE_CHECKING`

### Formatting
- Use Ruff for auto-formatting (Black compatible)
- Line length: follow Ruff defaults
- Use double quotes for strings unless single quotes are needed for escaping

### Comments
1. Comments should not duplicate code.
2. Good comments do not excuse unclear code.
3. If you can't write a clear comment, there may be a problem with the code.
4. Comments should dispel confusion, not cause it.
5. Explain unidiomatic code in comments.
6. Provide links to the original source of copied code.
7. Include links to external references where they will be most helpful.
8. Add comments when fixing bugs.
9. Use `TODO:` prefix for comments to mark incomplete implementations.
10. All comments should end with a period.

### Types
- Use comprehensive type hints with `numpy.typing` for arrays
- Use `Literal` for string literal types
- Use `TypedDict` for structured data dictionaries
- Use `TypeAlias` for complex type definitions
- Use `npt.NDArray` for NumPy arrays with specific dtypes
- Enable `py.typed` marker for type checking

### Naming Conventions
- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private functions/methods: leading underscore `_function_name`
- Variables: descriptive `snake_case` names

### Error Handling
- Use specific exceptions: `ValueError`, `TypeError`, `FileNotFoundError`
- Use `warnings.warn()` for non-critical issues
- Validate inputs early with descriptive error messages
- Use try/except blocks for external operations

### Documentation
- Use Zensical for documentation generation
- Use NumPy docstring format for all functions and methods, including private helpers
  (prefixed with `_`) — they require full Parameters, Returns, and Raises sections just
  like public ones
- Include Parameters, Returns, Raises sections
- Document complex algorithms with references
- Use type hints in docstrings when helpful
- Include default values in the type parameter as `arg : type, default: value`, or `arg
  : type, optional` when the default is `None` — never write `arg : type or None,
  default: None`
- When describing the fallback behaviour of an optional parameter, write "If not
  provided, ..." — not "If `None`, ..."
- For boolean parameters, start the description with "Whether to ..." — not "If
  `True`/`False`, ..."
- Use single backticks for inline code (Zensical/MarkDocs style, not Sphinx rst)
- Use full package names in docstrings (e.g., `xarray.DataArray` not `xr.DataArray`)
- Use `list[...]`, `tuple[...]` syntax instead of "list of..." descriptions
- Document array shapes as `(X, Y, Z) numpy.ndarray` or `(X, Y, Z) xarray.DataArray`

#### Multiple Return Values
When a function returns multiple values, document each return value on a separate line in the `Returns` section:
```python
Returns
-------
first_value : type
    Description of the first return value.
second_value : type
    Description of the second return value.
```
Do not use `tuple[type1, type2]` as the return type in the docstring.

#### Constants
- Document module-level constants with triple-quoted docstrings placed immediately after the constant.
- Include a description of the constant's purpose and contents.

#### Attribution and Cross-references
- For code adapted from other projects (e.g., nilearn), add a NOTICE file reference at the
  module level: "Portions of this file are derived from [Project], which is licensed under
  the [License]. See `NOTICE` file for details."
- Use mkdocs-style links for cross-references to other functions/classes:
  `[function_name][confusius.module.path.function_name]` or
  `[ClassName][confusius.module.path.ClassName]`
- Example: `[fit][confusius.glm._models.OLSModel.fit]` or
  `[SeedBasedMaps][confusius.connectivity.SeedBasedMaps]`
- Do NOT use Sphinx-style reference sections (`.. [1]`)

### Code Structure
- Use pathlib.Path for file operations
- Use context managers for file handling
- Prefer functional programming where appropriate
- Use list/dict comprehensions for simple transformations
- Keep functions focused on single responsibilities

### Module Organization
- **Cross-module shared utilities live in `confusius/_utils/<topic>.py`** with public
  function names (e.g. `confusius/_utils/coordinates.py` exports
  `get_coordinate_spacings`, not `_get_coordinate_spacings`). The leading `_` on the
  package conveys "internal API"; names inside it are not prefixed. Group by topic
  (`stack.py`, `coordinates.py`, `timing.py`, `io.py`, `atlas.py`, `plotting.py`, etc.) —
  do not pile everything into a single file.
- **Module-private shared helpers live in `<module>/_utils.py`**
  (e.g. `registration/_utils.py`, `plotting/_utils.py`). Use this when a helper is shared
  by 2+ files inside one module but not used outside. Same naming convention: the file
  is private, the names within are public.
- **Never import a `_name` across module boundaries.** If you need a private function
  from another module, that is a signal to either (a) inline it, (b) make it public in
  the same file, or (c) promote it to `_utils/`. The underscore is a real boundary, not
  decoration — respect it.

### Performance
- Use NumPy operations for array computations
- Use Dask for large array processing
- Prefer vectorized operations over loops
- Use appropriate data types to minimize memory usage

## Commit Message Convention

This project follows the [Commitizen](https://commitizen.github.io/cz-cli/) convention for commit messages.

### Format
```
<type>(<scope>): <short summary>

<body>
```

### Types
- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code changes that neither fix a bug nor add a feature
- **perf**: Performance improvements
- **test**: Adding or correcting tests
- **chore**: Changes to build process or auxiliary tools

### Scopes
Use a scope that describes the affected component:
- `io`, `nifti`, `autc`, `zarr` - for I/O modules
- `signal`, `spatial`, `iq`, `reduce`, `clutter` - for signal/IQ processing
- `extract`, `validation` - for signal extraction and input validation
- `atlas`, `registration` - for atlas integration and volume registration
- `connectivity`, `multipose` - for connectivity and multi-pose analysis
- `qc` - for quality control
- `xarray`, `io-accessor`, `plotting`, `napari` - for UI and xarray extensions
- `docs`, `mkdocs`, `api` - for documentation
- `tests` - for test infrastructure

### Examples
```
feat(nifti): add support for NIfTI sidecar metadata

docs(mkdocs): update installation instructions

test(nifti): add fixtures for 2D/3D/4D NIfTI files

refactor(iq): simplify power reduction algorithm
```

## Changelog

The changelog lives at `docs/changelog.md`. Entries are grouped by version tag, newest
first. When adding an entry, place it under the **current development version** (the
`X.Y.Z.devN` heading at the top), never under an already-released version.

Within the development version, entries go under section labels (emoji headings),
ordered as below. Add only the labels you need:

- `### :boom: Breaking changes`
- `### :sparkles: Enhancements`
- `### :zap: Performance`
- `### :bug: Fixes`
- `### :books: Documentation`
- `### :wrench: Maintenance`

Napari plugin related changes may carry an entry with a prefix **[Napari plugin]**.
Write entries that are clear and concise, from the user's perspective: describe the
user-facing effect, not the implementation. End each entry with a link to its pull
request, e.g. `([#123](https://github.com/confusius-tools/confusius/pull/123))`.


## Testing Guidelines

### Philosophy
- **No useless tests**: Tests must fail if the function returns garbage. Avoid tests that only
  check shape preservation or that output differs from input.
- **Concise test suite**: No redundant tests. Each test should verify something unique.
- **Test public API only**: Do not test private functions (prefixed with `_`). They are
  implementation details covered by testing the public functions that use them.
- **No `# pragma: no cover`**: Do not add coverage pragmas to skip lines.
- **Do not force unreachable tests**: If a defensive branch is unreachable because an upstream
  library invariant prevents constructing that state (e.g., xarray coordinate shape consistency),
  do not add brittle tests just to satisfy coverage.

### What to Test
1. **Edge cases**: Empty inputs, boundary conditions, special values.
2. **Error validation**: Ensure expected exceptions are raised for invalid inputs.
3. **Reference implementations**: Compare against known-correct implementations (e.g., scipy
   for wrappers, naive implementations for optimized code).

### When to Use Property-Based Tests
- **Only** when no reference implementation exists.
- Examples: mathematical properties (idempotence, commutativity, invariants).
- Prefer reference implementation tests when available.

### Test Structure
- Use pytest fixtures for reusable test data. Always check for existing fixtures in
  `conftest.py` files before creating new test data.
- Use `numpy.testing.assert_allclose` for floating-point comparisons.
- Use `numpy.testing.assert_array_equal` for exact comparisons.
- Use `pytest.raises` for expected exceptions.
- Use `pytest.warns` for expected warnings.
- Keep tests fast by using small array sizes.
- Use seeded random number generators for reproducibility.

### Visual Regression Tests
- Use `@pytest.mark.mpl_image_compare` for plot output tests.
- Run `just generate-baselines` to regenerate baseline images after intentional plot changes.
- Run tests with `uv run pytest --mpl` to enable image comparison checks.
