# ConfUSIus Agent Guidelines

## Project Status

This is a **pre-alpha package** under rapid iteration. Backward compatibility is not a concern - feel free to make breaking API changes when they improve the design.

## Release Process

Use the `/release NEW_VERSION` skill (`.claude/skills/release/SKILL.md`) to perform a
full release. It handles version bumps across all files, lock file sync, pre-commit
checks, commit + tag creation (with annotated tag message), a gated push step, and
generation of the GitHub release message and Discord announcement.

## Build/Lint/Test Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [just](https://github.com/casey/just) as a command runner.

### Build & Environment
- `uv sync` - Install dependencies and sync the virtual environment
- `uv build` - Build the package

### Documentation
- `just docs` (or `just d`) - Build documentation using Zensical
- `just clean-docs` (or `just cd`) - Clean documentation build directory and generated API files

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
- Use NumPy docstring format for all public functions
- Include Parameters, Returns, Raises sections
- Document complex algorithms with references
- Use type hints in docstrings when helpful
- Include default values in the type parameter as `arg : type, default: value`, or `arg
  : type, optional` when the default is `None`
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

## Testing Guidelines

### Philosophy
- **No useless tests**: Tests must fail if the function returns garbage. Avoid tests that only
  check shape preservation or that output differs from input.
- **Concise test suite**: No redundant tests. Each test should verify something unique.
- **Test public API only**: Do not test private functions (prefixed with `_`). They are
  implementation details covered by testing the public functions that use them.

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
