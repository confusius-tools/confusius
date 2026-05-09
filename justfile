set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Build documentation.
docs:
    just gallery
    uv run zensical build --strict

# Serve documentation locally for development.
serve-docs:
    uv run zensical serve

# Build the examples gallery from docs/examples/*.py.
gallery:
    uv run python tools/build_gallery.py

# Remove generated gallery artifacts and the gallery cache.
gallery-clean:
    rm -rf docs/examples/_built docs/examples/index.md .cache/gallery

# Clean documentation build artifacts.
clean-docs:
    rm -rf .cache/
    rm -rf site/

# Generate documentation images.
generate-doc-images:
    uv run docs/images/home/generate.py
    uv run docs/images/gui/generate.py
    uv run docs/images/qc/generate.py
    uv run docs/images/visualization/generate.py

# Run all tests.
test:
    uv run pytest tests/ --mpl

# Run tests with verbose output.
test-verbose:
    uv run pytest tests/ -v --mpl

# Generate baseline images for visual regression tests.
generate-baselines:
    rm -f tests/unit/test_plotting/baseline/*.png
    uv run pytest --mpl-generate-path=tests/unit/test_plotting/baseline tests/unit/test_plotting/test_image.py::TestPlotVolumeVisualRegression tests/unit/test_plotting/test_image.py::TestPlotContoursVisualRegression

# Run all pre-commit hooks.
pre-commit:
    uv run prek run --all-files

# Aliases
alias d := docs
alias cd := clean-docs
alias g := gallery
alias gc := gallery-clean
alias gdi := generate-doc-images
alias sd := serve-docs
alias t := test
alias tv := test-verbose
alias pc := pre-commit
