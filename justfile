set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Build the examples gallery from docs/examples/*.py.
gallery:
    uv run python tools/build_gallery.py

# Remove generated gallery artifacts and the gallery cache.
[unix]
clean-gallery:
    rm -rf docs/examples/_built docs/examples/index.md .gallery-cache

# Remove generated gallery artifacts and the gallery cache.
[windows]
clean-gallery:
    foreach ($p in 'docs/examples/_built', 'docs/examples/index.md', '.gallery-cache') { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

# Build documentation.
docs: gallery
    uv run zensical build --strict

# Serve documentation locally for development.
serve-docs: gallery
    uv run zensical serve

# Clean documentation build artifacts.
[unix]
clean-docs: clean-gallery
    rm -rf .cache/ site/

# Clean documentation build artifacts.
[windows]
clean-docs: clean-gallery
    foreach ($p in '.cache', 'site') { if (Test-Path $p) { Remove-Item -Recurse -Force $p } }

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
[unix]
generate-baselines:
    rm -f tests/unit/test_plotting/baseline/*.png
    uv run pytest --mpl-generate-path=tests/unit/test_plotting/baseline \
        tests/unit/test_plotting/test_image.py::TestPlotVolumeVisualRegression \
        tests/unit/test_plotting/test_image.py::TestPlotContoursVisualRegression \
        tests/unit/test_plotting/test_image.py::TestPlotCarpetVisualRegression \
        tests/unit/test_plotting/test_image_composite.py::TestAddCompositeVisualRegression \
        tests/unit/test_plotting/test_image_stat_map.py::TestPlotStatMapVisualRegression \
        tests/unit/test_plotting/test_matrix.py::TestPlotMatrixVisualRegression

# Generate baseline images for visual regression tests.
[windows]
generate-baselines:
    if (Test-Path tests/unit/test_plotting/baseline/*.png) { Remove-Item -Force tests/unit/test_plotting/baseline/*.png }
    uv run pytest --mpl-generate-path=tests/unit/test_plotting/baseline `
        tests/unit/test_plotting/test_image.py::TestPlotVolumeVisualRegression `
        tests/unit/test_plotting/test_image.py::TestPlotContoursVisualRegression `
        tests/unit/test_plotting/test_image.py::TestPlotCarpetVisualRegression `
        tests/unit/test_plotting/test_image_composite.py::TestAddCompositeVisualRegression `
        tests/unit/test_plotting/test_image_stat_map.py::TestPlotStatMapVisualRegression `
        tests/unit/test_plotting/test_matrix.py::TestPlotMatrixVisualRegression

# Run all pre-commit hooks.
pre-commit:
    uv run prek run --all-files

# Aliases
alias d := docs
alias cd := clean-docs
alias g := gallery
alias cg := clean-gallery
alias gdi := generate-doc-images
alias sd := serve-docs
alias t := test
alias tv := test-verbose
alias pc := pre-commit
