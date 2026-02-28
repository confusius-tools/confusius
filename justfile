# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Build documentation.
docs:
    uv run zensical build

# Serve documentation locally for development.
serve-docs:
    uv run zensical serve

# Clean documentation build artifacts.
clean-docs:
    rm -rf .cache/
    rm -rf site/

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
alias sd := serve-docs
alias t := test
alias tv := test-verbose
alias pc := pre-commit
