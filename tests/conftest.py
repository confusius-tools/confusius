"""Global fixtures shared across all tests."""

import os
import sys

import numpy as np
import pytest

# Use the offscreen Qt backend when no display is available (e.g. CI).
# Must be set before any Qt import so that QApplication does not try to
# connect to an X server and abort.
# There is a mismatch between QT offscreen support and vispy OpenGL support on Windows
# https://github.com/napari/napari/issues/5355
# https://github.com/napari/napari/issues/804
# On Windows CI, we setup the headless display on the GitHub Actions workflow
if sys.platform != "win32":
    # Linux/macOS CI usually has no display; use the offscreen Qt backend.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
