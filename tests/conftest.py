"""Global fixtures shared across all tests."""

import os
import sys

import numpy as np
import pytest

# Use the offscreen Qt backend when no display is available (e.g. CI).
# Must be set before any Qt import so that QApplication does not try to
# connect to an X server and abort.
# QT offscreen + vispy OpenGL crashes on Windows and macOS:
# https://github.com/napari/napari/issues/5355
# https://github.com/napari/napari/issues/804
# Windows: real headless display is set up via GitHub Actions (Mesa3D).
# macOS: the runner already has a window server, so no offscreen needed.
if sys.platform not in ("win32", "darwin"):
    # Linux CI usually has no display; use the offscreen Qt backend.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
