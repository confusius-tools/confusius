"""Global fixtures shared across all tests."""

import os
import sys

import numpy as np
import pytest

# Configure the Qt/OpenGL backend for headless testing.
# Must be set before any Qt import so that QApplication does not try to
# connect to a display server and abort.
# Windows runners (desktop and CI) always have a display, so we keep the
# native "windows" platform plugin.
if sys.platform != "win32":
    # Linux/macOS CI usually has no display; use the offscreen Qt backend.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
