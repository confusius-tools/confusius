"""Global fixtures shared across all tests."""

import os

import numpy as np
import pytest

# Use the offscreen Qt backend when no display is available (e.g. CI).
# Must be set before any Qt import so that QApplication does not try to
# connect to an X server and abort.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)
