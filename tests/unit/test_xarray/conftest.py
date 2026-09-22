"""Fixtures shared by `confusius.xarray` constructor tests."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def identity_pose_affines() -> Callable[[int], npt.NDArray[np.float64]]:
    """Return a factory building an `(npose, 4, 4)` stack of identity affines."""

    def _make(npose: int) -> npt.NDArray[np.float64]:
        return np.stack([np.eye(4) for _ in range(npose)])

    return _make
