"""Tests for confusius._utils.plotting."""

import numpy as np
import pytest

from confusius._utils.plotting import scale_min_max


def test_scale_min_max_ignores_inf_but_preserves_nan():
    arr = np.array([-np.inf, 0.0, 5.0, 10.0, np.nan])
    result = scale_min_max(arr)

    # -inf/inf are excluded from the bounds and clipped into [0, 1]; nan is
    # excluded from the bounds too but has no position on the scale to clip to.
    assert result[0] == 0.0
    np.testing.assert_allclose(result[1:4], [0.0, 0.5, 1.0])
    assert np.isnan(result[4])


def test_scale_min_max_raises_on_no_finite_values():
    arr = np.array([-np.inf, np.inf, np.nan])
    with pytest.raises(ValueError, match="no finite values"):
        scale_min_max(arr)
