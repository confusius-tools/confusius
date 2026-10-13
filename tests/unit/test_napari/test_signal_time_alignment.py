"""Coordinate-based alignment must not infer correspondence from sample counts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from confusius._utils.signals import align_signal_to_time


def test_equal_length_shifted_times_interpolate_and_do_not_extrapolate():
    """Equal lengths still require real-time alignment."""
    result = align_signal_to_time([0, 2, 4], [0, 4, 16], [1, 3, 5])
    assert_allclose(result, [2, 10, np.nan])


@pytest.mark.parametrize("times", [[0, 0, 1], [0, 2, 1], [0, np.nan, 2], []])
def test_invalid_source_time_coordinates_raise(times):
    """Reject ambiguous, non-monotonic, missing and non-finite coordinates."""
    with pytest.raises(ValueError, match="Time coordinates"):
        align_signal_to_time(times, np.zeros(len(times)), [0, 1])


def test_mismatched_values_raise():
    """Coordinate/value length mismatches cannot be aligned positionally."""
    with pytest.raises(ValueError, match="matching lengths"):
        align_signal_to_time([0, 1], [10], [0, 1])
