"""Tests for confusius.validation.validate_mask."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from confusius.validation import validate_mask
from confusius.xarray import create_voxeldata


def _data_and_mask(values):
    """Build a (time, k, j, i) data array and a matching (k, j, i) mask.

    Parameters
    ----------
    values : array_like
        1D mask values, reshaped onto a `(1, 1, len(values))` voxel grid.

    Returns
    -------
    data : xarray.DataArray
        A `(time, k, j, i)` array sharing the mask's voxel grid.
    mask : xarray.DataArray
        A `(k, j, i)` mask wrapping `values`.
    """
    values = np.asarray(values).reshape(1, 1, -1)
    data = create_voxeldata(
        np.zeros((3, *values.shape)),
        dims=("time", "k", "j", "i"),
        dt=1.0,
        spacing=(1.0, 1.0, 1.0),
    )
    mask = create_voxeldata(values, dims=("k", "j", "i"), spacing=(1.0, 1.0, 1.0))
    return data, mask


@pytest.mark.parametrize("region_id", [1, 7, 256, 512, 1009])
def test_coerces_integer_label_to_boolean(region_id):
    """A single-label integer mask {0, region_id} is returned as a boolean mask.

    Region ids that are multiples of 256 (256, 512) are included because casting the
    raw integer mask to `numpy.uint8` would wrap them to 0; the boolean coercion must
    not depend on the label value.
    """
    raw = np.zeros(8, dtype=np.int32)
    raw[2:5] = region_id
    data, mask = _data_and_mask(raw)

    result = validate_mask(mask, data)

    assert result.dtype == bool
    assert_array_equal(result.values.ravel(), raw != 0)


def test_passes_boolean_through():
    """A boolean mask is returned as boolean with its values unchanged."""
    raw = np.zeros(8, dtype=bool)
    raw[1:4] = True
    data, mask = _data_and_mask(raw)

    result = validate_mask(mask, data)

    assert result.dtype == bool
    assert_array_equal(result.values.ravel(), raw)


def test_return_dtype_as_bool_false_preserves_dtype():
    """With return_dtype_as_bool=False the mask is returned with its original dtype."""
    raw = np.zeros(8, dtype=np.int32)
    raw[2:5] = 512
    data, mask = _data_and_mask(raw)

    result = validate_mask(mask, data, coerce_bool=False)

    assert result.dtype == np.int32
    assert_array_equal(result.values.ravel(), raw)


def test_coerced_mask_preserves_dims_and_coords():
    """The coerced mask keeps the input dimensions and coordinates."""
    raw = np.zeros(6, dtype=np.int32)
    raw[3:] = 512
    data, mask = _data_and_mask(raw)

    result = validate_mask(mask, data)

    assert result.dims == mask.dims
    assert_array_equal(result.coords["i"].values, mask.coords["i"].values)
