"""Acquisition semantics retained around delegated VoxelData extraction."""

import numpy as np
import pytest
import xarray as xr

from confusius.extract import extract_with_labels, extract_with_mask, unmask
from confusius.validation import ensure_labels, ensure_mask


@pytest.mark.parametrize(
    "operation", [extract_with_mask, extract_with_labels, ensure_mask, ensure_labels]
)
def test_delegation_retains_acquisition_validation(sample_voxeldata_3dt, operation):
    """Generic delegation must not bypass invalid fUSI timing metadata."""
    data = sample_voxeldata_3dt.copy(deep=True)
    data.time.attrs["volume_acquisition_reference"] = "invalid"
    spatial = data.isel(time=0, drop=True).copy(
        data=np.ones(data.shape[-3:], dtype=int)
    )
    args = (
        (data, spatial)
        if operation in (extract_with_mask, extract_with_labels)
        else (spatial, data)
    )
    with pytest.raises(ValueError, match="volume_acquisition_reference"):
        operation(*args)


@pytest.mark.parametrize("kind", ["mask", "labels"])
@pytest.mark.parametrize("use_flox", [False, True])
def test_fusi_accessors_retain_timestamp_defaults(sample_voxeldata_3dt, kind, use_flox):
    """Both backends receive the same acquisition defaults through `.fusi`."""
    data = sample_voxeldata_3dt.copy(deep=True)
    data.time.attrs = {"units": "s"}
    membership = np.zeros(data.shape[-3:], dtype=int)
    membership[0, 0, 0] = membership[1, 1, 1] = 1
    spatial = data.isel(time=0, drop=True).copy(data=membership)
    operation = (
        data.fusi.extract.with_mask if kind == "mask" else data.fusi.extract.with_labels
    )
    with xr.set_options(use_flox=use_flox), pytest.warns(UserWarning):
        result = operation(spatial)
    if kind == "mask":
        np.testing.assert_array_equal(result, data.values[:, membership != 0])
    else:
        np.testing.assert_allclose(
            result[:, 0], data.values[:, membership != 0].mean(axis=-1)
        )
    assert result.time.attrs["units"] == "s"
    assert result.time.attrs["volume_acquisition_reference"] == "start"
    assert result.time.attrs["volume_acquisition_duration"] == pytest.approx(
        np.diff(data.time.values).mean()
    )
    assert data.time.attrs == {"units": "s"}


def test_unmask_preserves_mask_named_frames(sample_voxeldata_3d_oblique):
    """Delegated reconstruction retains the mask's index-owned world frames."""
    mask = sample_voxeldata_3d_oblique.copy(
        data=np.zeros(sample_voxeldata_3d_oblique.shape, dtype=bool)
    )
    mask.data.flat[[0, -1]] = True
    frame = np.array(
        [
            [0.0, -1.0, 0.0, 3.0],
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    mask = mask.vd.set_affine("reference", frame)
    result = unmask(np.array([3.0, 8.0]), mask)
    expected = np.zeros(mask.shape)
    expected.flat[[0, -1]] = [3.0, 8.0]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result.vd.get_affine("reference"), frame)
    np.testing.assert_array_equal(result.vd.voxel_to_world, mask.vd.voxel_to_world)
