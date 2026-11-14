"""Direct VoxelData extraction exports and separate acquisition validation."""

import numpy as np
import pytest
import voxeldata as vd
import xarray as xr

from confusius.validation import ensure_labels, ensure_mask


@pytest.mark.parametrize("operation", [ensure_mask, ensure_labels])
def test_mask_validators_retain_acquisition_validation(sample_voxeldata_3dt, operation):
    """Explicit ConfUSIus validation still rejects invalid acquisition metadata."""
    data = sample_voxeldata_3dt.copy(deep=True)
    data.time.attrs["volume_acquisition_reference"] = "invalid"
    spatial = data.isel(time=0, drop=True).copy(
        data=np.ones(data.shape[-3:], dtype=int)
    )
    with pytest.raises(ValueError, match="volume_acquisition_reference"):
        operation(spatial, data)


@pytest.mark.parametrize("kind", ["mask", "labels"])
@pytest.mark.parametrize("use_flox", [False, True])
def test_voxeldata_accessors_preserve_timestamps_without_defaults(
    sample_voxeldata_3dt, kind, use_flox
):
    """Extraction preserves timestamp metadata without acquisition interpretation."""
    data = sample_voxeldata_3dt.copy(deep=True)
    data.time.attrs = {"units": "s"}
    membership = np.zeros(data.shape[-3:], dtype=int)
    membership[0, 0, 0] = membership[1, 1, 1] = 1
    spatial = data.isel(time=0, drop=True).copy(data=membership)
    operation = (
        data.vd.extract_with_mask if kind == "mask" else data.vd.extract_with_labels
    )
    with xr.set_options(use_flox=use_flox):
        result = operation(spatial)
    if kind == "mask":
        np.testing.assert_array_equal(result, data.values[:, membership != 0])
    else:
        np.testing.assert_allclose(
            result[:, 0], data.values[:, membership != 0].mean(axis=-1)
        )
    assert result.time.attrs == {"units": "s"}
    assert data.time.attrs == {"units": "s"}


@pytest.mark.parametrize("name", ["extract_with_mask", "extract_with_labels", "unmask"])
def test_extraction_exports_are_voxeldata_functions(name):
    """The public extraction namespace contains direct exports, not wrappers."""
    from confusius import extract

    assert getattr(extract, name) is getattr(vd, name)


def test_voxeldata_unmask_preserves_mask_named_frames(sample_voxeldata_3d_oblique):
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
    signals = xr.DataArray([3.0, 8.0], dims="space")
    result = signals.vd.unmask(mask, fill_value=-1.0)
    expected = np.full(mask.shape, -1.0)
    expected.flat[[0, -1]] = [3.0, 8.0]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result.vd.affines["reference"], frame)
    np.testing.assert_array_equal(result.vd.voxel_to_world, mask.vd.voxel_to_world)
