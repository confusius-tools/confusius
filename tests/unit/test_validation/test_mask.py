"""Tests for confusius.validation.ensure_mask/validate_mask/ensure_labels/validate_labels."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
)
from confusius.validation import ensure_labels, ensure_mask, validate_labels, validate_mask


@pytest.mark.parametrize("region_id", [1, 7, 256, 512, 1009])
def test_coerces_integer_label_to_boolean(sample_fusi_3dt, sample_voxeldata_mask, region_id):
    """A single-label integer mask {0, region_id} is returned as a boolean mask.

    Region ids that are multiples of 256 (256, 512) are included because casting the
    raw integer mask to `numpy.uint8` would wrap them to 0; the boolean coercion must
    not depend on the label value.
    """
    mask = sample_voxeldata_mask(np.int32)
    mask.values.flat[2:5] = region_id

    result = ensure_mask(mask, sample_fusi_3dt)

    assert result.dtype == bool
    assert_array_equal(result.values.ravel(), mask.values.ravel() != 0)


def test_passes_boolean_through(sample_fusi_3dt, sample_voxeldata_mask):
    """A boolean mask is returned as boolean with its values unchanged."""
    mask = sample_voxeldata_mask()
    mask.values.flat[1:4] = True

    result = ensure_mask(mask, sample_fusi_3dt)

    assert result.dtype == bool
    assert_array_equal(result.values.ravel(), mask.values.ravel())


def test_return_dtype_as_bool_false_preserves_dtype(sample_fusi_3dt, sample_voxeldata_mask):
    """With return_dtype_as_bool=False the mask is returned with its original dtype."""
    mask = sample_voxeldata_mask(np.int32)
    mask.values.flat[2:5] = 512

    result = ensure_mask(mask, sample_fusi_3dt, coerce_bool=False)

    assert result.dtype == np.int32
    assert_array_equal(result.values.ravel(), mask.values.ravel())


def test_coerced_mask_preserves_dims_and_coords(sample_fusi_3dt, sample_voxeldata_mask):
    """The coerced mask keeps the input dimensions and coordinates."""
    mask = sample_voxeldata_mask(np.int32)
    mask.values.flat[3:] = 512

    result = ensure_mask(mask, sample_fusi_3dt)

    assert result.dims == mask.dims
    assert_array_equal(result.coords["i"].values, mask.coords["i"].values)


def test_validate_mask_returns_none_for_canonical_aligned_input(
    sample_fusi_3dt, sample_voxeldata_mask
):
    """validate_mask is a pure check: it returns None, not a canonicalized mask."""
    mask = sample_voxeldata_mask()

    assert validate_mask(mask, sample_fusi_3dt) is None


def test_validate_mask_rejects_scalar_reduced_data(sample_fusi_3dt, sample_voxeldata_mask):
    """validate_mask does not canonicalize: a scalar-reduced data dim is rejected.

    Unlike ensure_mask, which restores data.isel(k=0)'s dropped k dim before
    checking, validate_mask requires both mask and data to already be canonical
    VoxelData arrays.
    """
    mask = sample_voxeldata_mask()
    reduced_data = sample_fusi_3dt.isel(k=0)

    with pytest.raises(ValueError, match="native voxel dimension"):
        validate_mask(mask, reduced_data)


def test_validate_mask_rejects_misaligned_grid(sample_fusi_3dt, sample_voxeldata_mask):
    """validate_mask rejects a mask whose voxel grid doesn't match data's."""
    mask = sample_voxeldata_mask()
    shifted_affine = get_voxel_to_world_affine(sample_fusi_3dt).copy()
    shifted_affine[:3, 3] += 1.0
    mask = attach_voxel_to_world_index(
        mask.drop_vars(("z", "y", "x")),
        shifted_affine,
        world_coord_attrs={
            name: dict(sample_fusi_3dt.coords[name].attrs)
            for name in get_voxel_to_world_coord_names(sample_fusi_3dt)
        },
    )

    with pytest.raises(ValueError, match="does not share data's voxel grid"):
        validate_mask(mask, sample_fusi_3dt)


def test_ensure_labels_canonicalizes_scalar_reduced_dim(sample_fusi_3dt, sample_roi_labels):
    """ensure_labels restores a scalar-reduced voxel dim before validating."""
    result = ensure_labels(sample_roi_labels.isel(k=0), sample_fusi_3dt.isel(k=0))

    assert result.dims == ("k", "j", "i")


def test_validate_labels_returns_none_for_canonical_aligned_input(
    sample_fusi_3dt, sample_roi_labels
):
    """validate_labels is a pure check: it returns None, not canonicalized labels."""
    assert validate_labels(sample_roi_labels, sample_fusi_3dt) is None


def test_validate_labels_rejects_scalar_reduced_data(sample_fusi_3dt, sample_roi_labels):
    """validate_labels does not canonicalize: a scalar-reduced data dim is rejected."""
    with pytest.raises(ValueError, match="native voxel dimension"):
        validate_labels(sample_roi_labels, sample_fusi_3dt.isel(k=0))
