"""Tests for mask extraction and reconstruction."""

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
    has_voxel_to_world_index,
)
from confusius.extract import extract_with_mask, unmask


def test_extract_with_mask_selects_expected_voxels(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Extraction retains the masked voxel signals and spatial index."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    signals = extract_with_mask(sample_voxeldata_3dt, mask)

    assert signals.dims == ("time", "space")
    assert signals.indexes["space"].names == ["k", "j", "i"]
    assert list(signals.indexes["space"]) == [(0, 1, 2), (1, 2, 3)]
    np.testing.assert_array_equal(
        signals.values,
        sample_voxeldata_3dt.values[:, [0, 1], [1, 2], [2, 3]],
    )


def test_extract_with_mask_restores_scalar_voxel_dim(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """`data` with a scalar-reduced voxel dim (e.g. from `.isel(k=0)`) is canonicalized.

    `.isel(k=0)` collapses `k` to a scalar coordinate, dropping the dim itself, which
    only `ensure_voxeldata` restores -- exercises that `extract_with_mask`
    canonicalizes `data` itself rather than relying on `ensure_mask`'s internal
    (and discarded) canonicalization of it.
    """
    single_k = sample_voxeldata_3dt.isel(k=0)
    mask = make_sample_voxeldata_mask().isel(k=[0])
    mask.data[0, 1, 2] = True

    signals = extract_with_mask(single_k, mask)

    assert signals.dims == ("time", "space")
    np.testing.assert_array_equal(
        signals.values, sample_voxeldata_3dt.values[:, 0, [1], [2]]
    )


@pytest.mark.parametrize(
    ("dtype", "label_value"),
    [(np.int32, 7), (np.int32, 256), (np.float64, 5.0), (np.float32, 1.0)],
)
def test_extract_with_mask_accepts_binary_numeric_mask(
    sample_voxeldata_3dt: xr.DataArray,
    make_sample_voxeldata_mask,
    dtype: type[np.generic],
    label_value: float,
) -> None:
    """A binary numeric mask (0 and one non-zero value) selects like a boolean mask.

    Covers both a single-label integer mask (produced by
    `AtlasAccessor.get_masks`) and a float binary mask (produced by tools such as
    FSL/NiBabel that don't support a boolean NIfTI dtype), see #382.
    """
    boolean_mask = make_sample_voxeldata_mask()
    boolean_mask.data[0, 1, 2] = True
    numeric_mask = boolean_mask.astype(dtype) * label_value

    expected = extract_with_mask(sample_voxeldata_3dt, boolean_mask)
    result = extract_with_mask(sample_voxeldata_3dt, numeric_mask)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("dtype", "values", "message"),
    [
        (np.int32, (1, 2), "2 distinct non-zero"),
        (np.float64, (1.0, 2.0), "2 distinct non-zero"),
        (str, ("a",), "binary numeric dtype"),
    ],
)
def test_extract_with_mask_rejects_invalid_mask_values(
    sample_voxeldata_3dt: xr.DataArray,
    make_sample_voxeldata_mask,
    dtype: type[np.generic] | type[float] | type[str],
    values: tuple[float, ...],
    message: str,
) -> None:
    """Only boolean and binary numeric masks are accepted."""
    mask = make_sample_voxeldata_mask(dtype)
    mask.data.flat[: len(values)] = values

    with pytest.raises(TypeError, match=message):
        extract_with_mask(sample_voxeldata_3dt, mask)


def test_extract_with_mask_rejects_misaligned_coordinates(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Extraction rejects a mask whose affine differs from data's.

    The mask's voxel-space (k/j/i) coordinates match data's exactly; only the
    voxel_to_world affine (origin shifted) differs, so this specifically exercises
    that affine mismatches are caught, not just voxel-coordinate mismatches.
    """
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    shifted_affine = get_voxel_to_world_affine(sample_voxeldata_3dt).copy()
    shifted_affine[:3, 3] += 1.0
    mask = attach_voxel_to_world_index(
        mask.drop_vars(("z", "y", "x")),
        shifted_affine,
        units=get_voxel_to_world_units(sample_voxeldata_3dt),
    )

    with pytest.raises(ValueError, match="does not share data's voxel grid"):
        extract_with_mask(sample_voxeldata_3dt, mask)


def test_extract_with_mask_preserves_dask_laziness(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Dask-backed fUSI data remain lazy after extraction."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    chunked = sample_voxeldata_3dt.chunk({"time": 2})

    result = extract_with_mask(chunked, mask)

    assert hasattr(result.data, "chunks")
    xr.testing.assert_identical(
        result.compute(), extract_with_mask(sample_voxeldata_3dt, mask)
    )


def test_unmask_reconstructs_masked_fusi_grid(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Extraction followed by reconstruction preserves selected voxel signals."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    signals = extract_with_mask(sample_voxeldata_3dt, mask)
    restored = unmask(signals, mask)

    assert restored.dims == sample_voxeldata_3dt.dims
    assert has_voxel_to_world_index(restored)
    np.testing.assert_allclose(
        get_voxel_to_world_affine(restored), get_voxel_to_world_affine(mask)
    )
    assert get_voxel_to_world_units(restored) == get_voxel_to_world_units(mask)
    np.testing.assert_array_equal(
        restored.coords["time"].values, sample_voxeldata_3dt.coords["time"].values
    )
    np.testing.assert_array_equal(
        restored.values[:, mask.values], sample_voxeldata_3dt.values[:, mask.values]
    )
    assert np.all(restored.values[:, ~mask.values] == 0.0)


def test_unmask_rejects_invalid_mask_dtype(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Reconstruction rejects masks that are neither boolean nor binary numeric."""
    mask = make_sample_voxeldata_mask(str)
    mask.data[0, 1, 2] = "a"

    with pytest.raises(TypeError, match="binary numeric dtype"):
        unmask(np.array([1.0]), mask)


def test_unmask_rejects_non_voxeldata_mask() -> None:
    """unmask requires a VoxelData mask; it always reconstructs a VoxelData array."""
    mask = xr.DataArray(
        [[False, True], [True, False]],
        dims=("row", "column"),
        coords={"row": ["a", "b"], "column": [0, 1]},
    )

    with pytest.raises(ValueError, match="missing voxel dimension"):
        unmask(np.array([3.0, 7.0]), mask)


def test_unmask_validates_signal_space_size(
    sample_voxeldata_3dt: xr.DataArray, make_sample_voxeldata_mask
) -> None:
    """Reconstruction requires one value for every selected voxel."""
    mask = make_sample_voxeldata_mask()
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    with pytest.raises(ValueError, match="doesn't match"):
        unmask(np.array([1.0]), mask)
