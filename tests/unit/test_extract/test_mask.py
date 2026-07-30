"""Tests for mask extraction and reconstruction."""

import numpy as np
import pytest
import xarray as xr

from confusius.extract import extract_with_mask, unmask


def _make_mask(
    data: xr.DataArray,
    dtype: type[np.generic] | type[bool] | type[float] = bool,
) -> xr.DataArray:
    """Create an empty mask on `data`'s spatial grid.

    Parameters
    ----------
    data : xarray.DataArray
        fUSI DataArray supplying the spatial grid.
    dtype : type[numpy.generic] or type[bool] or type[float], default: bool
        Mask data type.

    Returns
    -------
    xarray.DataArray
        Zero-valued mask with `data`'s `z`, `y`, and `x` coordinates.
    """
    return xr.DataArray(
        np.zeros(tuple(data.sizes[dim] for dim in ("z", "y", "x")), dtype=dtype),
        dims=("z", "y", "x"),
        coords={dim: data.coords[dim] for dim in ("z", "y", "x")},
    )


def test_extract_with_mask_selects_expected_voxels(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Extraction retains the masked voxel signals and spatial index."""
    mask = _make_mask(sample_3dt_volume)
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    signals = extract_with_mask(sample_3dt_volume, mask)

    assert signals.dims == ("time", "space")
    np.testing.assert_array_equal(
        signals.values,
        sample_3dt_volume.values[:, [0, 1], [1, 2], [2, 3]],
    )


def test_extract_with_mask_supports_generic_feature_dimensions() -> None:
    """Extraction remains available for non-fUSI feature grids."""
    data = xr.DataArray(np.arange(12).reshape(3, 4), dims=("time", "feature"))
    mask = xr.DataArray([True, False, True, False], dims="feature")

    signals = extract_with_mask(data, mask)

    assert signals.dims == ("time", "space")
    np.testing.assert_array_equal(signals.values, [[0, 2], [4, 6], [8, 10]])


def test_extract_with_mask_accepts_single_label_integer_mask(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """A single non-zero integer label has the same selection as a boolean mask."""
    boolean_mask = _make_mask(sample_3dt_volume)
    boolean_mask.data[0, 1, 2] = True
    integer_mask = boolean_mask.astype(np.int32) * 7

    expected = extract_with_mask(sample_3dt_volume, boolean_mask)
    result = extract_with_mask(sample_3dt_volume, integer_mask)

    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    ("dtype", "values", "message"),
    [
        (float, (1.0,), "single-label integer dtype"),
        (np.int32, (1, 2), "2 distinct non-zero"),
    ],
)
def test_extract_with_mask_rejects_invalid_mask_values(
    sample_3dt_volume: xr.DataArray,
    dtype: type[np.generic] | type[float],
    values: tuple[float, ...],
    message: str,
) -> None:
    """Only boolean and single-label integer masks are accepted."""
    mask = _make_mask(sample_3dt_volume, dtype)
    mask.data.flat[: len(values)] = values

    with pytest.raises(TypeError, match=message):
        extract_with_mask(sample_3dt_volume, mask)


def test_extract_with_mask_rejects_misaligned_coordinates(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Extraction rejects masks from a different fUSI grid."""
    mask = _make_mask(sample_3dt_volume)
    mask.data[0, 1, 2] = True
    mask = mask.assign_coords(x=mask.x + 1.0)

    with pytest.raises(ValueError, match="does not match between mask and data"):
        extract_with_mask(sample_3dt_volume, mask)


def test_extract_with_mask_preserves_dask_laziness(
    sample_3dt_volume: xr.DataArray,
) -> None:
    """Dask-backed fUSI data remain lazy after extraction."""
    mask = _make_mask(sample_3dt_volume)
    mask.data[0, 1, 2] = True
    chunked = sample_3dt_volume.chunk({"time": 2})

    result = extract_with_mask(chunked, mask)

    assert hasattr(result.data, "chunks")
    xr.testing.assert_identical(
        result.compute(), extract_with_mask(sample_3dt_volume, mask)
    )


def test_unmask_reconstructs_masked_fusi_grid(sample_3dt_volume: xr.DataArray) -> None:
    """Extraction followed by reconstruction preserves selected voxel signals."""
    mask = _make_mask(sample_3dt_volume)
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    signals = extract_with_mask(sample_3dt_volume, mask)
    restored = unmask(signals, mask)

    np.testing.assert_array_equal(
        restored.values[:, mask.values], sample_3dt_volume.values[:, mask.values]
    )
    assert np.all(restored.values[:, ~mask.values] == 0.0)


def test_unmask_rejects_invalid_mask_dtype(sample_3dt_volume: xr.DataArray) -> None:
    """Reconstruction rejects masks that are neither boolean nor integer."""
    mask = _make_mask(sample_3dt_volume, float)
    mask.data[0, 1, 2] = 1.0

    with pytest.raises(TypeError, match="single-label integer dtype"):
        unmask(np.array([1.0]), mask)


def test_unmask_supports_generic_mask_dimensions() -> None:
    """Reconstruction remains available for non-fUSI feature grids."""
    mask = xr.DataArray(
        [[False, True], [True, False]],
        dims=("row", "column"),
        coords={"row": ["a", "b"], "column": [0, 1]},
    )

    restored = unmask(np.array([3.0, 7.0]), mask)

    assert restored.dims == ("row", "column")
    np.testing.assert_array_equal(restored.values, [[0.0, 3.0], [7.0, 0.0]])


def test_unmask_validates_signal_space_size(sample_3dt_volume: xr.DataArray) -> None:
    """Reconstruction requires one value for every selected voxel."""
    mask = _make_mask(sample_3dt_volume)
    mask.data[0, 1, 2] = True
    mask.data[1, 2, 3] = True

    with pytest.raises(ValueError, match="doesn't match"):
        unmask(np.array([1.0]), mask)
