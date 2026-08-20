"""Tests for extract.extract_with_labels."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from confusius import extract
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
)
from confusius.xarray import create_voxeldata


def _canonical(data, dims):
    """Build a canonical voxel-grid DataArray for `data` with `dims`.

    Parameters
    ----------
    data : numpy.ndarray or dask.array.Array
        Raw array whose rank matches `dims`.
    dims : sequence[str]
        Dimension names, e.g. `("time", "k", "j", "i")`.

    Returns
    -------
    xarray.DataArray
        Canonical DataArray with a `VoxelToWorldIndex` on its `k`/`j`/`i` dims.
    """
    dt = 1.0 if "time" in dims else None
    return create_voxeldata(data, dims=dims, spacing=(1.0, 1.0, 1.0), dt=dt)


def _labels_like(data, dims, reference):
    """Build a canonical label map sharing `reference`'s voxel grid.

    Parameters
    ----------
    data : numpy.ndarray
        Raw label values, shaped to match `dims`.
    dims : sequence[str]
        Dimension names. Any non-`k`/`j`/`i` dim (e.g. `mask`) must be leading.
    reference : xarray.DataArray
        Canonical DataArray whose `k`/`j`/`i` coordinates and affine the label map
        must share.

    Returns
    -------
    xarray.DataArray
        Canonical label map with `reference`'s `VoxelToWorldIndex`.
    """
    spatial_dims = ("k", "j", "i")
    labels = xr.DataArray(
        data,
        dims=dims,
        coords={dim: reference.coords[dim] for dim in spatial_dims},
    )
    world_coord_names = get_voxel_to_world_coord_names(reference)
    return attach_voxel_to_world_index(
        labels,
        get_voxel_to_world_affine(reference),
        world_coord_attrs={
            name: dict(reference.coords[name].attrs) for name in world_coord_names
        },
    )


class TestWithLabels:
    """Tests for extract.extract_with_labels function."""

    def test_labels_type_validation(self, sample_fusi_3dt):
        """Test that non-DataArray labels raises TypeError."""
        with pytest.raises(TypeError, match="xarray.DataArray"):
            extract.extract_with_labels(
                sample_fusi_3dt,
                np.zeros((4, 6, 8), dtype=int),  # ty: ignore[invalid-argument-type]
            )

    def test_labels_dtype_validation(self, sample_fusi_3dt):
        """Test that non-integer labels raises TypeError."""
        labels = _canonical(np.random.rand(*sample_fusi_3dt.shape[1:]), ("k", "j", "i"))
        with pytest.raises(TypeError, match="integer dtype"):
            extract.extract_with_labels(sample_fusi_3dt, labels)

    def test_boolean_labels_rejected(self, sample_fusi_3dt):
        """Test that boolean dtype labels raises TypeError."""
        labels = _canonical(
            np.ones(sample_fusi_3dt.shape[1:], dtype=bool), ("k", "j", "i")
        )
        with pytest.raises(TypeError, match="integer dtype"):
            extract.extract_with_labels(sample_fusi_3dt, labels)

    def test_missing_spatial_dim(self, sample_fusi_3dt):
        """Test that labels missing native voxel dims raises ValueError."""
        labels = xr.DataArray(np.array([1, 0, 2], dtype=int), dims=["w"])
        with pytest.raises(
            ValueError, match="native voxel dimensions|missing voxel dimension"
        ):
            extract.extract_with_labels(sample_fusi_3dt, labels)

    def test_output_dims_4d(self, sample_fusi_3dt):
        """Test that spatial dims are replaced by region for 3D+t data."""
        labels_data = np.zeros((4, 6, 8), dtype=int)
        labels_data[:2, :, :] = 1
        labels_data[2:, :, :] = 2
        labels = _labels_like(labels_data, ("k", "j", "i"), sample_fusi_3dt)

        result = extract.extract_with_labels(sample_fusi_3dt, labels)

        assert result.dims == ("time", "region")
        np.testing.assert_array_equal(result.coords["region"].values, [1, 2])

    def test_restores_scalar_voxel_dim(self, sample_fusi_3dt):
        """`data` with a scalar-reduced voxel dim (e.g. from `.isel(k=0)`) works.

        `.isel(k=0)` collapses `k` to a scalar coordinate, dropping the dim itself,
        which only `ensure_voxeldata` restores -- exercises that
        `extract_with_labels` canonicalizes `data` itself rather than relying on
        `validate_labels`'s internal (and discarded) canonicalization of it.
        """
        single_k = sample_fusi_3dt.isel(k=0)
        labels_data = np.zeros(sample_fusi_3dt.shape[1:], dtype=int)
        labels_data[:, :, :4] = 1
        labels_data[:, :, 4:] = 2
        labels = _labels_like(labels_data, ("k", "j", "i"), sample_fusi_3dt).isel(
            k=[0]
        )

        result = extract.extract_with_labels(single_k, labels)

        assert result.dims == ("time", "region")
        np.testing.assert_array_equal(result.coords["region"].values, [1, 2])

    def test_output_dims_3d(self):
        """Test that spatial dims are fully replaced for pure spatial data."""
        data = _canonical(np.ones((3, 4, 5)), ("k", "j", "i"))
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2
        labels_data[2, :, :] = 3
        labels = _canonical(labels_data, ("k", "j", "i"))

        result = extract.extract_with_labels(data, labels)

        assert result.dims == ("region",)
        np.testing.assert_array_equal(result.coords["region"].values, [1, 2, 3])

    def test_background_excluded(self):
        """Test that label 0 (background) is not included in output."""
        data = _canonical(np.ones((1, 5, 5)), ("k", "j", "i"))
        labels_data = np.zeros((1, 5, 5), dtype=int)
        labels_data[:, 2:, :] = 1
        labels = _canonical(labels_data, ("k", "j", "i"))

        result = extract.extract_with_labels(data, labels)

        assert 0 not in result.coords["region"].values
        assert 1 in result.coords["region"].values

    @pytest.mark.parametrize(
        "reduction,np_func",
        [
            ("mean", np.mean),
            ("sum", np.sum),
            ("median", np.median),
            ("min", np.min),
            ("max", np.max),
            ("var", np.var),
            ("std", np.std),
        ],
    )
    def test_reduction_correctness(self, reduction, np_func):
        """Test that each reduction matches the corresponding numpy function."""
        rng = np.random.default_rng(0)
        data_vals = rng.random((3, 4, 5))
        data = _canonical(data_vals, ("k", "j", "i"))

        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2
        labels = _canonical(labels_data, ("k", "j", "i"))

        result = extract.extract_with_labels(data, labels, reduction=reduction)

        np.testing.assert_allclose(
            result.sel(region=1).values, np_func(data_vals[0, :, :])
        )
        np.testing.assert_allclose(
            result.sel(region=2).values, np_func(data_vals[1, :, :])
        )

    def test_invalid_reduction(self):
        """Test that an invalid reduction string raises ValueError."""
        data = _canonical(np.ones((1, 3, 4)), ("k", "j", "i"))
        labels = _canonical(np.ones((1, 3, 4), dtype=int), ("k", "j", "i"))

        with pytest.raises(ValueError, match="Invalid reduction"):
            extract.extract_with_labels(data, labels, reduction="invalid")  # ty: ignore[invalid-argument-type]

    def test_dask_laziness(self):
        """Test that the result is lazy when the input is a Dask-backed array."""
        rng = np.random.default_rng(0)
        data_vals = rng.random((10, 3, 4, 5))
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2

        data_dask = _canonical(
            da.from_array(data_vals, chunks=(10, 3, 4, 5)), ("time", "k", "j", "i")
        )
        labels = _canonical(labels_data, ("k", "j", "i"))

        result = extract.extract_with_labels(data_dask, labels)

        # Result must still be lazy.
        assert isinstance(result.data, da.Array)

        # Values must match the eager reference.
        data_eager = _canonical(data_vals, ("time", "k", "j", "i"))
        expected = extract.extract_with_labels(data_eager, labels)
        np.testing.assert_allclose(result.values, expected.values)

    def test_stacked_masks_format(self, sample_fusi_3dt):
        """Test extraction with stacked mask format (masks, z, y, x)."""
        _, nz, ny, nx = sample_fusi_3dt.shape

        # Build a stacked mask with two named regions.
        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0, :, :] = 1  # Region "VISp": first k-slice.
        mask_data[1, 1, :, :] = 2  # Region "AUDp": second k-slice.
        labels = _labels_like(
            mask_data, ("mask", "k", "j", "i"), sample_fusi_3dt
        ).assign_coords(mask=["VISp", "AUDp"])

        result = extract.extract_with_labels(sample_fusi_3dt, labels)

        assert set(result.dims) == {"time", "region"}
        np.testing.assert_array_equal(result.coords["region"].values, ["VISp", "AUDp"])
        np.testing.assert_allclose(
            result.sel(region="VISp").values,
            sample_fusi_3dt.values[:, 0, :, :].mean(axis=(-2, -1)),
        )
        np.testing.assert_allclose(
            result.sel(region="AUDp").values,
            sample_fusi_3dt.values[:, 1, :, :].mean(axis=(-2, -1)),
        )

    def test_stacked_masks_overlapping(self, sample_fusi_3dt):
        """Test extraction with overlapping stacked masks."""
        _, nz, ny, nx = sample_fusi_3dt.shape

        # Region "A": k-slices 0 and 1; Region "B": k-slices 1 and 2 — k=1 overlaps.
        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0:2, :, :] = 1  # Region "A": slices 0–1.
        mask_data[1, 1:3, :, :] = 2  # Region "B": slices 1–2.
        labels = _labels_like(
            mask_data, ("mask", "k", "j", "i"), sample_fusi_3dt
        ).assign_coords(mask=["A", "B"])

        result = extract.extract_with_labels(sample_fusi_3dt, labels)

        assert set(result.dims) == {"time", "region"}
        np.testing.assert_array_equal(result.coords["region"].values, ["A", "B"])
        np.testing.assert_allclose(
            result.sel(region="A").values,
            sample_fusi_3dt.values[:, 0:2, :, :].mean(axis=(-3, -2, -1)),
        )
        np.testing.assert_allclose(
            result.sel(region="B").values,
            sample_fusi_3dt.values[:, 1:3, :, :].mean(axis=(-3, -2, -1)),
        )

    def test_stacked_masks_duplicate_ids_non_overlapping(self, sample_fusi_3dt):
        """Non-overlapping layers sharing the same raw id must stay distinct regions.

        Regression test: layer position along `mask`, not the layer's own non-zero
        value, is what identifies a region — mirrors Atlas.get_masks reusing a
        region's id across its left/right hemisphere layers.
        """
        _, nz, ny, nx = sample_fusi_3dt.shape

        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0, :, :] = 7  # Region "VISp_L": first k-slice, id 7.
        mask_data[1, 1, :, :] = 7  # Region "VISp_R": second k-slice, same id 7.
        labels = _labels_like(
            mask_data, ("mask", "k", "j", "i"), sample_fusi_3dt
        ).assign_coords(mask=["VISp_L", "VISp_R"])

        result = extract.extract_with_labels(sample_fusi_3dt, labels)

        np.testing.assert_array_equal(
            result.coords["region"].values, ["VISp_L", "VISp_R"]
        )
        np.testing.assert_allclose(
            result.sel(region="VISp_L").values,
            sample_fusi_3dt.values[:, 0, :, :].mean(axis=(-2, -1)),
        )
        np.testing.assert_allclose(
            result.sel(region="VISp_R").values,
            sample_fusi_3dt.values[:, 1, :, :].mean(axis=(-2, -1)),
        )

    def test_stacked_masks_duplicate_ids_overlapping(self, sample_fusi_3dt):
        """Overlapping layers sharing the same raw id must stay distinct regions."""
        _, nz, ny, nx = sample_fusi_3dt.shape

        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0:2, :, :] = 3  # Region "A": slices 0-1, id 3.
        mask_data[1, 1:3, :, :] = 3  # Region "B": slices 1-2, same id 3.
        labels = _labels_like(
            mask_data, ("mask", "k", "j", "i"), sample_fusi_3dt
        ).assign_coords(mask=["A", "B"])

        result = extract.extract_with_labels(sample_fusi_3dt, labels)

        np.testing.assert_array_equal(result.coords["region"].values, ["A", "B"])
        np.testing.assert_allclose(
            result.sel(region="A").values,
            sample_fusi_3dt.values[:, 0:2, :, :].mean(axis=(-3, -2, -1)),
        )
        np.testing.assert_allclose(
            result.sel(region="B").values,
            sample_fusi_3dt.values[:, 1:3, :, :].mean(axis=(-3, -2, -1)),
        )

    def test_stacked_mask_layer_wrong_nonzero_count_raises(self, sample_fusi_3dt):
        """A layer with zero or multiple distinct non-zero values must raise."""
        _, nz, ny, nx = sample_fusi_3dt.shape

        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0, :, :] = 1
        mask_data[1, 1, : ny // 2, :] = 2  # Layer 1 has two distinct values below.
        mask_data[1, 1, ny // 2 :, :] = 3
        labels = _labels_like(
            mask_data, ("mask", "k", "j", "i"), sample_fusi_3dt
        ).assign_coords(mask=["A", "B"])

        with pytest.raises(ValueError, match="exactly one unique non-zero"):
            extract.extract_with_labels(sample_fusi_3dt, labels)

    def test_dask_spatial_chunks(self):
        """Test correctness when spatial dims are chunked in the Dask array."""
        rng = np.random.default_rng(42)
        data_vals = rng.random((10, 3, 4, 5))
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2

        data_dask = _canonical(
            da.from_array(data_vals, chunks=(5, 1, 2, 3)), ("time", "k", "j", "i")
        )
        labels = _canonical(labels_data, ("k", "j", "i"))

        result = extract.extract_with_labels(data_dask, labels)

        data_eager = _canonical(data_vals, ("time", "k", "j", "i"))
        expected = extract.extract_with_labels(data_eager, labels)
        np.testing.assert_allclose(result.values, expected.values)

    def test_dask_backed_labels(self):
        """Test that Dask-backed labels do not raise and produce correct results.

        Regression test for: flox raises ValueError when the groupby array is a
        Dask array and expected_groups is not provided.
        """
        rng = np.random.default_rng(0)
        data_vals = rng.random((10, 3, 4, 5))
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2

        data = _canonical(data_vals, ("time", "k", "j", "i"))
        labels_dask = _canonical(
            da.from_array(labels_data, chunks=(1, 2, 3)), ("k", "j", "i")
        )

        result = extract.extract_with_labels(data, labels_dask)

        labels_eager = _canonical(labels_data, ("k", "j", "i"))
        expected = extract.extract_with_labels(data, labels_eager)
        np.testing.assert_allclose(result.values, expected.values)
