"""Tests for extract.extract_with_labels."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from confusius import extract


class TestWithLabels:
    """Tests for extract.extract_with_labels function."""

    def test_labels_type_validation(self, sample_4d_volume):
        """Test that non-DataArray labels raises TypeError."""
        with pytest.raises(TypeError, match="xarray.DataArray"):
            extract.extract_with_labels(
                sample_4d_volume,
                np.zeros((4, 6, 8), dtype=int),  # type: ignore[arg-type]
            )

    def test_labels_dtype_validation(self, sample_4d_volume):
        """Test that non-integer labels raises TypeError."""
        labels = xr.DataArray(
            np.random.rand(*sample_4d_volume.shape[1:]),
            dims=["z", "y", "x"],
        )
        with pytest.raises(TypeError, match="integer dtype"):
            extract.extract_with_labels(sample_4d_volume, labels)

    def test_boolean_labels_rejected(self, sample_4d_volume):
        """Test that boolean dtype labels raises TypeError."""
        labels = xr.DataArray(
            np.ones(sample_4d_volume.shape[1:], dtype=bool),
            dims=["z", "y", "x"],
        )
        with pytest.raises(TypeError, match="integer dtype"):
            extract.extract_with_labels(sample_4d_volume, labels)

    def test_missing_spatial_dim(self, sample_4d_volume):
        """Test that labels with dimension not in data raises ValueError."""
        labels = xr.DataArray(np.array([1, 0, 2], dtype=int), dims=["w"])
        with pytest.raises(ValueError, match="missing spatial dimensions.*'w'"):
            extract.extract_with_labels(sample_4d_volume, labels)

    def test_output_dims_4d(self, sample_4d_volume):
        """Test that spatial dims are replaced by region for 3D+t data."""
        labels_data = np.zeros((4, 6, 8), dtype=int)
        labels_data[:2, :, :] = 1
        labels_data[2:, :, :] = 2
        labels = xr.DataArray(
            labels_data,
            dims=["z", "y", "x"],
            coords={
                "z": sample_4d_volume.coords["z"],
                "y": sample_4d_volume.coords["y"],
                "x": sample_4d_volume.coords["x"],
            },
        )

        result = extract.extract_with_labels(sample_4d_volume, labels)

        assert result.dims == ("time", "region")
        np.testing.assert_array_equal(result.coords["region"].values, [1, 2])

    def test_output_dims_3d(self):
        """Test that spatial dims are fully replaced for pure spatial data."""
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"])
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2
        labels_data[2, :, :] = 3
        labels = xr.DataArray(labels_data, dims=["z", "y", "x"])

        result = extract.extract_with_labels(data, labels)

        assert result.dims == ("region",)
        np.testing.assert_array_equal(result.coords["region"].values, [1, 2, 3])

    def test_background_excluded(self):
        """Test that label 0 (background) is not included in output."""
        data = xr.DataArray(np.ones((5, 5)), dims=["y", "x"])
        labels_data = np.zeros((5, 5), dtype=int)
        labels_data[2:, :] = 1
        labels = xr.DataArray(labels_data, dims=["y", "x"])

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
        data = xr.DataArray(data_vals, dims=["z", "y", "x"])

        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2
        labels = xr.DataArray(labels_data, dims=["z", "y", "x"])

        result = extract.extract_with_labels(data, labels, reduction=reduction)

        np.testing.assert_allclose(
            result.sel(region=1).values, np_func(data_vals[0, :, :])
        )
        np.testing.assert_allclose(
            result.sel(region=2).values, np_func(data_vals[1, :, :])
        )

    def test_invalid_reduction(self):
        """Test that an invalid reduction string raises ValueError."""
        data = xr.DataArray(np.ones((3, 4)), dims=["y", "x"])
        labels = xr.DataArray(np.ones((3, 4), dtype=int), dims=["y", "x"])

        with pytest.raises(ValueError, match="Invalid reduction"):
            extract.extract_with_labels(data, labels, reduction="invalid")  # type: ignore[arg-type]

    def test_dask_laziness(self):
        """Test that the result is lazy when the input is a Dask-backed array."""
        rng = np.random.default_rng(0)
        data_vals = rng.random((10, 3, 4, 5))
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2

        data_dask = xr.DataArray(
            da.from_array(data_vals, chunks=(10, 3, 4, 5)),
            dims=["time", "z", "y", "x"],
        )
        labels = xr.DataArray(labels_data, dims=["z", "y", "x"])

        result = extract.extract_with_labels(data_dask, labels)

        # Result must still be lazy.
        assert isinstance(result.data, da.Array)

        # Values must match the eager reference.
        data_eager = xr.DataArray(data_vals, dims=["time", "z", "y", "x"])
        expected = extract.extract_with_labels(data_eager, labels)
        np.testing.assert_allclose(result.values, expected.values)

    def test_stacked_masks_format(self, sample_4d_volume):
        """Test extraction with stacked mask format (masks, z, y, x)."""
        _, nz, ny, nx = sample_4d_volume.shape

        # Build a stacked mask with two named regions.
        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0, :, :] = 1  # Region "VISp": first z-slice.
        mask_data[1, 1, :, :] = 2  # Region "AUDp": second z-slice.
        labels = xr.DataArray(
            mask_data,
            dims=["mask", "z", "y", "x"],
            coords={
                "mask": ["VISp", "AUDp"],
                "z": sample_4d_volume.coords["z"],
                "y": sample_4d_volume.coords["y"],
                "x": sample_4d_volume.coords["x"],
            },
        )

        result = extract.extract_with_labels(sample_4d_volume, labels)

        assert set(result.dims) == {"time", "region"}
        np.testing.assert_array_equal(result.coords["region"].values, ["VISp", "AUDp"])
        np.testing.assert_allclose(
            result.sel(region="VISp").values,
            sample_4d_volume.values[:, 0, :, :].mean(axis=(-2, -1)),
        )
        np.testing.assert_allclose(
            result.sel(region="AUDp").values,
            sample_4d_volume.values[:, 1, :, :].mean(axis=(-2, -1)),
        )

    def test_stacked_masks_overlapping(self, sample_4d_volume):
        """Test extraction with overlapping stacked masks."""
        _, nz, ny, nx = sample_4d_volume.shape

        # Region "A": z-slices 0 and 1; Region "B": z-slices 1 and 2 — z=1 overlaps.
        mask_data = np.zeros((2, nz, ny, nx), dtype=int)
        mask_data[0, 0:2, :, :] = 1  # Region "A": slices 0–1.
        mask_data[1, 1:3, :, :] = 2  # Region "B": slices 1–2.
        labels = xr.DataArray(
            mask_data,
            dims=["mask", "z", "y", "x"],
            coords={
                "mask": ["A", "B"],
                "z": sample_4d_volume.coords["z"],
                "y": sample_4d_volume.coords["y"],
                "x": sample_4d_volume.coords["x"],
            },
        )

        result = extract.extract_with_labels(sample_4d_volume, labels)

        assert set(result.dims) == {"time", "region"}
        np.testing.assert_array_equal(result.coords["region"].values, ["A", "B"])
        np.testing.assert_allclose(
            result.sel(region="A").values,
            sample_4d_volume.values[:, 0:2, :, :].mean(axis=(-3, -2, -1)),
        )
        np.testing.assert_allclose(
            result.sel(region="B").values,
            sample_4d_volume.values[:, 1:3, :, :].mean(axis=(-3, -2, -1)),
        )

    def test_dask_spatial_chunks(self):
        """Test correctness when spatial dims are chunked in the Dask array."""
        rng = np.random.default_rng(42)
        data_vals = rng.random((10, 3, 4, 5))
        labels_data = np.zeros((3, 4, 5), dtype=int)
        labels_data[0, :, :] = 1
        labels_data[1, :, :] = 2

        data_dask = xr.DataArray(
            da.from_array(data_vals, chunks=(5, 1, 2, 3)),
            dims=["time", "z", "y", "x"],
        )
        labels = xr.DataArray(labels_data, dims=["z", "y", "x"])

        result = extract.extract_with_labels(data_dask, labels)

        data_eager = xr.DataArray(data_vals, dims=["time", "z", "y", "x"])
        expected = extract.extract_with_labels(data_eager, labels)
        np.testing.assert_allclose(result.values, expected.values)
