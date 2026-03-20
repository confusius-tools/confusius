"""Tests for spatial smoothing functions."""

import numpy as np
import pytest
import scipy.ndimage
import xarray as xr

from confusius.spatial import smooth_volume


def _spacing(da, dim):
    """Return the uniform voxel spacing (mm) for a dimension."""
    return float(np.diff(da.coords[dim].values)[0])


class TestSmoothVolume:
    """Tests for smooth_volume."""

    def test_matches_scipy_3d(self, sample_3d_volume):
        """smooth_volume should match scipy.ndimage.gaussian_filter on a 3D volume."""
        vol = sample_3d_volume
        fwhm = 0.4

        smoothed = smooth_volume(vol, fwhm=fwhm)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm * fwhm_to_sigma / _spacing(vol, d) for d in ["z", "y", "x"]
        ]
        expected = scipy.ndimage.gaussian_filter(vol.values.astype(float), expected_sigmas)

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_matches_scipy_4d_skips_time(self, sample_4d_volume):
        """Time dimension should not be smoothed (sigma=0)."""
        vol = sample_4d_volume
        fwhm = 0.4

        smoothed = smooth_volume(vol, fwhm=fwhm)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [0.0] + [
            fwhm * fwhm_to_sigma / _spacing(vol, d) for d in ["z", "y", "x"]
        ]
        expected = scipy.ndimage.gaussian_filter(vol.values.astype(float), expected_sigmas)

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_anisotropic_fwhm_dict(self, sample_3d_volume):
        """Per-dimension FWHM dict should produce the correct per-dim sigmas."""
        vol = sample_3d_volume
        fwhm_dict = {"z": 0.6, "y": 0.2, "x": 0.4}

        smoothed = smooth_volume(vol, fwhm=fwhm_dict)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm_dict[d] * fwhm_to_sigma / _spacing(vol, d) for d in ["z", "y", "x"]
        ]
        expected = scipy.ndimage.gaussian_filter(vol.values.astype(float), expected_sigmas)

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_selected_dims_only(self, sample_3d_volume):
        """Smoothing only selected dims should leave the rest unchanged."""
        vol = sample_3d_volume
        fwhm = 0.4

        smoothed = smooth_volume(vol, fwhm=fwhm, dims=["z", "x"])

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm * fwhm_to_sigma / _spacing(vol, "z"),
            0.0,  # y not smoothed.
            fwhm * fwhm_to_sigma / _spacing(vol, "x"),
        ]
        expected = scipy.ndimage.gaussian_filter(vol.values.astype(float), expected_sigmas)

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_preserves_coords_and_attrs(self, sample_3d_volume):
        """Output should have identical coordinates and attributes."""
        vol = sample_3d_volume
        smoothed = smooth_volume(vol, fwhm=0.3)

        assert smoothed.dims == vol.dims
        assert smoothed.shape == vol.shape
        assert smoothed.attrs == vol.attrs
        for dim in vol.dims:
            np.testing.assert_array_equal(smoothed.coords[dim], vol.coords[dim])

    def test_zero_fwhm_is_identity(self, sample_3d_volume):
        """FWHM=0 should return a result numerically identical to the input."""
        vol = sample_3d_volume
        smoothed = smooth_volume(vol, fwhm=0.0)
        np.testing.assert_allclose(smoothed.values, vol.values, rtol=1e-10)

    def test_fwhm_correct_on_impulse(self):
        """Smoothing a Dirac delta should produce a blob with the requested FWHM.

        Uses a point source at the center of a large volume and verifies that
        the number of voxels above half-maximum along each axis equals
        fwhm / spacing (requires fwhm / spacing to be an odd integer so the
        measurement is exact).
        """
        # Spacing 1.0, FWHM 9.0 → 9 voxels wide at half-max (odd integer).
        spacing = 1.0
        fwhm_val = 9.0
        shape = (40, 41, 42)
        center = tuple(s // 2 for s in shape)

        data = np.zeros(shape)
        data[center] = 1.0
        vol = xr.DataArray(
            data,
            dims=["z", "y", "x"],
            coords={d: np.arange(s) * spacing for d, s in zip(["z", "y", "x"], shape)},
        )

        smoothed = smooth_volume(vol, fwhm=fwhm_val)
        arr = smoothed.values

        above_half_max = arr > 0.5 * arr.max()
        expected_voxels = int(fwhm_val / spacing)
        for axis in range(3):
            # Project onto this axis by collapsing the other two.
            proj = above_half_max.any(axis=tuple(i for i in range(3) if i != axis))
            assert proj.sum() == expected_voxels

    def test_nans_propagate_by_default(self, sample_3d_volume):
        """NaNs propagate to neighbouring voxels when ensure_finite=False (default)."""
        vol = sample_3d_volume.copy()
        vol.values[2, 3, 4] = np.nan

        smoothed = smooth_volume(vol, fwhm=0.3)

        assert np.isnan(smoothed.values).any()

    def test_ensure_finite_suppresses_nan_propagation(self, sample_3d_volume):
        """ensure_finite=True should replace non-finite values so they don't spread."""
        vol = sample_3d_volume.copy()
        vol.values[2, 3, 4] = np.nan

        smoothed = smooth_volume(vol, fwhm=0.3, ensure_finite=True)

        assert not np.isnan(smoothed.values).any()

    def test_dask_chunked_time_ok(self, sample_4d_volume):
        """Dask arrays chunked along time (not spatial dims) should work."""
        pytest.importorskip("dask.array")
        vol = sample_4d_volume
        vol_dask = vol.chunk({"time": 5})  # Only time is chunked.

        smoothed = smooth_volume(vol_dask, fwhm=0.3)
        smoothed_eager = smooth_volume(vol, fwhm=0.3)

        np.testing.assert_allclose(
            smoothed.compute().values, smoothed_eager.values, rtol=1e-10
        )

    def test_raises_invalid_dim(self, sample_3d_volume):
        """Should raise ValueError for dimensions not in the DataArray."""
        with pytest.raises(ValueError, match="not present in the DataArray"):
            smooth_volume(sample_3d_volume, fwhm=0.3, dims=["z", "nonexistent"])

    def test_raises_nonuniform_spacing(self):
        """Should raise ValueError if a smoothed dim has non-uniform spacing."""
        coords = np.concatenate([np.arange(5), np.arange(6, 12)]) * 0.1
        vol = xr.DataArray(
            np.ones((11, 8, 10)),
            dims=["z", "y", "x"],
            coords={"z": coords, "y": np.arange(8) * 0.1, "x": np.arange(10) * 0.1},
        )
        with pytest.raises(ValueError, match="non-uniform or undefined coordinate spacing"):
            smooth_volume(vol, fwhm=0.3)

    def test_raises_missing_coord(self):
        """Should raise ValueError if a smoothed dim has no coordinate."""
        vol = xr.DataArray(np.ones((8, 10, 12)), dims=["z", "y", "x"])
        with pytest.raises(ValueError, match="non-uniform or undefined coordinate spacing"):
            smooth_volume(vol, fwhm=0.3)

    def test_raises_unknown_fwhm_key(self, sample_3d_volume):
        """Should raise ValueError if fwhm dict contains unknown dim names."""
        with pytest.raises(ValueError, match="not in the set of smoothed dimensions"):
            smooth_volume(sample_3d_volume, fwhm={"z": 0.3, "w": 0.2})

    def test_raises_chunked_spatial_dim(self, sample_3d_volume):
        """Should raise ValueError if a smoothed spatial dim is Dask-chunked."""
        pytest.importorskip("dask.array")
        vol_dask = sample_3d_volume.chunk({"z": 2})  # Spatial dim chunked.
        with pytest.raises(ValueError, match="is chunked"):
            smooth_volume(vol_dask, fwhm=0.3)
