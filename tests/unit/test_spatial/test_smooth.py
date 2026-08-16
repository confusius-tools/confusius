"""Tests for spatial smoothing functions."""

import numpy as np
import pytest
import scipy.ndimage
import xarray as xr

from confusius._utils.geometry import attach_voxel_to_world_index
from confusius.spatial import smooth_volume


def _add_identity_affine(da):
    """Attach identity voxel-to-world geometry."""
    world_names = tuple("zyx"[-len(da.dims) :])
    return attach_voxel_to_world_index(
        da,
        np.eye(len(da.dims) + 1),
        world_coord_attrs={name: {"units": "mm"} for name in world_names},
    )


def _spacing(da, dim):
    """Return the world voxel spacing for a dimension."""
    return float(da.fusi.spacing[dim])


class TestSmoothVolume:
    """Tests for smooth_volume."""

    def test_matches_scipy_3d(self, sample_fusi_3d):
        """smooth_volume should match scipy.ndimage.gaussian_filter on a 3D volume."""
        vol = sample_fusi_3d
        fwhm = 0.4

        smoothed = smooth_volume(vol, fwhm=fwhm)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm * fwhm_to_sigma / _spacing(vol, d) for d in ["k", "j", "i"]
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_matches_scipy_4d_skips_time(self, sample_fusi_3dt):
        """Time dimension should not be smoothed (sigma=0)."""
        vol = sample_fusi_3dt
        fwhm = 0.4

        smoothed = smooth_volume(vol, fwhm=fwhm)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [0.0] + [
            fwhm * fwhm_to_sigma / _spacing(vol, d) for d in ["k", "j", "i"]
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_anisotropic_fwhm_dict(self, sample_fusi_3d):
        """Per-dimension FWHM dict should produce the correct per-dim sigmas."""
        vol = sample_fusi_3d
        fwhm_dict = {"k": 0.6, "j": 0.2, "i": 0.4}

        smoothed = smooth_volume(vol, fwhm=fwhm_dict)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm_dict[d] * fwhm_to_sigma / _spacing(vol, d) for d in ["k", "j", "i"]
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_selected_dims_only(self, sample_fusi_3d):
        """A dict FWHM should smooth only the listed dimensions."""
        vol = sample_fusi_3d
        fwhm = 0.4

        smoothed = smooth_volume(vol, fwhm={"k": fwhm, "i": fwhm})

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm * fwhm_to_sigma / _spacing(vol, "k"),
            0.0,  # j not smoothed.
            fwhm * fwhm_to_sigma / _spacing(vol, "i"),
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_fwhm_dict_infers_smoothed_dims(self, sample_fusi_3d):
        """A dict FWHM should define the smoothed dimensions."""
        vol = sample_fusi_3d
        fwhm_dict = {"k": 0.6, "i": 0.4}

        smoothed = smooth_volume(vol, fwhm=fwhm_dict)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm_dict["k"] * fwhm_to_sigma / _spacing(vol, "k"),
            0.0,
            fwhm_dict["i"] * fwhm_to_sigma / _spacing(vol, "i"),
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_fwhm_dict_can_smooth_time(self, sample_fusi_3dt):
        """A dict FWHM should be able to target non-spatial dimensions like time."""
        vol = sample_fusi_3dt
        fwhm_dict = {"time": 1.0}

        smoothed = smooth_volume(vol, fwhm=fwhm_dict)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            fwhm_dict["time"] * fwhm_to_sigma / _spacing(vol, "time"),
            0.0,
            0.0,
            0.0,
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

    def test_preserves_coords_and_attrs(self, sample_fusi_3d):
        """Output should have identical coordinates and attributes."""
        vol = sample_fusi_3d
        smoothed = smooth_volume(vol, fwhm=0.3)

        assert smoothed.dims == vol.dims
        assert smoothed.shape == vol.shape
        assert smoothed.attrs == vol.attrs
        for dim in vol.dims:
            np.testing.assert_array_equal(smoothed.coords[dim], vol.coords[dim])

    def test_zero_fwhm_is_identity(self, sample_fusi_3d):
        """FWHM=0 should return a result numerically identical to the input."""
        vol = sample_fusi_3d
        smoothed = smooth_volume(vol, fwhm=0.0)
        np.testing.assert_allclose(smoothed.values, vol.values, rtol=1e-10)

    def test_singleton_dim_is_not_smoothed(self):
        """Length-1 dimensions should be ignored instead of requiring spacing."""
        data = np.zeros((5, 1, 7))
        data[2, 0, 3] = 1.0
        vol = _add_identity_affine(
            xr.DataArray(
                data,
                dims=["k", "j", "i"],
                coords={
                    "k": np.arange(5),
                    "j": [0],
                    "i": np.arange(7),
                },
            )
        )

        smoothed = smooth_volume(vol, fwhm=0.4)

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sigmas = [
            0.4 * fwhm_to_sigma / _spacing(vol, "k"),
            0.0,
            0.4 * fwhm_to_sigma / _spacing(vol, "i"),
        ]
        expected = scipy.ndimage.gaussian_filter(
            vol.values.astype(float), expected_sigmas
        )

        np.testing.assert_allclose(smoothed.values, expected, rtol=1e-10)

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
        vol = _add_identity_affine(
            xr.DataArray(
                data,
                dims=["k", "j", "i"],
                coords={d: np.arange(s) for d, s in zip(["k", "j", "i"], shape)},
            )
        )

        smoothed = smooth_volume(vol, fwhm=fwhm_val)
        arr = smoothed.values

        above_half_max = arr > 0.5 * arr.max()
        expected_voxels = int(fwhm_val / spacing)
        for axis in range(3):
            # Project onto this axis by collapsing the other two.
            proj = above_half_max.any(axis=tuple(i for i in range(3) if i != axis))
            assert proj.sum() == expected_voxels

    def test_nans_propagate_by_default(self, sample_fusi_3d):
        """NaNs propagate to neighbouring voxels when ensure_finite=False (default)."""
        vol = sample_fusi_3d.copy()
        vol.values[2, 3, 4] = np.nan

        smoothed = smooth_volume(vol, fwhm=0.3)

        assert np.isnan(smoothed.values).any()

    def test_ensure_finite_suppresses_nan_propagation(self, sample_fusi_3d):
        """ensure_finite=True should replace non-finite values so they don't spread."""
        vol = sample_fusi_3d.copy()
        vol.values[2, 3, 4] = np.nan

        smoothed = smooth_volume(vol, fwhm=0.3, ensure_finite=True)

        assert not np.isnan(smoothed.values).any()

    def test_dask_chunked_time_ok(self, sample_fusi_3dt):
        """Dask arrays chunked along time (not spatial dims) should work."""
        pytest.importorskip("dask.array")
        vol = sample_fusi_3dt
        vol_dask = vol.chunk({"time": 5})  # Only time is chunked.

        smoothed = smooth_volume(vol_dask, fwhm=0.3)
        smoothed_eager = smooth_volume(vol, fwhm=0.3)

        np.testing.assert_allclose(
            smoothed.compute().values, smoothed_eager.values, rtol=1e-10
        )

    def test_raises_invalid_dim(self, sample_fusi_3d):
        """Should raise ValueError for dimensions not in the DataArray."""
        with pytest.raises(ValueError, match="not present in the DataArray"):
            smooth_volume(sample_fusi_3d, fwhm={"k": 0.3, "nonexistent": 0.3})

    def test_raises_nonuniform_spacing(self):
        """Should raise ValueError if a smoothed dim has non-uniform spacing."""
        coords = np.concatenate([np.arange(5), np.arange(6, 12)]) * 0.1
        vol = xr.DataArray(
            np.ones((11, 8, 10)),
            dims=["k", "j", "i"],
            coords={"k": coords, "j": np.arange(8) * 0.1, "i": np.arange(10) * 0.1},
        )
        with pytest.raises(ValueError, match="native voxel dimensions `k/j/i`"):
            smooth_volume(vol, fwhm=0.3)

    def test_raises_missing_coord(self):
        """Should raise ValueError if a core dim has no coordinate."""
        vol = xr.DataArray(np.ones((8, 10, 12)), dims=["k", "j", "i"])
        with pytest.raises(ValueError, match="native voxel dimensions `k/j/i`"):
            smooth_volume(vol, fwhm=0.3)

    def test_raises_unknown_fwhm_key(self, sample_fusi_3d):
        """Should raise ValueError if fwhm dict contains dim names not in the array."""
        with pytest.raises(ValueError, match="not present in the DataArray"):
            smooth_volume(sample_fusi_3d, fwhm={"k": 0.3, "w": 0.2})

    def test_raises_chunked_spatial_dim(self, sample_fusi_3d):
        """Should raise ValueError if a smoothed spatial dim is Dask-chunked."""
        pytest.importorskip("dask.array")
        vol_dask = sample_fusi_3d.chunk({"k": 2})  # Spatial dim chunked.
        with pytest.raises(ValueError, match="is chunked"):
            smooth_volume(vol_dask, fwhm=0.3)
