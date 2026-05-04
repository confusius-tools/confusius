"""Unit tests for volumewise registration functions."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from confusius.registration.volumewise import register_volumewise


class TestRegisterVolumewise:
    """Tests for register_volumewise function."""

    def test_missing_time_dimension_raises(self):
        """Data without 'time' dimension raises ValueError."""
        data = xr.DataArray(np.zeros((10, 10)), dims=("y", "x"))
        with pytest.raises(ValueError, match="Time dimension 'time' not found"):
            register_volumewise(data)

    def test_h5py_backed_raises_with_parallel_jobs(self, scan_2d):
        """h5py-backed DataArray (from a .scan file) raises TypeError when n_jobs != 1."""
        with pytest.raises(TypeError, match="h5py dataset"):
            register_volumewise(scan_2d, n_jobs=2)

    def test_h5py_backed_works_with_n_jobs_1(self, scan_2d):
        """h5py-backed DataArray (from a .scan file) with n_jobs=1 does not raise."""
        # n_jobs=1 (serial) should not raise for h5py-backed data.
        result = register_volumewise(scan_2d, n_jobs=1, transform="translation")
        assert result.shape == scan_2d.shape

    def test_wrong_dimensionality_raises(self):
        """Data that is neither 2D+t nor 3D+t raises ValueError."""
        # 1D+time = 2D total.
        data = xr.DataArray(np.zeros((5, 10)), dims=("time", "x"))
        with pytest.raises(ValueError, match="Expected 3D or 4D data"):
            register_volumewise(data)

    @pytest.mark.parametrize(
        ("data_fixture", "dims"),
        [
            ("sample_2d_dataarray", ("time", "y", "x")),
            ("sample_3d_dataarray", ("time", "z", "y", "x")),
        ],
    )
    def test_identical_frames_unchanged(self, data_fixture, dims, request):
        """Identical frames remain unchanged after registration (2D and 3D)."""
        data = request.getfixturevalue(data_fixture)
        result = register_volumewise(data, n_jobs=1, transform="translation")

        assert result.dims == dims
        assert result.shape == data.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_2d_recovers_known_shift(self, sample_2d_image):
        """2D registration recovers a known translation."""
        # Create data with a shifted frame.
        n_frames = 3
        shift_x, shift_y = 2, 3

        frames = [sample_2d_image.copy() for _ in range(n_frames)]
        # Shift frame 1 by rolling (simulates translation).
        frames[1] = np.roll(np.roll(frames[1], shift_y, axis=0), shift_x, axis=1)

        data = xr.DataArray(
            np.stack(frames, axis=0),
            dims=("time", "y", "x"),
            coords={
                "time": np.arange(n_frames) * 0.1,
                "y": np.arange(32) * 1.0,  # 1mm spacing.
                "x": np.arange(32) * 1.0,
            },
        )

        result = register_volumewise(
            data, reference_time=0, n_jobs=1, transform="translation"
        )

        # Check motion parameters recovered the shift.
        motion_df = result.attrs["motion_params"]
        # Frame 1 should have approximately the opposite translation.
        assert abs(motion_df.loc[motion_df.index[1], "trans_x"]) < shift_x + 1
        assert abs(motion_df.loc[motion_df.index[1], "trans_y"]) < shift_y + 1

    def test_output_has_motion_metadata_attributes(self, sample_2d_dataarray):
        """Output has motion metadata attributes."""
        result = register_volumewise(sample_2d_dataarray, reference_time=2, n_jobs=1)

        assert "registration" not in result.attrs
        assert result.attrs["reference_time"] == 2
        assert "motion_params" in result.attrs

    def test_preserves_input_attributes(self, sample_2d_dataarray):
        """Input attributes are preserved in output."""
        sample_2d_dataarray.attrs["custom_attr"] = "test_value"

        result = register_volumewise(sample_2d_dataarray, n_jobs=1)

        assert result.attrs["custom_attr"] == "test_value"

    def test_preserves_coordinates(self, sample_2d_dataarray):
        """Coordinates are preserved in output."""
        result = register_volumewise(sample_2d_dataarray, n_jobs=1)

        assert_allclose(
            result.coords["time"].values, sample_2d_dataarray.coords["time"].values
        )
        assert_allclose(
            result.coords["y"].values, sample_2d_dataarray.coords["y"].values
        )
        assert_allclose(
            result.coords["x"].values, sample_2d_dataarray.coords["x"].values
        )

    def test_different_reference_time(self, sample_2d_dataarray):
        """Can use different reference time indices."""
        result = register_volumewise(sample_2d_dataarray, reference_time=2, n_jobs=1)

        assert result.attrs["reference_time"] == 2

    def test_transform_option(self, sample_2d_dataarray):
        """transform parameter changes registration behavior."""
        # Both should work without error.
        result_no_rot = register_volumewise(
            sample_2d_dataarray, n_jobs=1, transform="translation"
        )
        result_with_rot = register_volumewise(
            sample_2d_dataarray, n_jobs=1, transform="rigid"
        )

        # Motion params should have rotation column in both cases.
        assert "rotation" in result_no_rot.attrs["motion_params"].columns
        assert "rotation" in result_with_rot.attrs["motion_params"].columns

    def test_singleton_dimension_handling(self, sample_2d_image):
        """Singleton spatial dimensions are handled correctly."""
        # Create data with a singleton z dimension (2D slice in 3D array).
        # The voxdim attribute provides spacing for the singleton z coordinate so
        # that no "spacing is undefined" warning is raised.
        z_coord = xr.DataArray([0.0], dims=("z",), attrs={"voxdim": 0.2})
        data = xr.DataArray(
            sample_2d_image[np.newaxis, np.newaxis, :, :].repeat(3, axis=0),
            dims=("time", "z", "y", "x"),
            coords={
                "time": np.arange(3) * 0.1,
                "z": z_coord,
                "y": np.arange(32) * 0.1,
                "x": np.arange(32) * 0.1,
            },
        )

        result = register_volumewise(data, n_jobs=1)

        # Should preserve the singleton dimension.
        assert result.dims == data.dims
        assert result.shape == data.shape
        assert result.sizes["z"] == 1
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_output_dimension_order_matches_input(self, sample_2d_image):
        """Output dimension order matches input regardless of internal transposition."""
        # Create data with non-standard dimension order.
        data = xr.DataArray(
            np.stack([sample_2d_image] * 3, axis=2),
            dims=("y", "x", "time"),
            coords={
                "y": np.arange(32) * 0.1,
                "x": np.arange(32) * 0.1,
                "time": np.arange(3) * 0.1,
            },
        )

        result = register_volumewise(data, n_jobs=1)

        assert result.dims == ("y", "x", "time")
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, data.values, atol=1e-3)

    def test_multi_resolution_does_not_crash(self, sample_3d_dataarray):
        """Multi-resolution pyramid completes without error."""
        result = register_volumewise(
            sample_3d_dataarray,
            n_jobs=1,
            transform="translation",
            use_multi_resolution=True,
        )
        assert result.shape == sample_3d_dataarray.shape
        # Identical frames should produce nearly identical output.
        assert_allclose(result.values, sample_3d_dataarray.values, atol=1e-3)
