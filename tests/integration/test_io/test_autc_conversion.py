"""Integration tests for AUTCDAT to Zarr conversion."""

import warnings
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
import zarr

from confusius.io.autc import convert_autc_dats_to_zarr


class TestAUTCConversion:
    """Integration tests for AUTCDAT conversion to Zarr."""

    def test_conversion(self, synthetic_autc_session, tmp_path):
        """Test all conversion parameters and verification.

        Tests:
        - Conversion of multiple DAT files.
        - Custom x_axis and z_axis coordinates.
        - Volume sampling frequency.
        - Block times provided by user.
        - Probe metadata parameters.
        - Output verification using Xarray.
        """
        output_path = tmp_path / "output.zarr"

        custom_lateral_coords = np.linspace(-10, 10, 4)
        custom_axial_coords = np.linspace(0, 20, 6)

        # Block times that would be extracted by rig-specific utilities.
        block_times = np.array([1.0, 3.0, 5.0, 7.0])

        convert_autc_dats_to_zarr(
            synthetic_autc_session,
            output_path,
            lateral_coords=custom_lateral_coords,
            axial_coords=custom_axial_coords,
            transmit_frequency=3000000.0,
            probe_n_elements=64,
            probe_pitch=0.00025,
            beamforming_sound_velocity=1480.0,
            plane_wave_angles=np.array([-15.0, 0.0, 15.0]),
            compound_sampling_frequency=500.0,
            pulse_repetition_frequency=1500.0,
            beamforming_method="Fourier",
            block_times=block_times,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 12
            assert ds["iq"].dtype == np.complex64

            assert np.all(np.isfinite(ds["iq"].values))

            iq = ds["iq"].values
            block1_frame1_value = 1 + 1j
            block2_frame1_value = 101 + 101j

            np.testing.assert_allclose(iq[0, 0, 0, 0], block1_frame1_value)
            np.testing.assert_allclose(iq[3, 0, 0, 0], block2_frame1_value)

            np.testing.assert_allclose(ds["x"].values, custom_lateral_coords)
            np.testing.assert_allclose(ds["z"].values, np.array([0.0]))
            np.testing.assert_allclose(ds["y"].values, custom_axial_coords)

            expected_times = np.array(
                [
                    1.0,
                    1.002,
                    1.004,
                    3.0,
                    3.002,
                    3.004,
                    5.0,
                    5.002,
                    5.004,
                    7.0,
                    7.002,
                    7.004,
                ]
            )
            np.testing.assert_allclose(ds["time"].values, expected_times, rtol=1e-5)

            assert ds["time"].attrs["units"] == "s"
            assert ds["time"].attrs["long_name"] == "Time"
            assert ds["time"].attrs["volume_acquisition_reference"] == "start"
            assert ds["time"].attrs["volume_acquisition_duration"] == pytest.approx(
                3 / 1500.0
            )

            assert ds["z"].attrs["units"] == "mm"
            assert ds["z"].attrs["long_name"] == "Elevation"

            assert ds["y"].attrs["units"] == "mm"
            assert ds["y"].attrs["long_name"] == "Depth"

            assert ds["x"].attrs["units"] == "mm"
            assert ds["x"].attrs["long_name"] == "Lateral"

            # Verify voxdim is stored as per-coordinate attribute.
            assert ds["z"].attrs["voxdim"] == pytest.approx(0.4)
            assert ds["y"].attrs["voxdim"] == pytest.approx(4.0)
            assert ds["x"].attrs["voxdim"] == pytest.approx(20.0 / 3.0)
            assert ds["iq"].attrs["transmit_frequency"] == 3000000.0
            assert ds["iq"].attrs["probe_number_of_elements"] == 64
            assert ds["iq"].attrs["probe_pitch"] == 0.00025
            assert ds["iq"].attrs["beamforming_sound_velocity"] == 1480.0
            assert ds["iq"].attrs["plane_wave_angles"] == [-15.0, 0.0, 15.0]
            assert ds["iq"].attrs["compound_sampling_frequency"] == 500.0
            assert ds["iq"].attrs["pulse_repetition_frequency"] == 1500.0
            assert ds["iq"].attrs["beamforming_method"] == "Fourier"

    def test_overwrite_existing(self, synthetic_autc_session, tmp_path):
        """Test overwriting existing Zarr output."""
        output_path = tmp_path / "output.zarr"

        with pytest.warns(UserWarning, match="coords not provided"):
            convert_autc_dats_to_zarr(
                synthetic_autc_session,
                output_path,
                show_progress=False,
            )

        first_shape = cast(
            zarr.Array, zarr.open_group(output_path, mode="r")["iq"]
        ).shape

        with pytest.warns(UserWarning, match="coords not provided"):
            convert_autc_dats_to_zarr(
                synthetic_autc_session,
                output_path,
                overwrite=True,
                show_progress=False,
            )

        second_shape = cast(
            zarr.Array, zarr.open_group(output_path, mode="r")["iq"]
        ).shape
        assert first_shape == second_shape

    def test_frames_per_shard_not_multiple(self, synthetic_autc_session, tmp_path):
        """Raise `ValueError` when `frames_per_shard` is not a multiple of `frames_per_chunk`."""
        output_path = tmp_path / "output.zarr"

        with pytest.raises(
            ValueError,
            match="frames_per_shard.*must be a multiple of.*frames_per_chunk",
        ):
            convert_autc_dats_to_zarr(
                synthetic_autc_session,
                output_path,
                frames_per_chunk=3,
                frames_per_shard=5,
                show_progress=False,
            )

    def test_block_times_without_frequency_error(
        self, synthetic_autc_session, tmp_path
    ):
        """Raise `ValueError` when `block_times` provided without `compound_sampling_frequency`."""
        output_path = tmp_path / "output.zarr"

        with pytest.raises(
            ValueError,
            match="compound_sampling_frequency is required when block_times is provided",
        ):
            convert_autc_dats_to_zarr(
                synthetic_autc_session,
                output_path,
                block_times=[0.0, 1.0, 2.0],
                show_progress=False,
            )

    def test_zarr_kwargs_override_warning(
        self, synthetic_autc_session, tmp_path, caplog
    ):
        """Warn when `zarr_kwargs` contains keys that are handled by function parameters."""
        output_path = tmp_path / "output.zarr"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_autc_dats_to_zarr(
                synthetic_autc_session,
                output_path,
                zarr_kwargs={"shape": (10, 6, 1, 4)},
                show_progress=False,
            )

            assert any(
                "zarr_kwargs contains keys that are handled" in str(warning)
                for warning in w
            )

    def test_with_sharding(self, synthetic_autc_session, tmp_path):
        """Test conversion with Zarr sharding enabled.

        Verifies that sharding configuration produces valid output and that shard files
        are created on disk.
        """
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            synthetic_autc_session,
            output_path,
            lateral_coords=np.linspace(-10, 10, 4),
            axial_coords=np.linspace(0, 20, 6),
            frames_per_chunk=3,
            frames_per_shard=6,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 12
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

        zarr_group = zarr.open_group(output_path, mode="r")
        iq_array = zarr_group["iq"]
        assert isinstance(iq_array, zarr.Array)
        assert iq_array.shards == (6, 1, 6, 4)

    def test_cleanup_on_error(self, synthetic_autc_session, tmp_path):
        """Test that incomplete zarr store is cleaned up on error.

        Verifies that if an exception occurs during conversion, the incomplete
        zarr store is removed to avoid leaving corrupted data on disk.
        """
        output_path = tmp_path / "output.zarr"

        mock_array = MagicMock()
        mock_array.__setitem__ = MagicMock(
            side_effect=Exception("Simulated error during conversion")
        )

        class MockZarrGroup:
            def __init__(self, real_group):
                self._real_group = real_group

            def create_array(self, name, **kwargs):
                if name == "iq":
                    return mock_array
                return self._real_group.create_array(name, **kwargs)

            def __getitem__(self, key):
                return self._real_group[key]

            def __getattr__(self, name):
                return getattr(self._real_group, name)

        original_open_group = zarr.open_group

        def mock_open_group(*args, **kwargs):
            real_group = original_open_group(*args, **kwargs)
            return MockZarrGroup(real_group)

        with patch("confusius.io.autc.zarr.open_group", side_effect=mock_open_group):
            with pytest.raises(Exception, match="Simulated error during conversion"):
                convert_autc_dats_to_zarr(
                    synthetic_autc_session,
                    output_path,
                    lateral_coords=np.linspace(-10, 10, 4),
                    axial_coords=np.linspace(0, 20, 6),
                    show_progress=False,
                )

        assert not output_path.exists()

    def test_time_from_sampling_frequency(self, synthetic_autc_session, tmp_path):
        """Test time coordinate computed from compound_sampling_frequency (no block_times).

        When block_times is not provided, time should be computed as:
        time[i] = i / compound_sampling_frequency
        """
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            synthetic_autc_session,
            output_path,
            lateral_coords=np.linspace(-10, 10, 4),
            axial_coords=np.linspace(0, 20, 6),
            compound_sampling_frequency=1000.0,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "time" in ds.coords

            expected_times = np.arange(12) / 1000.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_first_blocks(self, synthetic_autc_session, tmp_path):
        """Test skipping the first blocks during conversion.

        Verifies that skip_first_blocks correctly removes blocks from the beginning
        and that time coordinates are properly adjusted.
        """
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            synthetic_autc_session,
            output_path,
            lateral_coords=np.linspace(-10, 10, 4),
            axial_coords=np.linspace(0, 20, 6),
            compound_sampling_frequency=500.0,
            skip_first_blocks=1,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            # Original: 4 blocks * 3 frames = 12, after skipping 1 block = 9 frames.
            assert ds["iq"].shape[0] == 9

            # Data should start from block 2 (values 101+101j).
            iq = ds["iq"].values
            np.testing.assert_allclose(iq[0, 0, 0, 0], 101 + 101j)

            # Time should still be continuous (frames 3-11).
            expected_times = np.arange(3, 12) / 500.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_last_blocks(self, synthetic_autc_session, tmp_path):
        """Test skipping the last blocks during conversion.

        Verifies that skip_last_blocks correctly removes blocks from the end
        and that time coordinates are properly adjusted.
        """
        output_path = tmp_path / "output.zarr"

        convert_autc_dats_to_zarr(
            synthetic_autc_session,
            output_path,
            lateral_coords=np.linspace(-10, 10, 4),
            axial_coords=np.linspace(0, 20, 6),
            compound_sampling_frequency=500.0,
            skip_last_blocks=1,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            # Original: 4 blocks * 3 frames = 12, after skipping 1 block = 9 frames.
            assert ds["iq"].shape[0] == 9

            # Data should be from first 3 blocks (values 1+1j).
            iq = ds["iq"].values
            np.testing.assert_allclose(iq[0, 0, 0, 0], 1 + 1j)

            # Time should be normal starting from 0.
            expected_times = np.arange(9) / 500.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_both_first_and_last(self, synthetic_autc_session, tmp_path):
        """Test skipping both first and last blocks with block_times."""
        output_path = tmp_path / "output.zarr"

        # block_times for retained blocks only (2 middle blocks).
        block_times = np.array([1.0, 2.0])

        convert_autc_dats_to_zarr(
            synthetic_autc_session,
            output_path,
            lateral_coords=np.linspace(-10, 10, 4),
            axial_coords=np.linspace(0, 20, 6),
            compound_sampling_frequency=500.0,
            block_times=block_times,
            skip_first_blocks=1,
            skip_last_blocks=1,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            # Original: 4 blocks * 3 frames = 12, after skipping 2 blocks = 6 frames.
            assert ds["iq"].shape[0] == 6

            # Time should use the provided block_times directly.
            expected_times = np.array([1.0, 1.002, 1.004, 2.0, 2.002, 2.004])
            np.testing.assert_allclose(ds["time"].values, expected_times, rtol=1e-5)

    def test_skip_blocks_error(self, synthetic_autc_session, tmp_path):
        """Test error when trying to skip too many blocks."""
        output_path = tmp_path / "output.zarr"

        with pytest.raises(ValueError, match="Cannot skip"):
            convert_autc_dats_to_zarr(
                synthetic_autc_session,
                output_path,
                skip_first_blocks=2,
                skip_last_blocks=2,  # Would skip all 4 blocks.
                show_progress=False,
            )
