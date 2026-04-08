"""Integration tests for EchoFrame DAT to Zarr conversion."""

import warnings
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
import zarr

from confusius.io.echoframe import convert_echoframe_dat_to_zarr


class TestEchoFrameConversion:
    """Integration tests for EchoFrame DAT conversion to Zarr."""

    def test_conversion(self, synthetic_echoframe_session, tmp_path):
        """Test all conversion parameters and verification.

        Tests:
        - Basic conversion with cleaned DAT file.
        - Data values verification (deterministic data pattern).
        - Coordinate values (x, z, y, time).
        - Coordinate attributes (units, long_name).
        - All metadata attributes from MAT file.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 6
            assert ds["iq"].dtype == np.complex64

            assert np.all(np.isfinite(ds["iq"].values))

            iq = ds["iq"].values
            block1_volume1_value = 1 + 1j
            block2_volume1_value = 101 + 101j

            np.testing.assert_allclose(iq[0, 0, 0, 0], block1_volume1_value)
            np.testing.assert_allclose(iq[3, 0, 0, 0], block2_volume1_value)

            np.testing.assert_allclose(ds["time"].values, np.arange(6) / 1000.0)
            assert ds["time"].attrs["units"] == "s"
            assert ds["time"].attrs["long_name"] == "Time"
            assert ds["time"].attrs["volume_acquisition_reference"] == "start"
            assert ds["time"].attrs["volume_acquisition_duration"] == pytest.approx(
                5 / 5000.0
            )

            np.testing.assert_allclose(ds["x"].values, np.linspace(0, 0.4, 4))
            assert ds["x"].attrs["units"] == "mm"
            assert ds["x"].attrs["long_name"] == "Lateral"

            np.testing.assert_allclose(ds["z"].values, np.array([0.0]))
            assert ds["z"].attrs["units"] == "mm"
            assert ds["z"].attrs["long_name"] == "Elevation"

            np.testing.assert_allclose(ds["y"].values, np.linspace(0, 0.3, 6))
            assert ds["y"].attrs["units"] == "mm"
            assert ds["y"].attrs["long_name"] == "Depth"

            # Verify voxdim is stored as per-coordinate attribute.
            assert ds["z"].attrs["voxdim"] == pytest.approx(0.4)
            assert ds["y"].attrs["voxdim"] == pytest.approx(0.06)
            assert ds["x"].attrs["voxdim"] == pytest.approx(0.13333333333333333)
            assert ds["iq"].attrs["transmit_frequency"] == 15.625e6
            assert ds["iq"].attrs["probe_number_of_elements"] == 128
            assert ds["iq"].attrs["probe_pitch"] == pytest.approx(0.3)
            assert ds["iq"].attrs["beamforming_sound_velocity"] == 1540.0
            assert ds["iq"].attrs["plane_wave_angles"] == [-10.0, -5.0, 0.0, 5.0, 10.0]
            assert ds["iq"].attrs["compound_sampling_frequency"] == 1000.0
            assert ds["iq"].attrs["beamforming_method"] == "DAS"

    def test_with_padding(self, synthetic_echoframe_session_with_padding, tmp_path):
        """Test conversion of DAT file with padding (uncleaned data).

        Verifies that padding bytes are correctly removed and output shape is correct.
        """
        dat_path = synthetic_echoframe_session_with_padding / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session_with_padding / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 6
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

    def test_with_cropping(self, synthetic_echoframe_session_cropped, tmp_path):
        """Test conversion with cropping metadata.

        Verifies that cropping ROI is correctly applied and output shape reflects
        the cropped dimensions.
        """
        dat_path = synthetic_echoframe_session_cropped / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session_cropped / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 3
            assert ds["iq"].shape[1] == 1  # z is the stacking dimension
            assert ds["iq"].shape[2] == 4  # y is the depth dimension
            assert ds["iq"].shape[3] == 3
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

    def test_with_block_times(self, synthetic_echoframe_session, tmp_path):
        """Test conversion with block times provided by user.

        Verifies that time coordinates are correctly computed from block times.
        Users should provide block times based on their specific rig setup.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        # Block times that would be extracted by rig-specific utilities.
        block_times = np.array([1.0, 3.0])

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            block_times=block_times,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert "time" in ds.coords

            assert ds["iq"].shape[0] == 6
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

            expected_times = np.array([1.0, 1.001, 1.002, 3.0, 3.001, 3.002])
            np.testing.assert_allclose(ds["time"].values, expected_times, rtol=1e-5)

    def test_overwrite_existing(self, synthetic_echoframe_session, tmp_path):
        """Test overwriting existing Zarr output."""
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            show_progress=False,
        )

        first_shape = cast(
            zarr.Array, zarr.open_group(output_path, mode="r")["iq"]
        ).shape

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            overwrite=True,
            show_progress=False,
        )

        second_shape = cast(
            zarr.Array, zarr.open_group(output_path, mode="r")["iq"]
        ).shape
        assert first_shape == second_shape

    def test_volumes_per_shard_not_multiple(
        self, synthetic_echoframe_session, tmp_path
    ):
        """Raise ValueError when volumes_per_shard is not a multiple of volumes_per_chunk."""
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        with pytest.raises(
            ValueError,
            match="volumes_per_shard.*must be a multiple of.*volumes_per_chunk",
        ):
            convert_echoframe_dat_to_zarr(
                dat_path,
                meta_path,
                output_path,
                volumes_per_chunk=3,
                volumes_per_shard=5,
                show_progress=False,
            )

    def test_zarr_kwargs_override_warning(
        self, synthetic_echoframe_session, tmp_path, caplog
    ):
        """Warn when zarr_kwargs contains keys that are handled by function parameters."""
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            convert_echoframe_dat_to_zarr(
                dat_path,
                meta_path,
                output_path,
                zarr_kwargs={"shape": (10, 6, 1, 4)},
                show_progress=False,
            )

            assert any(
                "zarr_kwargs contains keys that are handled" in str(warning)
                for warning in w
            )

    def test_with_sharding(self, synthetic_echoframe_session, tmp_path):
        """Test conversion with Zarr sharding enabled.

        Verifies that sharding configuration produces valid output with reduced file count.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            volumes_per_chunk=3,
            volumes_per_shard=6,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 6
            assert ds["iq"].dtype == np.complex64
            assert np.all(np.isfinite(ds["iq"].values))

        zarr_group = zarr.open_group(output_path, mode="r")
        iq_array = zarr_group["iq"]
        assert iq_array.shards == (6, 1, 6, 4)

    def test_cleanup_on_error(self, synthetic_echoframe_session, tmp_path):
        """Test that incomplete zarr store is cleaned up on error.

        Verifies that if an exception occurs during conversion, the incomplete
        zarr store is removed to avoid leaving corrupted data on disk.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
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

        with patch(
            "confusius.io.echoframe.zarr.open_group", side_effect=mock_open_group
        ):
            with pytest.raises(Exception, match="Simulated error during conversion"):
                convert_echoframe_dat_to_zarr(
                    dat_path,
                    meta_path,
                    output_path,
                    show_progress=False,
                )

        assert not output_path.exists()

    def test_skip_first_blocks(self, synthetic_echoframe_session, tmp_path):
        """Test skipping the first blocks during conversion.

        Verifies that skip_first_blocks correctly removes blocks from the beginning
        and that time coordinates are properly adjusted.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            skip_first_blocks=1,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            # Original: 2 blocks * 3 volumes = 6, after skipping 1 block = 3 volumes.
            assert ds["iq"].shape[0] == 3

            # Data should start from block 2 (values 101+101j).
            iq = ds["iq"].values
            np.testing.assert_allclose(iq[0, 0, 0, 0], 101 + 101j)

            # Time should still be continuous starting from 3/1000.
            expected_times = np.array([3.0, 4.0, 5.0]) / 1000.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_last_blocks(self, synthetic_echoframe_session, tmp_path):
        """Test skipping the last blocks during conversion.

        Verifies that skip_last_blocks correctly removes blocks from the end
        and that time coordinates are properly adjusted.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            skip_last_blocks=1,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            # Original: 2 blocks * 3 volumes = 6, after skipping 1 block = 3 volumes.
            assert ds["iq"].shape[0] == 3

            # Data should be from block 1 only (values 1+1j).
            iq = ds["iq"].values
            np.testing.assert_allclose(iq[0, 0, 0, 0], 1 + 1j)

            # Time should be normal starting from 0.
            expected_times = np.arange(3) / 1000.0
            np.testing.assert_allclose(ds["time"].values, expected_times)

    def test_skip_both_first_and_last(self, synthetic_echoframe_session, tmp_path):
        """Test skipping both first and last blocks.

        Verifies that both parameters work together correctly.
        """
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        # This would skip all blocks, so we need a session with more blocks.
        # With 2 blocks total, skipping 1 first and 0 last to test the logic.
        convert_echoframe_dat_to_zarr(
            dat_path,
            meta_path,
            output_path,
            skip_first_blocks=1,
            skip_last_blocks=0,
            show_progress=False,
        )

        assert output_path.exists()

        with xr.open_zarr(output_path) as ds:
            assert "iq" in ds
            assert ds["iq"].shape[0] == 3  # 1 block remaining * 3 volumes.

    def test_skip_blocks_error(self, synthetic_echoframe_session, tmp_path):
        """Test error when trying to skip too many blocks."""
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        with pytest.raises(ValueError, match="Cannot skip"):
            convert_echoframe_dat_to_zarr(
                dat_path,
                meta_path,
                output_path,
                skip_first_blocks=1,
                skip_last_blocks=1,  # Would skip all 2 blocks.
                show_progress=False,
            )

    def test_block_times_length_mismatch(self, synthetic_echoframe_session, tmp_path):
        """Raise ValueError when block_times length does not match number of blocks."""
        dat_path = synthetic_echoframe_session / "fUSi_BF.dat"
        meta_path = synthetic_echoframe_session / "ScanParameters.mat"
        output_path = tmp_path / "output.zarr"

        # Session has 2 blocks; providing 3 block times should raise.
        with pytest.raises(ValueError, match="block_times length"):
            convert_echoframe_dat_to_zarr(
                dat_path,
                meta_path,
                output_path,
                block_times=np.array([1.0, 2.0, 3.0]),
                show_progress=False,
            )
