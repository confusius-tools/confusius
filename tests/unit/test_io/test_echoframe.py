"""Unit tests for confusius.io.echoframe module."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from confusius.io.echoframe import (
    load_echoframe_dat,
    load_echoframe_metadata,
    load_echoframe_rf_timetag,
)


def _create_echoframe_metadata(
    path: Path,
    n_x: int = 4,
    n_z: int = 6,
    n_volumes: int = 3,
    crop: bool = False,
    cropping_roi: list | None = None,
    transmit_receive_time_mus: float = 200.0,
) -> None:
    """Create a synthetic EchoFrame MAT metadata file.

    Parameters
    ----------
    path : Path
        Path to create the MAT file.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of depth samples.
    n_volumes : int, default: 3
        Number of volumes per block.
    crop : bool, default: False
        Whether cropping is enabled.
    cropping_roi : list, optional
        Cropping ROI [z_start, z_end, x_start, x_end] (1-indexed).
        Used only if crop=True.
    transmit_receive_time_mus : float, default: 200.0
        Transmit-receive period in microseconds.
    """
    with h5py.File(path, "w") as f:
        recon_spec = f.create_group("ReconSpec")
        receive_spec = f.create_group("ReceiveSpec")
        probe_spec = f.create_group("ProbeSpec")
        transmit_spec = f.create_group("TransmitSpec")

        recon_spec["cropBF"] = np.array([crop], dtype=np.uint8)

        if crop and cropping_roi is not None:
            recon_spec["croppingROI"] = np.array([cropping_roi], dtype=np.float64)
            z = int(cropping_roi[1] - cropping_roi[0] + 1)
            x = int(cropping_roi[3] - cropping_roi[2] + 1)
        else:
            z = n_z
            x = n_x
            recon_spec["nz"] = np.array([[z]], dtype=np.int32)
            recon_spec["nx"] = np.array([[x]], dtype=np.int32)

        receive_spec["nRepeats"] = np.array([n_volumes], dtype=np.int32)
        receive_spec["transmitReceiveTimeMus"] = np.array([transmit_receive_time_mus])

        recon_spec["x_axis"] = np.linspace(0, x * 0.1, x)
        recon_spec["z_axis"] = np.linspace(0, z * 0.05, z)
        recon_spec["c0"] = np.array([1540.0])
        recon_spec["method"] = np.array([ord(c) for c in "DAS"], dtype=np.uint8)
        probe_spec["Fc"] = np.array([15.625e6])
        probe_spec["nElementsX"] = np.array([128])
        probe_spec["pitchX"] = np.array([0.0003])
        transmit_spec["steerX"] = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])


def _create_echoframe_dat_file(
    path: Path,
    n_blocks: int = 2,
    n_x: int = 4,
    n_z: int = 6,
    n_volumes: int = 3,
    padding_bytes: int = 0,
    known_pattern: bool = True,
) -> None:
    """Create a synthetic EchoFrame DAT file.

    Parameters
    ----------
    path : Path
        Path to create the DAT file.
    n_blocks : int, default: 2
        Number of acquisition blocks.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of depth samples.
    n_volumes : int, default: 3
        Number of volumes per block.
    padding_bytes : int, default: 0
        Padding bytes per block (0 for cleaned files).
    known_pattern : bool, default: True
        If True, use known pattern for verification. If False, use random data.
    """
    header_dtype = np.uint64
    n_header_items = 5
    data_size = n_x * n_volumes * n_z  # Elements per block (excluding y=1).

    with open(path, "wb") as f:
        # Write header
        header = np.array(
            [
                1,  # version
                n_header_items * 8,  # header_size in bytes (5 * 8 = 40)
                n_blocks,  # n_buffers
                data_size,  # effective_buffer_size
                padding_bytes,  # padding_bytes
            ],
            dtype=header_dtype,
        )
        header.tofile(f)

        for block_idx in range(n_blocks):
            if known_pattern:
                data = np.full(
                    (n_volumes, n_x, 1, n_z),
                    complex(block_idx + 1.0, 1.0),
                    dtype=np.complex64,
                )
            else:
                data = np.random.randn(n_volumes, n_x, 1, n_z).astype(np.complex64)
            data.tofile(f)

            if padding_bytes > 0:
                padding = np.zeros(padding_bytes, dtype=np.uint8)
                padding.tofile(f)


@pytest.fixture
def echoframe_meta_file(tmp_path):
    """Create a synthetic EchoFrame MAT metadata file."""
    meta_path = tmp_path / "ScanParameters.mat"
    _create_echoframe_metadata(meta_path, n_x=4, n_z=6, n_volumes=3)
    return meta_path


@pytest.fixture
def echoframe_dat_no_padding(tmp_path, echoframe_meta_file):
    """Create a synthetic EchoFrame DAT file without padding (cleaned)."""
    dat_path = tmp_path / "fUSi_BF_cleaned.dat"
    _create_echoframe_dat_file(
        dat_path, n_blocks=2, n_x=4, n_z=6, n_volumes=3, padding_bytes=0
    )
    return dat_path, echoframe_meta_file


@pytest.fixture
def echoframe_dat_with_padding(tmp_path, echoframe_meta_file):
    """Create a synthetic EchoFrame DAT file with padding (uncleaned)."""
    dat_path = tmp_path / "fUSi_BF.dat"
    padding_bytes = 16
    _create_echoframe_dat_file(
        dat_path,
        n_blocks=2,
        n_x=4,
        n_z=6,
        n_volumes=3,
        padding_bytes=padding_bytes,
    )
    return dat_path, echoframe_meta_file


@pytest.fixture
def echoframe_dat_cropped(tmp_path):
    """Create synthetic EchoFrame files with cropping enabled."""
    meta_path = tmp_path / "ScanParameters.mat"
    # Cropping ROI: [z_start=1, z_end=4, x_start=1, x_end=3]
    # This gives z=4, x=3 (smaller than full 6x4).
    _create_echoframe_metadata(
        meta_path, n_x=6, n_z=6, n_volumes=3, crop=True, cropping_roi=[1, 4, 1, 3]
    )

    dat_path = tmp_path / "fUSi_BF.dat"
    _create_echoframe_dat_file(
        dat_path, n_blocks=1, n_x=6, n_z=6, n_volumes=3, padding_bytes=0
    )
    return dat_path, meta_path


class TestLoadEchoFrameDat:
    """Tests for load_echoframe_dat function."""

    def test_loads_file_without_padding(self, echoframe_dat_no_padding):
        """load_echoframe_dat loads files without padding correctly."""
        dat_path, meta_path = echoframe_dat_no_padding

        data = load_echoframe_dat(dat_path, meta_path)

        assert data.shape == (2, 3, 4, 1, 6)  # (n_blocks, n_volumes, n_x, 1, n_z)
        assert data.dtype == np.complex64

        # Verify values match the known pattern (block_idx+1, 1).
        assert data[0, 0, 0, 0, 0] == complex(1.0, 1.0), "Block 0 value corrupted"
        assert data[1, 0, 0, 0, 0] == complex(2.0, 1.0), "Block 1 value corrupted"

    def test_loads_file_with_padding(self, echoframe_dat_with_padding):
        """`load_echoframe_dat` loads files with padding correctly."""
        dat_path, meta_path = echoframe_dat_with_padding

        data = load_echoframe_dat(dat_path, meta_path)

        assert data.shape == (2, 3, 4, 1, 6)
        assert data.dtype == np.complex64

        # Verify all blocks have the expected pattern.
        # Block 0 should have real part = 1.0, Block 1 should have real part = 2.0.
        # If padding is not handled correctly, Block 1 will be corrupted.
        expected_block_0 = complex(1.0, 1.0)
        expected_block_1 = complex(2.0, 1.0)

        assert data[0, 0, 0, 0, 0] == expected_block_0, "Block 0 is corrupted"
        assert data[1, 0, 0, 0, 0] == expected_block_1, (
            f"Block 1 is corrupted! Got {data[1, 0, 0, 0, 0]}, expected {expected_block_1}. "
            "This may indicate that padding bytes are not being skipped correctly."
        )

    def test_loads_file_with_cropping(self, echoframe_dat_cropped):
        """`load_echoframe_dat` respects cropping ROI from metadata."""
        dat_path, meta_path = echoframe_dat_cropped

        data = load_echoframe_dat(dat_path, meta_path)

        # With cropping ROI [1, 4, 1, 3]: z=4, x=3.
        assert data.shape == (1, 3, 3, 1, 4)

    def test_missing_dat_file(self, tmp_path, echoframe_meta_file):
        """`load_echoframe_dat` raises `ValueError` for missing DAT file."""
        non_existent = tmp_path / "nonexistent.dat"

        with pytest.raises(ValueError):
            load_echoframe_dat(non_existent, echoframe_meta_file)

    def test_missing_mat_file(self, echoframe_dat_no_padding):
        """`load_echoframe_dat` raises ValueError for missing MAT file."""
        dat_path, _ = echoframe_dat_no_padding
        non_existent = Path("/tmp/nonexistent.mat")

        with pytest.raises(ValueError):
            load_echoframe_dat(dat_path, non_existent)

    def test_data_accessible(self, echoframe_dat_no_padding):
        """`load_echoframe_dat` returns accessible memmap with correct values."""
        dat_path, meta_path = echoframe_dat_no_padding

        data = load_echoframe_dat(dat_path, meta_path)

        # Block 0 is filled with complex(1, 1) in the known pattern.
        sample = data[0, 0, 0, 0, 0]
        assert sample == complex(1.0, 1.0)

    def test_different_blocks_accessible(self, echoframe_dat_no_padding):
        """`load_echoframe_dat` allows accessing different blocks with correct values."""
        dat_path, meta_path = echoframe_dat_no_padding

        data = load_echoframe_dat(dat_path, meta_path)

        first_block = data[0]
        last_block = data[-1]

        assert first_block.shape == (3, 4, 1, 6)
        assert last_block.shape == (3, 4, 1, 6)
        # Known pattern: block 0 → complex(1, 1), block 1 → complex(2, 1).
        assert first_block[0, 0, 0, 0] == complex(1.0, 1.0)
        assert last_block[0, 0, 0, 0] == complex(2.0, 1.0)


@pytest.fixture
def echoframe_meta_file_no_crop(tmp_path):
    """Create a synthetic EchoFrame metadata file without cropping."""
    meta_path = tmp_path / "ScanParameters.mat"
    _create_echoframe_metadata(
        meta_path,
        n_x=4,
        n_z=6,
        transmit_receive_time_mus=200.0,
    )
    return meta_path


@pytest.fixture
def echoframe_meta_file_crop(tmp_path):
    """Create a synthetic EchoFrame metadata file with cropping enabled."""
    meta_path = tmp_path / "ScanParameters.mat"
    # Full grid: 6×6; cropping ROI [z_start=1, z_end=4, x_start=1, x_end=3] (1-indexed)
    # gives z=4, x=3 after cropping.
    _create_echoframe_metadata(
        meta_path,
        n_x=6,
        n_z=6,
        crop=True,
        cropping_roi=[1, 4, 1, 3],
    )
    return meta_path


class TestLoadEchoFrameMetadata:
    """Tests for `load_echoframe_metadata` function."""

    def test_prf_derived_from_transmit_receive_time(self, echoframe_meta_file_no_crop):
        """`load_echoframe_metadata` converts transmit-receive period to PRF."""
        meta = load_echoframe_metadata(echoframe_meta_file_no_crop)

        # transmitReceiveTimeMus = 200 µs → PRF = 1 / (200e-6) = 5000 Hz.
        assert meta["pulse_repetition_frequency"] == pytest.approx(5000.0)

    def test_compound_sampling_frequency(self, echoframe_meta_file_no_crop):
        """`load_echoframe_metadata` derives CSF as PRF divided by number of angles."""
        meta = load_echoframe_metadata(echoframe_meta_file_no_crop)

        # 5 angles, PRF = 5000 Hz → CSF = 1000 Hz.
        assert meta["compound_sampling_frequency"] == pytest.approx(1000.0)

    def test_beamforming_method_decoded(self, echoframe_meta_file_no_crop):
        """`load_echoframe_metadata` decodes ASCII char codes to a string."""
        meta = load_echoframe_metadata(echoframe_meta_file_no_crop)

        assert meta["beamforming_method"] == "DAS"

    def test_cropping_applied_to_coords(self, echoframe_meta_file_crop):
        """`load_echoframe_metadata` applies the cropping ROI to spatial coordinates."""
        meta = load_echoframe_metadata(echoframe_meta_file_crop)

        # ROI [1, 4, 1, 3] (1-indexed) → z: indices 0–3 (4 points), x: indices 0–2 (3 points).
        assert meta["axial_coords"].size == 4
        assert meta["lateral_coords"].size == 3

    def test_missing_file(self, tmp_path):
        """`load_echoframe_metadata` raises `ValueError` for a missing file."""
        with pytest.raises(ValueError):
            load_echoframe_metadata(tmp_path / "nonexistent.mat")


def _create_rf_timetag_dat_file(
    path: Path,
    n_buffers: int = 3,
    buffer_size: int = 10,
    padding_bytes: int = 0,
    known_values: bool = True,
) -> None:
    """Create a synthetic EchoFrame RF time tag DAT file.

    Parameters
    ----------
    path : Path
        Path to create the DAT file.
    n_buffers : int, default: 3
        Number of acquisition buffers.
    buffer_size : int, default: 10
        Number of float64 time tag values per buffer.
    padding_bytes : int, default: 0
        Padding bytes appended after each buffer.
    known_values : bool, default: True
        If True, fill each buffer with its 1-based index as a float64 sentinel.
        If False, use random data.
    """
    header_dtype = np.uint64
    n_header_items = 5
    header_size = n_header_items * 8

    with open(path, "wb") as f:
        header = np.array(
            [1, header_size, n_buffers, buffer_size, padding_bytes],
            dtype=header_dtype,
        )
        header.tofile(f)

        rng = np.random.default_rng(42)
        for buf_idx in range(n_buffers):
            if known_values:
                data = np.full(buffer_size, float(buf_idx + 1), dtype=np.float64)
            else:
                data = rng.random(buffer_size)
            data.tofile(f)

            if padding_bytes > 0:
                np.zeros(padding_bytes, dtype=np.uint8).tofile(f)


@pytest.fixture
def rf_timetag_no_padding(tmp_path):
    """Create a synthetic RF time tag DAT file without padding."""
    path = tmp_path / "rf_timetag.dat"
    _create_rf_timetag_dat_file(path, n_buffers=3, buffer_size=10, padding_bytes=0)
    return path


@pytest.fixture
def rf_timetag_with_padding(tmp_path):
    """Create a synthetic RF time tag DAT file with padding."""
    path = tmp_path / "rf_timetag_padded.dat"
    _create_rf_timetag_dat_file(path, n_buffers=3, buffer_size=10, padding_bytes=16)
    return path


class TestLoadEchoFrameRfTimetag:
    """Tests for `load_echoframe_rf_timetag` function."""

    def test_loads_without_padding(self, rf_timetag_no_padding):
        """`load_echoframe_rf_timetag` returns all values for files without padding."""
        data = load_echoframe_rf_timetag(rf_timetag_no_padding)

        assert isinstance(data, np.memmap)
        assert data.shape == (30,)
        assert data.dtype == np.float64

    def test_known_values_without_padding(self, rf_timetag_no_padding):
        """`load_echoframe_rf_timetag` reads correct values from each buffer."""
        data = load_echoframe_rf_timetag(rf_timetag_no_padding)

        # Buffer 0 → sentinel 1.0, buffer 1 → 2.0, buffer 2 → 3.0.
        np.testing.assert_array_equal(data[:10], 1.0)
        np.testing.assert_array_equal(data[10:20], 2.0)
        np.testing.assert_array_equal(data[20:], 3.0)

    def test_loads_with_padding(self, rf_timetag_with_padding):
        """`load_echoframe_rf_timetag` returns all values for files with padding."""
        data = load_echoframe_rf_timetag(rf_timetag_with_padding)

        assert isinstance(data, np.memmap)
        assert data.shape == (30,)
        assert data.dtype == np.float64

    def test_known_values_with_padding(self, rf_timetag_with_padding):
        """`load_echoframe_rf_timetag` skips padding bytes and reads correct values."""
        data = load_echoframe_rf_timetag(rf_timetag_with_padding)

        # If padding is not skipped, buffer 1 onward will be corrupted.
        np.testing.assert_array_equal(data[:10], 1.0)
        np.testing.assert_array_equal(data[10:20], 2.0)
        np.testing.assert_array_equal(data[20:], 3.0)

    def test_missing_file(self, tmp_path):
        """`load_echoframe_rf_timetag` raises `ValueError` for a missing file."""
        with pytest.raises(ValueError):
            load_echoframe_rf_timetag(tmp_path / "nonexistent.dat")
