"""Unit tests for confusius.io.iconeus_raw module."""

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import numpy.testing as nptest
import pytest
import xarray as xr

from confusius.io.iconeus_raw import load_iconeus_raw, load_iconeus_raw_metadata


def _encrypt(values: npt.ArrayLike, index: int) -> np.ndarray:
    """Encode values using the legacy HRAW obfuscation function."""
    return np.asarray(values, dtype=np.float64) * (1005 * index) + 72


def _create_iconeus_raw_files(
    tmp_path: Path,
    *,
    size_x: int = 3,
    size_y: int = 2,
    size_z: int = 4,
    n_compound: int = 1,
    n_frames: int = 5,
    n_blocks: int = 2,
    compound: bool = True,
    legacy: bool = False,
    truncate: bool = False,
    lateral_spacing: float = 0.1,
    elevation_spacing: float = 0.2,
    axial_spacing: float = 0.05,
    depth_near: float = 1.0,
    depth_far: float = 10.0,
) -> tuple[Path, Path, np.ndarray]:
    """Create synthetic Iconeus RAW/HRAW files and return the expected IQ array."""
    meta_path = tmp_path / "sample.hraw"
    raw_path = tmp_path / "sample.raw"

    with h5py.File(meta_path, "w") as f:
        f.create_dataset("F1", data=_encrypt([[15.625]], 1))
        f.create_dataset("F2", data=_encrypt([[11000.0]], 2))
        f.create_dataset("F3", data=_encrypt([[1540.0]], 3))
        f.create_dataset("F4", data=_encrypt([[500.0]], 4))
        f.create_dataset("F5", data=_encrypt([[1.0], [128.0]], 5))
        f.create_dataset("F6", data=_encrypt([[depth_near], [depth_far]], 6))
        f.create_dataset("F7", data=_encrypt([[-10.0], [0.0], [10.0]], 7))
        f.create_dataset(
            "F8",
            data=_encrypt([[lateral_spacing], [elevation_spacing], [axial_spacing]], 8),
        )
        f.create_dataset(
            "F9",
            data=_encrypt([size_x, size_y, size_z, n_compound, n_frames], 9).reshape(
                -1, 1
            ),
        )
        f.create_dataset("F10", data=_encrypt([[float(compound)]], 10))
        f.create_dataset("F11", data=_encrypt([[float(n_blocks)]], 11))
        f.create_dataset("F12", data=_encrypt([[float(legacy)]], 12))
        f.create_dataset("F13", data=np.asarray([[b"2Dscan"]]))

    expected = np.zeros(
        (size_z, size_y, size_x, n_frames, n_blocks), dtype=np.complex64
    )
    for block in range(n_blocks):
        for frame in range(n_frames):
            for x in range(size_x):
                for y in range(size_y):
                    for z in range(size_z):
                        expected[z, y, x, frame, block] = complex(
                            z + 10 * y + 100 * x + 1000 * frame + 100000 * block,
                            -(z + 10 * y + 100 * x),
                        )

    floats = np.empty(
        (2, size_z, size_y, size_x, n_compound, n_frames, n_blocks),
        dtype="<f4",
        order="F",
    )
    for band, component in enumerate((expected.real, expected.imag)):
        floats[band] = np.broadcast_to(
            component[:, :, :, None, :, :],
            (size_z, size_y, size_x, n_compound, n_frames, n_blocks),
        )

    with raw_path.open("wb") as stream:
        if legacy:
            np.zeros(117, dtype=np.uint8).tofile(stream)
        payload = floats.ravel(order="F")
        if truncate:
            payload = payload[:-1]
        payload.tofile(stream)

    expected_iq = np.transpose(expected[:, :, :, :, :], (4, 3, 1, 0, 2)).reshape(
        n_blocks * n_frames, size_y, size_z, size_x
    )
    return raw_path, meta_path, expected_iq


class TestLoadIconeusRawMetadata:
    """Tests for load_iconeus_raw_metadata."""

    def test_decodes_fields_and_converts_units(self, tmp_path):
        """Metadata fields are decrypted, reshaped, and converted to ConfUSIus units."""
        _, meta_path, _ = _create_iconeus_raw_files(
            tmp_path,
            lateral_spacing=0.11,
            elevation_spacing=0.4,
            axial_spacing=0.09856,
        )

        meta = load_iconeus_raw_metadata(meta_path)

        assert meta["size_x"] == 3
        assert meta["size_y"] == 2
        assert meta["size_z"] == 4
        assert meta["n_compound"] == 1
        assert meta["n_frames_per_block"] == 5
        assert meta["n_blocks"] == 2
        assert meta["lateral_spacing"] == pytest.approx(0.11)
        assert meta["elevation_spacing"] == pytest.approx(0.4)
        assert meta["axial_spacing"] == pytest.approx(0.09856)
        assert meta["depth_near"] == pytest.approx(1.0)
        assert meta["depth_far"] == pytest.approx(10.0)
        assert meta["transmit_frequency"] == pytest.approx(15.625e6)
        assert meta["pulse_repetition_frequency"] == pytest.approx(11000.0)
        assert meta["beamforming_sound_velocity"] == pytest.approx(1540.0)
        assert meta["frame_rate"] == pytest.approx(500.0)
        nptest.assert_allclose(meta["plane_wave_angles"], [-10.0, 0.0, 10.0])
        nptest.assert_allclose(meta["receive_aperture"], [1.0, 128.0])
        assert meta["compound"] is True
        assert meta["is_legacy_format"] is False
        assert meta["acquisition_mode"] == "2Dscan"

    def test_missing_field_raises(self, tmp_path):
        """A HRAW file missing a required field is rejected."""
        _, meta_path, _ = _create_iconeus_raw_files(tmp_path)
        with h5py.File(meta_path, "a") as f:
            del f["F7"]

        with pytest.raises(ValueError, match="missing fields"):
            load_iconeus_raw_metadata(meta_path)


class TestLoadIconeusRaw:
    """Tests for load_iconeus_raw."""

    def test_loads_axes_values_and_timing(self, tmp_path):
        """Loaded data matches the documented payload axis order and timing."""
        raw_path, meta_path, expected = _create_iconeus_raw_files(tmp_path)

        data = load_iconeus_raw(raw_path, meta_path)
        values = data.compute().values

        assert isinstance(data, xr.DataArray)
        assert data.dims == ("time", "k", "j", "i")
        assert data.shape == (10, 2, 4, 3)
        assert data.dtype == np.complex64
        nptest.assert_array_equal(values, expected)
        nptest.assert_allclose(data.time.values, np.arange(10) / 500.0)
        assert data.time.attrs["units"] == "s"
        assert data.time.attrs["volume_acquisition_duration"] == pytest.approx(0.002)

    def test_preserves_blocks_as_chunks(self, tmp_path):
        """Each acquisition block is one chunk along the time axis."""
        raw_path, meta_path, _ = _create_iconeus_raw_files(tmp_path)

        data = load_iconeus_raw(raw_path, meta_path)

        assert data.chunks == ((5, 5), (2,), (4,), (3,))

    def test_attaches_iq_metadata_and_geometry(self, tmp_path):
        """Acquisition metadata and probe geometry are attached to the array."""
        raw_path, meta_path, _ = _create_iconeus_raw_files(
            tmp_path,
            lateral_spacing=0.11,
            elevation_spacing=0.4,
            axial_spacing=0.09856,
            depth_near=1.0,
        )

        data = load_iconeus_raw(raw_path, meta_path)

        assert data.attrs["transmit_frequency"] == pytest.approx(15.625e6)
        assert data.attrs["beamforming_sound_velocity"] == pytest.approx(1540.0)
        assert data.attrs["compound_sampling_frequency"] == pytest.approx(500.0)
        assert data.attrs["pulse_repetition_frequency"] == pytest.approx(11000.0)
        assert data.attrs["depth"] == pytest.approx((1.0, 10.0))
        assert data.attrs["receive_aperture"] == [1, 128]
        assert data.attrs["iconeus_acquisition_mode"] == "2Dscan"
        assert data.attrs["iconeus_legacy_format"] is False
        assert "n_volumes_per_block" not in data.attrs
        assert "n_blocks" not in data.attrs
        assert "frame_rate" not in data.attrs
        assert data.name == "sample"

        nptest.assert_allclose(data.coords["z"].isel(k=1, j=0, i=0).values, 0.2)
        nptest.assert_allclose(data.coords["z"].isel(k=0, j=0, i=0).values, -0.2)
        nptest.assert_allclose(data.coords["y"].isel(k=0, j=1, i=0).values, 1.09856)
        nptest.assert_allclose(data.coords["y"].isel(k=0, j=0, i=0).values, 1.0)
        nptest.assert_allclose(data.coords["x"].isel(k=0, j=0, i=0).values, -0.11)
        nptest.assert_allclose(data.coords["x"].isel(k=0, j=0, i=2).values, 0.11)

    def test_discovers_hraw_sidecar(self, tmp_path):
        """Omitting meta_path finds the matching .hraw file next to the payload."""
        raw_path, _, expected = _create_iconeus_raw_files(tmp_path)

        data = load_iconeus_raw(raw_path)

        nptest.assert_array_equal(data.compute().values, expected)

    def test_legacy_prefix_is_skipped(self, tmp_path):
        """The legacy 117-byte prefix is not treated as IQ payload."""
        raw_path, meta_path, expected = _create_iconeus_raw_files(tmp_path, legacy=True)

        data = load_iconeus_raw(raw_path, meta_path)

        assert data.attrs["iconeus_legacy_format"] is True
        nptest.assert_array_equal(data.compute().values, expected)

    def test_uncompounded_acquisition_raises(self, tmp_path):
        """Uncompounded acquisitions are rejected explicitly."""
        raw_path, meta_path, _ = _create_iconeus_raw_files(tmp_path, compound=False)

        with pytest.raises(ValueError, match="not compounded"):
            load_iconeus_raw(raw_path, meta_path)

    def test_multiple_compound_images_raise(self, tmp_path):
        """A non-singleton compound axis is rejected instead of silently flattened."""
        raw_path, meta_path, _ = _create_iconeus_raw_files(tmp_path, n_compound=3)

        with pytest.raises(ValueError, match="multiple compound images"):
            load_iconeus_raw(raw_path, meta_path)

    def test_truncated_payload_raises(self, tmp_path):
        """A payload shorter than the metadata dimensions is rejected eagerly."""
        raw_path, meta_path, _ = _create_iconeus_raw_files(tmp_path, truncate=True)

        with pytest.raises(ValueError, match="shorter than the expected payload"):
            load_iconeus_raw(raw_path, meta_path)
