"""Unit tests for confusius.io.iconeus_raw module."""


import h5py
import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
import xarray as xr

from confusius.bids import FUSI_BIDS_FIELDS, from_bids
from confusius.io.iconeus_raw import load_iconeus_raw, load_iconeus_raw_metadata


def _encrypt(values: npt.ArrayLike, index: int) -> np.ndarray:
    """Encode values using the legacy HRAW obfuscation function."""
    return np.asarray(values, dtype=np.float64) * (1005 * index) + 72


class TestLoadIconeusRawMetadata:
    """Tests for load_iconeus_raw_metadata."""

    def test_decodes_fields_and_converts_units(self, make_iconeus_raw_files):
        """Metadata fields are decrypted, reshaped, and converted to ConfUSIus units."""
        _, meta_path, _ = make_iconeus_raw_files(
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

    @pytest.mark.parametrize("bound", ["depth_near", "depth_far"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_depth_raises(self, make_iconeus_raw_files, bound, value):
        """Non-finite near and far depth bounds are rejected."""
        _, meta_path, _ = make_iconeus_raw_files(**{bound: value})

        with pytest.raises(ValueError, match="HRAW depth bounds must be finite"):
            load_iconeus_raw_metadata(meta_path)

    @pytest.mark.parametrize("dimension", range(6))
    @pytest.mark.parametrize("value", [0, -1, 3.5, np.nan, np.inf, -np.inf])
    def test_invalid_dimensions_raise(self, make_iconeus_raw_files, dimension, value):
        """Every dimension and the block count must be finite positive integers."""
        _, meta_path, _ = make_iconeus_raw_files()
        with h5py.File(meta_path, "a") as f:
            field = 9 if dimension < 5 else 11
            values = (f[f"F{field}"][()] - 72) / (1005 * field)
            values.reshape(-1)[dimension if dimension < 5 else 0] = value
            f[f"F{field}"][...] = _encrypt(values, field)

        with pytest.raises(ValueError, match="finite positive integers"):
            load_iconeus_raw_metadata(meta_path)

    def test_nearly_integer_dimensions_are_rounded(self, make_iconeus_raw_files):
        """Floating-point decoding noise does not truncate whole-number counts."""
        _, meta_path, _ = make_iconeus_raw_files()
        with h5py.File(meta_path, "a") as f:
            f["F9"][...] = _encrypt([[3 - 1e-10], [2 + 1e-10], [4], [1], [5]], 9)
            f["F11"][...] = _encrypt([[2 - 1e-10]], 11)

        meta = load_iconeus_raw_metadata(meta_path)

        assert tuple(
            meta[key]
            for key in (
                "size_x",
                "size_y",
                "size_z",
                "n_compound",
                "n_frames_per_block",
                "n_blocks",
            )
        ) == (3, 2, 4, 1, 5, 2)

    @pytest.mark.parametrize(
        ("field", "values", "message"),
        [
            (1, [], "F1 is empty"),
            (6, [1], "depth must contain"),
            (8, [1, 2], "spacing must contain"),
            (9, [1, 2, 3, 4], "dimensions must contain"),
            (1, [0], "transmit frequency must be positive"),
            (6, [10, 1], "far depth must be greater"),
        ],
    )
    def test_invalid_metadata_raises(self, make_iconeus_raw_files, field, values, message):
        """Malformed fields fail at the metadata boundary with specific errors."""
        _, meta_path, _ = make_iconeus_raw_files()
        with h5py.File(meta_path, "a") as f:
            del f[f"F{field}"]
            f.create_dataset(f"F{field}", data=_encrypt(values, field))

        with pytest.raises(ValueError, match=message):
            load_iconeus_raw_metadata(meta_path)

    def test_unicode_acquisition_mode(self, make_iconeus_raw_files):
        """Variable-length UTF-8 strings are decoded without byte conversion."""
        _, meta_path, _ = make_iconeus_raw_files()
        with h5py.File(meta_path, "a") as f:
            del f["F13"]
            f.create_dataset("F13", data="2Dscan", dtype=h5py.string_dtype())

        assert load_iconeus_raw_metadata(meta_path)["acquisition_mode"] == "2Dscan"

    def test_missing_field_raises(self, make_iconeus_raw_files):
        """A HRAW file missing a required field is rejected."""
        _, meta_path, _ = make_iconeus_raw_files()
        with h5py.File(meta_path, "a") as f:
            del f["F7"]

        with pytest.raises(ValueError, match="missing fields"):
            load_iconeus_raw_metadata(meta_path)


class TestLoadIconeusRaw:
    """Tests for load_iconeus_raw."""

    def test_loads_axes_values_and_timing(self, make_iconeus_raw_files):
        """Loaded data matches the documented payload axis order and timing."""
        raw_path, meta_path, expected = make_iconeus_raw_files()

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

    def test_preserves_blocks_as_chunks(self, make_iconeus_raw_files):
        """Each acquisition block is one chunk along the time axis."""
        raw_path, meta_path, _ = make_iconeus_raw_files()

        data = load_iconeus_raw(raw_path, meta_path)

        assert data.chunks == ((5, 5), (2,), (4,), (3,))

    def test_attaches_iq_metadata_and_geometry(self, make_iconeus_raw_files):
        """Acquisition metadata and probe geometry are attached to the array."""
        raw_path, meta_path, _ = make_iconeus_raw_files(
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

    def test_metadata_names_follow_bids_or_iconeus_namespace(self, make_iconeus_raw_files):
        """Imported metadata uses standard BIDS names or the Iconeus namespace."""
        raw_path, _, _ = make_iconeus_raw_files()
        data = load_iconeus_raw(raw_path)
        bids_names = from_bids(dict.fromkeys(FUSI_BIDS_FIELDS))

        assert all(
            key in bids_names or key.startswith("iconeus_") for key in data.attrs
        )

    def test_discovers_hraw_sidecar(self, make_iconeus_raw_files):
        """Omitting meta_path finds the matching .hraw file next to the payload."""
        raw_path, _, expected = make_iconeus_raw_files()

        data = load_iconeus_raw(raw_path)

        nptest.assert_array_equal(data.compute().values, expected)

    def test_legacy_prefix_is_skipped(self, make_iconeus_raw_files):
        """The legacy 117-byte prefix is not treated as IQ payload."""
        raw_path, meta_path, expected = make_iconeus_raw_files(legacy=True)

        data = load_iconeus_raw(raw_path, meta_path)

        assert data.attrs["iconeus_legacy_format"] is True
        nptest.assert_array_equal(data.compute().values, expected)

    def test_uncompounded_acquisition_raises(self, make_iconeus_raw_files):
        """Uncompounded acquisitions are rejected explicitly."""
        raw_path, meta_path, _ = make_iconeus_raw_files(compound=False)

        with pytest.raises(ValueError, match="not compounded"):
            load_iconeus_raw(raw_path, meta_path)

    def test_multiple_compound_images_raise(self, make_iconeus_raw_files):
        """A non-singleton compound axis is rejected instead of silently flattened."""
        raw_path, meta_path, _ = make_iconeus_raw_files(n_compound=3)

        with pytest.raises(ValueError, match="multiple compound images"):
            load_iconeus_raw(raw_path, meta_path)

    def test_truncated_payload_raises(self, make_iconeus_raw_files):
        """A payload shorter than the metadata dimensions is rejected eagerly."""
        raw_path, meta_path, _ = make_iconeus_raw_files(truncate=True)

        with pytest.raises(ValueError, match="shorter than the expected payload"):
            load_iconeus_raw(raw_path, meta_path)
