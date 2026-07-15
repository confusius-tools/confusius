"""Unit tests for the binary SCAN v2 loader in confusius.io.scan."""

import struct
from pathlib import Path

import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

from confusius.io.scan import _SCAN_V2_OFFSETS, SCAN_V2_MAGIC, load_scan

_SIZE_X = 4
_SIZE_Y = 1
_SIZE_Z = 3
_N_TIME = 5
_NPOSE = 1
_NBLOCK = 1
_DT = 0.4
_DX_M = 0.00011
_DY_M = 0.0004
_DZ_M = 0.00009856
_STRINGS = ["default sequence", "proj-01", "sub-01", "ses-01"]


def _write_scan_v2(
    path: Path,
    payload: np.ndarray,
    *,
    size_x: int = _SIZE_X,
    size_y: int = _SIZE_Y,
    size_z: int = _SIZE_Z,
    n_time: int = _N_TIME,
    npose: int = _NPOSE,
    nblock_repeat: int = _NBLOCK,
    dt: float = _DT,
    times: np.ndarray | None = None,
    strings: list[str] | None = None,
    depth_start: float | None = None,
    payload_bytes_override: int | None = None,
) -> None:
    """Write a synthetic binary SCAN v2 file.

    The header reproduces the fixed-position layout the loader relies on (magic, sizes,
    voxel spacings, time coordinates) followed by a block of `uint32`-length-prefixed
    ASCII strings, then the little-endian `float64` payload.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path.
    payload : numpy.ndarray
        Payload array; written in C order as little-endian float64.
    size_x, size_y, size_z, n_time, npose, nblock_repeat : int
        Dimension fields written into the header.
    dt : float
        Sampling period written into the header.
    times : (n_time,) numpy.ndarray, optional
        Time-coordinate values. If not provided, `dt * (arange(n_time) + 1)` is used.
    strings : list[str], optional
        Provenance strings appended as length-prefixed records. If not provided,
        `_STRINGS` is used.
    depth_start : float, optional
        Depth-axis origin in mm. If provided, an adjacent `(start, end)` depth-range
        pair is embedded so the loader can recover the depth origin. If not provided,
        no depth range is written and the loader falls back to a zero origin.
    payload_bytes_override : int, optional
        Value to write into the payload-size header field instead of the true size.
        Used to exercise the size-mismatch error path.

    Returns
    -------
    None
        The file is written as a side effect.
    """
    if times is None:
        times = dt * (np.arange(n_time) + 1)
    if strings is None:
        strings = _STRINGS

    o = _SCAN_V2_OFFSETS
    time_end = o["time_coords"] + 8 * n_time

    # Optional adjacent (start, end) depth-range pair, spaced by the depth voxel count,
    # so the loader's span-search can recover the depth origin.
    depth_bytes = b""
    if depth_start is not None:
        depth_end = depth_start + (size_z - 1) * _DZ_M * 1e3
        depth_bytes = struct.pack("<dd", depth_start, depth_end)

    string_bytes = bytearray()
    for text in strings:
        encoded = text.encode("ascii")
        string_bytes += struct.pack("<I", len(encoded)) + encoded

    total_header_bytes = time_end + len(depth_bytes) + len(string_bytes)
    payload = np.ascontiguousarray(payload, dtype="<f8")
    payload_bytes = (
        payload_bytes_override if payload_bytes_override is not None else payload.nbytes
    )

    header = bytearray(total_header_bytes)
    header[0 : len(SCAN_V2_MAGIC)] = SCAN_V2_MAGIC
    struct.pack_into("<Q", header, o["total_header_bytes"], total_header_bytes)
    struct.pack_into("<Q", header, o["payload_bytes"], payload_bytes)
    struct.pack_into("<Q", header, o["size_x"], size_x)
    struct.pack_into("<Q", header, o["size_y"], size_y)
    struct.pack_into("<Q", header, o["size_z"], size_z)
    struct.pack_into("<Q", header, o["n_time"], n_time)
    struct.pack_into("<Q", header, o["npose"], npose)
    struct.pack_into("<Q", header, o["nblock_repeat"], nblock_repeat)
    struct.pack_into("<d", header, o["dt"], dt)
    struct.pack_into("<d", header, o["x_voxel_m"], _DX_M)
    struct.pack_into("<d", header, o["y_voxel_m"], _DY_M)
    struct.pack_into("<d", header, o["z_voxel_m"], _DZ_M)
    struct.pack_into(f"<{n_time}d", header, o["time_coords"], *times.tolist())
    header[time_end : time_end + len(depth_bytes)] = depth_bytes
    header[time_end + len(depth_bytes) : total_header_bytes] = string_bytes

    path.write_bytes(bytes(header) + payload.tobytes())


def _raw_payload(
    size_x: int = _SIZE_X,
    size_y: int = _SIZE_Y,
    size_z: int = _SIZE_Z,
    n_time: int = _N_TIME,
    npose: int = _NPOSE,
    nblock_repeat: int = _NBLOCK,
) -> np.ndarray:
    """Return a deterministic payload array in the SCAN v2 C-order layout.

    Parameters
    ----------
    size_x, size_y, size_z, n_time, npose, nblock_repeat : int
        Dimension sizes.

    Returns
    -------
    numpy.ndarray
        Array of shape `(n_time, npose, nblock_repeat, size_z, size_y, size_x)`.
    """
    shape = (n_time, npose, nblock_repeat, size_z, size_y, size_x)
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)


def _expected_confusius(raw: np.ndarray) -> np.ndarray:
    """Transform a raw v2 payload into the expected ConfUSIus array.

    Mirrors `_load_scan_v2`: squeeze `nblock_repeat`, swap depth/elevation, and squeeze
    a singleton pose axis.

    Parameters
    ----------
    raw : numpy.ndarray
        Array of shape `(n_time, npose, nblock_repeat, size_z, size_y, size_x)` with
        `nblock_repeat == 1`.

    Returns
    -------
    numpy.ndarray
        Expected array in ConfUSIus axis order.
    """
    sq = raw.squeeze(axis=2)
    swapped = np.transpose(sq, [0, 1, 3, 2, 4])
    if swapped.shape[1] == 1:
        return swapped.squeeze(axis=1)
    return swapped


@pytest.fixture
def scan_v2_path(tmp_path: Path) -> Path:
    """Path to a synthetic single-pose 2D SCAN v2 file."""
    path = tmp_path / "scan_v2_2d.scan"
    _write_scan_v2(path, _raw_payload())
    return path


@pytest.fixture
def scan_v2(scan_v2_path: Path) -> xr.DataArray:
    """Loaded single-pose 2D SCAN v2 DataArray."""
    return load_scan(scan_v2_path)


@pytest.fixture
def scan_v2_multipose_path(tmp_path: Path) -> Path:
    """Path to a synthetic multi-pose SCAN v2 file (sizeY > 1, npose > 1)."""
    path = tmp_path / "scan_v2_multipose.scan"
    raw = _raw_payload(size_y=2, npose=2)
    _write_scan_v2(path, raw, size_y=2, npose=2)
    return path


class TestLoadScanV2:
    """Tests for load_scan dispatching to the binary v2 loader."""

    def test_dims(self, scan_v2: xr.DataArray) -> None:
        """Single-pose v2 produces dims (time, z, y, x)."""
        assert scan_v2.dims == ("time", "z", "y", "x")

    def test_shape(self, scan_v2: xr.DataArray) -> None:
        """Shape maps Iconeus (sizeY, sizeZ, sizeX) to ConfUSIus (z, y, x)."""
        assert scan_v2.shape == (_N_TIME, _SIZE_Y, _SIZE_Z, _SIZE_X)

    def test_dtype_float64(self, scan_v2: xr.DataArray) -> None:
        """v2 data is float64."""
        assert scan_v2.dtype == np.float64

    def test_lazy(self, scan_v2: xr.DataArray) -> None:
        """v2 returns a lazy Dask-backed DataArray."""
        assert isinstance(scan_v2.data, dask_array.Array)

    def test_values(self, scan_v2: xr.DataArray) -> None:
        """Loaded values match the depth/elevation-swapped payload."""
        expected = _expected_confusius(_raw_payload())
        np.testing.assert_array_equal(scan_v2.values, expected)

    def test_time_coord(self, scan_v2: xr.DataArray) -> None:
        """Time coordinate matches header values with end-referenced metadata."""
        expected = _DT * (np.arange(_N_TIME) + 1)
        np.testing.assert_allclose(scan_v2.coords["time"].values, expected)
        assert scan_v2.coords["time"].attrs["units"] == "s"
        assert scan_v2.coords["time"].attrs["volume_acquisition_reference"] == "end"

    def test_spatial_voxdim(self, scan_v2: xr.DataArray) -> None:
        """Voxel dimensions come from the header spacings, in mm."""
        np.testing.assert_allclose(scan_v2.coords["x"].attrs["voxdim"], _DX_M * 1e3)
        np.testing.assert_allclose(scan_v2.coords["z"].attrs["voxdim"], _DY_M * 1e3)
        np.testing.assert_allclose(scan_v2.coords["y"].attrs["voxdim"], _DZ_M * 1e3)

    def test_lateral_coord_centered(self, scan_v2: xr.DataArray) -> None:
        """Lateral (x) coordinate is centered on zero with correct spacing."""
        expected = (np.arange(_SIZE_X) - (_SIZE_X - 1) / 2) * _DX_M * 1e3
        np.testing.assert_allclose(scan_v2.coords["x"].values, expected)

    def test_depth_coord_from_zero(self, scan_v2: xr.DataArray) -> None:
        """Depth (y) coordinate starts at zero when no depth range is in the header."""
        expected = np.arange(_SIZE_Z) * _DZ_M * 1e3
        np.testing.assert_allclose(scan_v2.coords["y"].values, expected)

    def test_depth_origin_recovered(self, tmp_path: Path) -> None:
        """Depth (y) origin is recovered from an embedded depth-range pair."""
        path = tmp_path / "scan_v2_depth.scan"
        _write_scan_v2(path, _raw_payload(), depth_start=1.0)
        da = load_scan(path)
        expected = 1.0 + np.arange(_SIZE_Z) * _DZ_M * 1e3
        np.testing.assert_allclose(da.coords["y"].values, expected)

    def test_spatial_units_mm(self, scan_v2: xr.DataArray) -> None:
        """Spatial coordinates are in mm."""
        for dim in ("x", "y", "z"):
            assert scan_v2.coords[dim].attrs["units"] == "mm"

    def test_no_physical_to_lab(self, scan_v2: xr.DataArray) -> None:
        """v2 carries an empty affines dict (no physical_to_lab yet)."""
        assert scan_v2.attrs["affines"] == {}

    def test_scan_format_attr(self, scan_v2: xr.DataArray) -> None:
        """v2 records its on-disk format."""
        assert scan_v2.attrs["iconeus_scan_format"] == "v2"

    def test_scan_mode_attr(self, scan_v2: xr.DataArray) -> None:
        """Single-pose v2 is reported as 2Dscan."""
        assert scan_v2.attrs["iconeus_scan_mode"] == "2Dscan"

    def test_header_strings_recovered(self, scan_v2: xr.DataArray) -> None:
        """Length-prefixed header strings are recovered best-effort."""
        for text in _STRINGS:
            assert text in scan_v2.attrs["iconeus_header_strings"]

    def test_name_from_stem(self, scan_v2: xr.DataArray, scan_v2_path: Path) -> None:
        """v2 DataArray name falls back to the file stem."""
        assert scan_v2.name == scan_v2_path.stem


class TestLoadScanV2Multipose:
    """Tests for multi-pose v2 files (inferred layout)."""

    def test_dims(self, scan_v2_multipose_path: Path) -> None:
        """Multi-pose v2 produces dims (time, pose, z, y, x)."""
        da = load_scan(scan_v2_multipose_path)
        assert da.dims == ("time", "pose", "z", "y", "x")

    def test_shape(self, scan_v2_multipose_path: Path) -> None:
        """Multi-pose shape keeps pose and maps elevation to z."""
        da = load_scan(scan_v2_multipose_path)
        assert da.shape == (_N_TIME, 2, 2, _SIZE_Z, _SIZE_X)

    def test_pose_coord(self, scan_v2_multipose_path: Path) -> None:
        """Multi-pose v2 has an integer pose coordinate."""
        da = load_scan(scan_v2_multipose_path)
        np.testing.assert_array_equal(da.coords["pose"].values, np.arange(2))

    def test_scan_mode_attr(self, scan_v2_multipose_path: Path) -> None:
        """Multi-pose v2 with time is reported as 4Dscan."""
        da = load_scan(scan_v2_multipose_path)
        assert da.attrs["iconeus_scan_mode"] == "4Dscan"

    def test_values(self, scan_v2_multipose_path: Path) -> None:
        """Multi-pose loaded values match the swapped payload."""
        da = load_scan(scan_v2_multipose_path)
        expected = _expected_confusius(_raw_payload(size_y=2, npose=2))
        np.testing.assert_array_equal(da.values, expected)


class TestLoadScanV2Errors:
    """Tests for v2 error handling and dispatch."""

    def test_bps_path_rejected(self, scan_v2_path: Path, tmp_path: Path) -> None:
        """bps_path is rejected for v2 files."""
        bps = tmp_path / "sidecar.bps"
        bps.write_bytes(b"")
        with pytest.raises(ValueError, match="bps_path is not supported for SCAN v2"):
            load_scan(scan_v2_path, bps_path=bps)

    def test_unrecognized_format_raises(self, tmp_path: Path) -> None:
        """A non-HDF5 file without the SCAN magic raises a descriptive error."""
        path = tmp_path / "mystery.scan"
        path.write_bytes(b"XXXX not a scan file")
        with pytest.raises(ValueError, match="not a SCAN file recognised"):
            load_scan(path)

    def test_payload_size_mismatch_raises(self, tmp_path: Path) -> None:
        """A payload-size field inconsistent with the dimensions raises."""
        path = tmp_path / "bad_size.scan"
        _write_scan_v2(path, _raw_payload(), payload_bytes_override=12345)
        with pytest.raises(ValueError, match="does not match the product"):
            load_scan(path)
