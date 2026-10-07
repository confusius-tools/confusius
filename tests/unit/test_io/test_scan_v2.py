"""Unit tests for the binary SCAN v2 loader in confusius.io.scan."""

import math
import struct
from pathlib import Path
from unittest.mock import patch

import dask.array as dask_array
import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import get_voxel_to_world_affine
from confusius.io.scan import (
    _SCAN_V2_OFFSETS,
    SCAN_V2_MAGIC,
    WORLD_TO_PROBE_PERMUTATION,
    load_bps,
    load_scan,
)
from confusius.validation import validate_voxeldata

_SIZE_X = 4
_SIZE_Y = 1
_SIZE_Z = 3
_N_TIME = 5
_NPOSE = 1
_NBLOCK = 1

_VOXEL_DIM_BY_WORLD_NAME = {"z": "k", "y": "j", "x": "i"}


def _world_coord_1d(da: xr.DataArray, name: str) -> np.ndarray:
    """Return a world coordinate's 1D values, reducing other axis-aligned dims."""
    coord = da.coords[name]
    dim = _VOXEL_DIM_BY_WORLD_NAME[name]
    if coord.dims == (dim,):
        return coord.values
    others = {d: 0 for d in coord.dims if d != dim}
    return coord.isel(others).values


_DT = 0.4
_DX_M = 0.00011
_DY_M = 0.0004
_DZ_M = 0.00009856
_SERIAL = "SN-0001"
_HARDWARE = "HW-1"
_SVD_CUTOFF = 60
_PD_WINDOW = 200
# Acquisition-block values. Transmit frequency deliberately differs from the probe
# center frequency so tests prove the two distinct fields are read from the right spots.
_PROBE_MODEL = "IcoPrime"
_CENTER_FREQ = 15.625
_TRANSMIT_FREQ = 9.0
_PITCH = 0.11
_SCAN_DEPTH_CM = 1.5
_FOCAL_DEPTH = 8.0
_PRF = 5500.0
_ADC = 62.5
_ANGLES = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
_ACQ_DEPTH_START = 1.0
# 6DOF probe pose written into the orientation block: (tx, ty, tz) in meters,
# (rx, ry, rz) in radians. A 90-degree rz gives a non-trivial rotation to validate.
_POSE = (0.0007, 0.0026, 0.0, 0.0, 0.0, math.pi / 2)


def _acquisition_block(
    n_time: int, size_z: int, depth_start: float, corrupt: str | None = None
) -> bytes:
    """Build the structured acquisition block the loader parses from `n_time`.

    Layout (immediately after the time-coordinate array): frame indices, a 64-byte
    orientation block, the frequency block (center freq, pitch, scan depth, focal depth),
    a 16-byte gap, the probe-name string, the depth range, the transmit/PRF/ADC block,
    then the plane-wave angle count and values.

    Parameters
    ----------
    n_time : int
        Number of time points (matches the frame-index count).
    size_z : int
        Number of depth voxels, used to space the depth range.
    depth_start : float
        Depth origin in mm; the depth range is `(depth_start, depth_start + span)`.
    corrupt : {"name", "depth", "pose"}, optional
        Inject a defect to exercise the loader's degradation guards: `"name"` writes a
        non-printable probe name, `"depth"` breaks the depth-range span so the loader's
        span-search fails and the anchor check cannot confirm alignment, `"pose"` writes
        an implausible probe pose so no affine is built.

    Returns
    -------
    bytes
        The packed acquisition block.
    """
    depth_end = depth_start + (size_z - 1) * _DZ_M * 1e3
    if corrupt == "depth":
        depth_end = depth_start + 99.0  # break the span so span-search finds no pair
    name = _PROBE_MODEL.encode("ascii")
    if corrupt == "name":
        name = bytes(range(1, len(name) + 1))  # valid length, non-printable
    # Orientation block: 6DOF pose (6 f64) + a flag slot + padding = 64 bytes.
    pose = (
        (5.0, *_POSE[1:]) if corrupt == "pose" else _POSE
    )  # 5 m translation: implausible
    orientation = struct.pack("<6d", *pose) + struct.pack("<Q", 1) + bytes(8)
    return (
        struct.pack(f"<{n_time}I", *range(n_time))  # frame indices
        + orientation
        + struct.pack("<dddd", _CENTER_FREQ, _PITCH, _SCAN_DEPTH_CM, _FOCAL_DEPTH)
        + bytes(16)  # unknown probe block
        + struct.pack("<H", len(name))
        + name
        + struct.pack("<dd", depth_start, depth_end)
        + struct.pack("<dddd", _TRANSMIT_FREQ, _PRF, _ADC, 1.0)
        + struct.pack("<I", len(_ANGLES))
        + struct.pack(f"<{len(_ANGLES)}d", *_ANGLES)
    )


# Provenance layout matching the observed on-disk order: sequence, project, subject,
# session, species, <type>, scan, <unknown>, <unknown>, experimenter, then the two
# hex-encoded trailing fields (serial, hardware).
_STRINGS = [
    "default sequence",
    "proj-01",
    "sub-01",
    "ses-01",
    "Rat",
    "T",
    "scan-01",
    "none",
    "None",
    "user-01",
    _SERIAL.encode("ascii").hex().upper(),
    _HARDWARE.encode("ascii").hex().upper(),
]


def _pack_scan_v2_string(text: str, length_fmt: str = "<L") -> bytes:
    """Pack a SCAN v2 length-prefixed string.

    Parameters
    ----------
    text : str
        String to pack.
    length_fmt : str, default: "<L"
        `struct` format for the length prefix.

    Returns
    -------
    bytes
        Packed length prefix and ASCII bytes.
    """
    encoded = text.encode("ascii")
    return struct.pack(length_fmt, len(encoded)) + encoded


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
    timestamp: int | None = None,
    acquisition: bool = False,
    corrupt_acquisition: str | None = None,
    payload_bytes_override: int | None = None,
    dim6_intents: list[int] | None = None,
    slice_offsets: np.ndarray | None = None,
    integration_window: float = _DT,
) -> None:
    """Write a PyIconeus-layout synthetic binary SCAN v2 file.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    payload : numpy.ndarray
        Payload in `(x, y, z, time, pose, dim6)` order.
    size_x, size_y, size_z, n_time, npose, nblock_repeat : int
        Header dimension fields.
    dt : float, default: `_DT`
        Time spacing in seconds.
    times : numpy.ndarray, optional
        Measured volume times. If not provided, a regular grid is used.
    strings : list of str, optional
        Provenance strings. If not provided, `_STRINGS` is used.
    depth_start : float, optional
        Depth origin in millimeters. If not provided, zero is used unless
        `acquisition` is set.
    timestamp : int, optional
        Unix timestamp for acquisition metadata.
    acquisition : bool, default: False
        Whether to write a non-identity probe pose and acquisition depth.
    corrupt_acquisition : {"pose", "name"}, optional
        Corruption mode used by error-path tests.
    payload_bytes_override : int, optional
        Payload byte count to write into the header.
    dim6_intents : list of int, optional
        Dim6 intent codes. If not provided, static SVD clutter filtering is used.
    slice_offsets : numpy.ndarray, optional
        Per-`k` offsets added to each measured volume time.
    integration_window : float, default: `_DT`
        Power Doppler integration duration in seconds.

    Returns
    -------
    None
        The file is written as a side effect.
    """
    if times is None:
        times = dt * (np.arange(n_time) + 1)
    if strings is None:
        strings = _STRINGS
    if depth_start is None:
        depth_start = _ACQ_DEPTH_START if acquisition else 0.0
    depth_end = depth_start + (size_z - 1) * _DZ_M * 1e3
    if timestamp is None:
        timestamp = 1_784_005_200

    o = _SCAN_V2_OFFSETS
    header = bytearray(o["dim6_count"])
    header[0 : len(SCAN_V2_MAGIC)] = SCAN_V2_MAGIC
    struct.pack_into("<Q", header, 0x04, 1)
    struct.pack_into("<Q", header, o["size_x"], size_x)
    struct.pack_into("<Q", header, o["size_y"], size_y)
    struct.pack_into("<Q", header, o["size_z"], size_z)
    struct.pack_into("<Q", header, o["n_time"], n_time)
    struct.pack_into("<Q", header, o["npose"], npose)

    if dim6_intents is None:
        dim6_intents = [0] * nblock_repeat
    header += struct.pack("<Q", len(dim6_intents))
    header += struct.pack(f"<{len(dim6_intents)}L", *dim6_intents)
    for intent in dim6_intents:
        if intent in {1, 3}:
            header += bytes(20)
        elif intent == 0:
            header += struct.pack("<LdLL", 0, dt, _SVD_CUTOFF, _PD_WINDOW)
        elif intent == 2:
            header += struct.pack("<ff", -1.5, 2.5) + bytes(12)
    header += struct.pack("<6d", _DX_M, _DY_M, _DZ_M, dt, 0.0, 0.0)

    measured = np.empty((n_time, npose, size_y), dtype="<f8")
    if slice_offsets is None:
        slice_offsets = np.zeros(size_y, dtype=np.float64)
    for t in range(n_time):
        measured[t, :, :] = times[t] + slice_offsets
    header += measured.ravel().tobytes()
    header += np.arange(measured.size, dtype="<u4").tobytes()

    translations = np.zeros((npose, 3), dtype="<f8")
    rotations = np.zeros((npose, 3), dtype="<f8")
    for pose in range(npose):
        if acquisition:
            translations[pose] = _POSE[:3]
            translations[pose, 1] += pose * 0.001
            rotations[pose] = _POSE[3:]
    if corrupt_acquisition == "pose":
        translations[0, 0] = 5.0
    header += translations.tobytes()
    header += rotations.tobytes()

    probe_name = _PROBE_MODEL if corrupt_acquisition != "name" else "bad-name"
    header += bytes(4)
    scan_mode_code = 3 if npose > 1 and n_time > 1 else 1 if n_time == 1 else 0
    header += struct.pack("<L", scan_mode_code)
    header += bytes(4)
    header += struct.pack("<Lddd", 0, _CENTER_FREQ, _PITCH, 1.5)
    header += bytes(8)
    header += struct.pack("<dH", 0.0, 128)
    header += _pack_scan_v2_string("2392", "<H")
    header += _pack_scan_v2_string(probe_name, "<H")
    header += struct.pack("<dd", depth_start, depth_end)
    header += struct.pack("<ddd", _TRANSMIT_FREQ, _PRF, _ADC)
    header += bytes(8)
    n_angles = 99_999 if corrupt_acquisition == "angles" else len(_ANGLES)
    header += struct.pack("<L", n_angles)
    if corrupt_acquisition != "angles":
        header += struct.pack(f"<{len(_ANGLES)}d", *_ANGLES)
    header += struct.pack("<L", 0) + bytes(8)
    header += struct.pack("<d", 12.0)
    header += bytes(4)
    header += struct.pack("<dL", 0.0, 0)
    header += struct.pack("<?", False) + bytes(1)
    header += struct.pack("<d", integration_window)

    sequence, project, subject, session, species, _, scan, _, _, user, *_ = strings
    for text in (sequence, project, "project description", subject, session, species):
        header += _pack_scan_v2_string(text)
    header += struct.pack("<LqQ", 0, timestamp, 0)
    header += _pack_scan_v2_string("subject description")
    header += struct.pack("<Lf", 0, 0.0)
    for text in ("", scan, "", "", "", user, "", ""):
        header += _pack_scan_v2_string(text)
    n_toggle = 99_999 if corrupt_acquisition == "toggles" else 0
    header += struct.pack("<qLL", timestamp, 0, n_toggle)
    header += struct.pack("<LLL", 1, 2, 3)

    total_header_bytes = len(header)
    payload = np.asarray(payload, dtype="<f8")
    payload_bytes = payload_bytes_override or payload.nbytes
    struct.pack_into("<Q", header, o["total_header_bytes"], total_header_bytes)
    struct.pack_into("<Q", header, o["payload_bytes"], payload_bytes)
    path.write_bytes(bytes(header) + payload.tobytes(order="F"))


def _raw_payload(
    size_x: int = _SIZE_X,
    size_y: int = _SIZE_Y,
    size_z: int = _SIZE_Z,
    n_time: int = _N_TIME,
    npose: int = _NPOSE,
    nblock_repeat: int = _NBLOCK,
) -> np.ndarray:
    """Return a deterministic payload array in the SCAN v2 Fortran layout.

    Parameters
    ----------
    size_x, size_y, size_z, n_time, npose, nblock_repeat : int
        Payload dimensions.

    Returns
    -------
    numpy.ndarray
        Payload with shape `(x, y, z, time, pose, dim6)`.
    """
    shape = (size_x, size_y, size_z, n_time, npose, nblock_repeat)
    return np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape, order="F")


def _expected_confusius(raw: np.ndarray) -> np.ndarray:
    """Transform a raw SCAN v2 payload to ConfUSIus order.

    Parameters
    ----------
    raw : numpy.ndarray
        Payload with shape `(x, y, z, time, pose, dim6)`.

    Returns
    -------
    numpy.ndarray
        Payload in VoxelData dimension order.
    """
    swapped = np.transpose(raw, [5, 3, 4, 1, 2, 0])
    if swapped.shape[0] == 1:
        swapped = swapped.squeeze(axis=0)
    pose_axis = 2 if swapped.ndim == 6 else 1
    if swapped.shape[pose_axis] == 1:
        swapped = swapped.squeeze(axis=pose_axis)
    return swapped


def _patch_u64(path: Path, offset: int, value: int) -> None:
    """Overwrite a little-endian uint64 header field in an existing file.

    Parameters
    ----------
    path : pathlib.Path
        File to patch in place.
    offset : int
        Byte offset of the field.
    value : int
        New value to write.

    Returns
    -------
    None
        The file is modified as a side effect.
    """
    data = bytearray(path.read_bytes())
    struct.pack_into("<Q", data, offset, value)
    path.write_bytes(data)


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


@pytest.fixture
def scan_v2_multiblock_path(tmp_path: Path) -> Path:
    """Path to a synthetic SCAN v2 file with dim6 count > 1."""
    path = tmp_path / "scan_v2_multiblock.scan"
    _write_scan_v2(path, _raw_payload(nblock_repeat=2), nblock_repeat=2)
    return path


class TestLoadScanV2:
    """Tests for load_scan dispatching to the binary v2 loader."""

    def test_dims(self, scan_v2: xr.DataArray) -> None:
        """Single-pose v2 produces voxel-to-world dims (time, k, j, i)."""
        assert scan_v2.dims == ("time", "k", "j", "i")

    def test_shape(self, scan_v2: xr.DataArray) -> None:
        """Shape maps Iconeus (sizeY, sizeZ, sizeX) to ConfUSIus (z, y, x)."""
        assert scan_v2.shape == (_N_TIME, _SIZE_Y, _SIZE_Z, _SIZE_X)

    def test_dtype_float64(self, scan_v2: xr.DataArray) -> None:
        """v2 data is float64."""
        assert scan_v2.dtype == np.float64

    def test_lazy(self, scan_v2: xr.DataArray) -> None:
        """v2 returns a lazy Dask-backed DataArray."""
        assert isinstance(scan_v2.data, dask_array.Array)

    def test_load_does_not_copy_payload(self, tmp_path: Path) -> None:
        """Loading must not copy the entire memory-mapped payload into RAM."""
        path = tmp_path / "lazy.scan"
        payload = _raw_payload()
        _write_scan_v2(path, payload)
        with patch.object(
            np.memmap, "copy", side_effect=AssertionError("Eager payload copy")
        ):
            data = load_scan(path)
        np.testing.assert_array_equal(data.values, _expected_confusius(payload))

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

    @pytest.mark.parametrize("time_origin", [0.0, 10.0])
    def test_slice_time_coord_for_stacked_slices(
        self, tmp_path: Path, time_origin: float
    ) -> None:
        """Slice integration windows fit within the consolidated volume window."""
        path = tmp_path / "scan_v2_stacked_slices.scan"
        times = time_origin + 0.4 + 2.4 * np.arange(_N_TIME)
        offsets = np.tile([0.0, 1.8, 0.6, 1.2], 4)
        _write_scan_v2(
            path,
            _raw_payload(size_y=16),
            size_y=16,
            dt=0.6,
            times=times,
            slice_offsets=offsets,
            integration_window=0.4,
        )
        da = load_scan(path)
        validate_voxeldata(da)
        assert da.coords["slice_time"].dims == ("time", "k")
        np.testing.assert_allclose(
            da.coords["slice_time"].values, times[:, np.newaxis] + offsets
        )
        np.testing.assert_allclose(da.coords["time"].values, times + offsets.max())
        assert da.coords["slice_time"].attrs["units"] == "s"
        assert da.coords["slice_time"].attrs["volume_acquisition_reference"] == "end"
        assert da.coords["slice_time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.4)
        assert da.coords["time"].attrs["volume_acquisition_duration"] == pytest.approx(
            2.2
        )

    def test_scalar_time_slice_time_coord(self, tmp_path: Path) -> None:
        """Single-volume SCAN v2 stacks keep 1D slice times."""
        path = tmp_path / "scan_v2_scalar_slice_time.scan"
        offsets = np.array([0.0, 0.05])
        _write_scan_v2(
            path,
            _raw_payload(size_y=2, n_time=1),
            size_y=2,
            n_time=1,
            acquisition=True,
            slice_offsets=offsets,
        )
        da = load_scan(path)
        assert "time" not in da.dims
        assert da.coords["slice_time"].dims == ("k",)
        np.testing.assert_allclose(da.coords["slice_time"].values, _DT + offsets)

    def test_elevation_spacing_from_header(self, scan_v2: xr.DataArray) -> None:
        """Elevation (z, singleton) spacing comes from the header spacing, in mm.

        The elevation axis has a single voxel, so its spacing can't be recovered
        from coordinate steps -- it must come from the voxel-to-world affine
        itself, matching the `y_voxel_m` header field.
        """
        affine = get_voxel_to_world_affine(scan_v2)
        spacing_z = np.linalg.norm(affine[:3, 0])
        np.testing.assert_allclose(spacing_z, _DY_M * 1e3)

    def test_lateral_coord_centered(self, scan_v2: xr.DataArray) -> None:
        """Lateral (x) coordinate is centered on zero with correct spacing."""
        expected = (np.arange(_SIZE_X) - (_SIZE_X - 1) / 2) * _DX_M * 1e3
        np.testing.assert_allclose(_world_coord_1d(scan_v2, "x"), expected)

    def test_depth_coord_from_zero(self, scan_v2: xr.DataArray) -> None:
        """Depth (y) coordinate starts at zero when no depth range is in the header."""
        expected = np.arange(_SIZE_Z) * _DZ_M * 1e3
        np.testing.assert_allclose(_world_coord_1d(scan_v2, "y"), expected)

    def test_depth_origin_recovered(self, tmp_path: Path) -> None:
        """Depth (y) origin is recovered from an embedded depth-range pair."""
        path = tmp_path / "scan_v2_depth.scan"
        _write_scan_v2(path, _raw_payload(), depth_start=1.0)
        da = load_scan(path)
        expected = 1.0 + np.arange(_SIZE_Z) * _DZ_M * 1e3
        np.testing.assert_allclose(_world_coord_1d(da, "y"), expected)

    def test_single_depth_voxel_zero_origin(self, tmp_path: Path) -> None:
        """A single-depth-voxel file has no span to match, so the origin is zero."""
        path = tmp_path / "scan_v2_depth1.scan"
        _write_scan_v2(path, _raw_payload(size_z=1), size_z=1)
        da = load_scan(path)
        np.testing.assert_array_equal(_world_coord_1d(da, "y"), [0.0])

    def test_spatial_units_mm(self, scan_v2: xr.DataArray) -> None:
        """Spatial coordinates are in mm."""
        for dim in ("x", "y", "z"):
            assert scan_v2.coords[dim].attrs["units"] == "mm"

    def test_no_probe_to_lab(self, scan_v2: xr.DataArray) -> None:
        """v2 carries an empty affines dict (no probe_to_lab yet)."""
        assert scan_v2.attrs["affines"] == {}

    def test_scan_format_attr(self, scan_v2: xr.DataArray) -> None:
        """v2 records its on-disk format."""
        assert scan_v2.attrs["iconeus_scan_format"] == "v2"

    def test_scan_mode_attr(self, scan_v2: xr.DataArray) -> None:
        """Single-pose v2 is reported as 2Dscan."""
        assert scan_v2.attrs["iconeus_scan_mode"] == "2Dscan"

    def test_provenance_fields_mapped(self, scan_v2: xr.DataArray) -> None:
        """Header strings map to v1-style provenance fields."""
        assert scan_v2.attrs["iconeus_project"] == "proj-01"
        assert scan_v2.attrs["iconeus_subject"] == "sub-01"
        assert scan_v2.attrs["iconeus_session"] == "ses-01"
        assert scan_v2.attrs["iconeus_scan"] == "scan-01"
        assert scan_v2.attrs["iconeus_experimenter"] == "user-01"
        assert scan_v2.attrs["iconeus_species"] == "Rat"

    def test_acquisition_datetime_recovered(self, tmp_path: Path) -> None:
        """Acquisition timestamp is recovered as a full ISO 8601 UTC datetime."""
        path = tmp_path / "scan_v2_ts.scan"
        # 1784005200 == 2026-07-14T05:00:00+00:00.
        _write_scan_v2(path, _raw_payload(), timestamp=1784005200)
        da = load_scan(path)
        assert da.attrs["iconeus_datetime"] == "2026-07-14T05:00:00+00:00"

    def test_name_from_scan_tag(self, scan_v2: xr.DataArray) -> None:
        """v2 DataArray name is taken from the recovered scan tag."""
        assert scan_v2.name == "scan-01"


class TestLoadScanV2Multipose:
    """Tests for multi-pose SCAN v2 files."""

    def test_loads_with_pose_affines(self, scan_v2_multipose_path: Path) -> None:
        """Multi-pose v2 files carry a pose dim and one affine per pose."""
        da = load_scan(scan_v2_multipose_path)
        assert da.dims == ("time", "pose", "k", "j", "i")
        assert da.shape == (_N_TIME, 2, 2, _SIZE_Z, _SIZE_X)
        assert get_voxel_to_world_affine(da).shape == (2, 4, 4)

    def test_static_multipose_has_no_time_dim(self, tmp_path: Path) -> None:
        """A single-time multi-pose v2 file loads as static 3Dscan data."""
        path = tmp_path / "scan_v2_static_multipose.scan"
        _write_scan_v2(
            path,
            _raw_payload(n_time=1, size_y=2, npose=2),
            n_time=1,
            size_y=2,
            npose=2,
        )
        da = load_scan(path)
        assert da.dims == ("pose", "k", "j", "i")
        assert da.attrs["iconeus_scan_mode"] == "3Dscan"


class TestLoadScanV2Multiblock:
    """Tests for dim6 payloads."""

    def test_shape_preserves_dim6(self, scan_v2_multiblock_path: Path) -> None:
        """The SCAN v2 dim6 axis is preserved as an extra VoxelData dimension."""
        da = load_scan(scan_v2_multiblock_path)
        assert da.dims == ("dim6", "time", "k", "j", "i")
        assert da.shape == (2, _N_TIME, _SIZE_Y, _SIZE_Z, _SIZE_X)

    def test_values(self, scan_v2_multiblock_path: Path) -> None:
        """Folded values match the transposed/reshaped payload."""
        da = load_scan(scan_v2_multiblock_path)
        expected = _expected_confusius(_raw_payload(nblock_repeat=2))
        np.testing.assert_array_equal(da.values, expected)

    def test_time_coord(self, scan_v2_multiblock_path: Path) -> None:
        """Dim6 does not alter the measured volume times."""
        da = load_scan(scan_v2_multiblock_path)
        np.testing.assert_allclose(
            da.coords["time"].values, _DT * (np.arange(_N_TIME) + 1)
        )
        assert da.coords["time"].attrs["volume_acquisition_reference"] == "end"
        assert da.coords["time"].attrs["volume_acquisition_duration"] == pytest.approx(
            _DT
        )

    @pytest.mark.parametrize("intent", [1, 3])
    def test_marker_dim6_intents_load(self, tmp_path: Path, intent: int) -> None:
        """Enhanced/brain-masked Doppler dim6 intents skip their fixed payload."""
        path = tmp_path / "scan_v2_marker_dim6.scan"
        _write_scan_v2(path, _raw_payload(), dim6_intents=[intent])
        da = load_scan(path)
        assert da.dims == ("time", "k", "j", "i")

    def test_velocity_band_dim6_attrs(self, tmp_path: Path) -> None:
        """Velocity-band dim6 metadata is exposed as attrs."""
        path = tmp_path / "scan_v2_velocity_dim6.scan"
        _write_scan_v2(path, _raw_payload(), dim6_intents=[2])
        da = load_scan(path)
        assert da.attrs["velocity_min"] == pytest.approx(-1.5)
        assert da.attrs["velocity_max"] == pytest.approx(2.5)


class TestLoadScanV2Acquisition:
    """Tests for BIDS-corresponding acquisition metadata."""

    @pytest.fixture
    def scan_v2_acq(self, tmp_path: Path) -> xr.DataArray:
        """Loaded v2 DataArray with a full acquisition block."""
        path = tmp_path / "scan_v2_acq.scan"
        _write_scan_v2(path, _raw_payload(), acquisition=True)
        return load_scan(path)

    def test_probe_fields(self, scan_v2_acq: xr.DataArray) -> None:
        """Probe model, center frequency, pitch, and focal depth are recovered."""
        assert scan_v2_acq.attrs["probe_model"] == _PROBE_MODEL
        assert scan_v2_acq.attrs["probe_center_frequency"] == pytest.approx(
            _CENTER_FREQ
        )
        assert scan_v2_acq.attrs["probe_pitch"] == pytest.approx(_PITCH)
        assert scan_v2_acq.attrs["probe_elevation_aperture"] == pytest.approx(1.5)

    def test_transmit_distinct_from_center(self, scan_v2_acq: xr.DataArray) -> None:
        """Transmit frequency is read from its own field, distinct from center freq."""
        assert scan_v2_acq.attrs["transmit_frequency"] == pytest.approx(_TRANSMIT_FREQ)
        assert scan_v2_acq.attrs["transmit_frequency"] != pytest.approx(_CENTER_FREQ)

    def test_sequence_fields(self, scan_v2_acq: xr.DataArray) -> None:
        """PRF, plane-wave angles, and imaging depth are recovered."""
        assert scan_v2_acq.attrs["pulse_repetition_frequency"] == pytest.approx(_PRF)
        np.testing.assert_allclose(scan_v2_acq.attrs["plane_wave_angles"], _ANGLES)
        np.testing.assert_allclose(
            scan_v2_acq.attrs["imaging_depth"],
            (_ACQ_DEPTH_START, _ACQ_DEPTH_START + (_SIZE_Z - 1) * _DZ_M * 1e3),
        )

    def test_filter_fields(self, scan_v2_acq: xr.DataArray) -> None:
        """SVD low cutoff and power-Doppler window come from fixed offsets."""
        assert scan_v2_acq.attrs["svd_low_cutoff"] == _SVD_CUTOFF
        assert scan_v2_acq.attrs["power_doppler_integration_duration"] == _DT

    def test_probe_to_lab_from_pose(self, scan_v2_acq: xr.DataArray) -> None:
        """probe_to_lab (folded into the primary affine) is built from the 6DOF pose.

        probe_to_lab is no longer exposed separately in attrs (ConfUSIus world
        coordinates for SCAN data are already lab space -- it is folded into the
        primary voxel-to-world affine at construction). This reconstructs the known
        local (pre-fold) affine from the fixture's own voxel-size/depth-origin
        constants and checks probe_to_lab @ local == the loaded primary affine.
        """
        tx, ty, tz, _, _, rz = _POSE
        cz, sz = math.cos(rz), math.sin(rz)
        probe_to_lab = np.eye(4)
        probe_to_lab[:3, :3] = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
        probe_to_lab[:3, 3] = (tx, ty, tz)
        perm = np.asarray(WORLD_TO_PROBE_PERMUTATION)
        probe_to_lab = perm.T @ probe_to_lab @ perm
        probe_to_lab[:3, 3] *= 1e3

        local_affine = np.eye(4)
        local_affine[:3, :3] = np.diag([_DY_M * 1e3, _DZ_M * 1e3, _DX_M * 1e3])
        local_affine[:3, 3] = [
            -((_SIZE_Y - 1) / 2) * _DY_M * 1e3,
            _ACQ_DEPTH_START,
            -((_SIZE_X - 1) / 2) * _DX_M * 1e3,
        ]

        np.testing.assert_allclose(
            get_voxel_to_world_affine(scan_v2_acq),
            probe_to_lab @ local_affine,
            atol=1e-9,
        )
        assert "probe_to_lab" not in scan_v2_acq.attrs.get("affines", {})

    def test_implausible_pose_raises(self, tmp_path: Path) -> None:
        """An implausible v2 pose fails loading instead of using probe geometry."""
        path = tmp_path / "scan_v2_badpose.scan"
        _write_scan_v2(
            path, _raw_payload(), acquisition=True, corrupt_acquisition="pose"
        )
        with pytest.raises(ValueError, match="probe-pose block"):
            load_scan(path)


class TestLoadScanV2WithBPS:
    """Tests for BPS composition on v2 files."""

    @pytest.fixture
    def scan_v2_acq_path(self, tmp_path: Path) -> Path:
        """Path to a v2 file with a valid acquisition block (so an affine exists)."""
        path = tmp_path / "scan_v2_acq_bps.scan"
        _write_scan_v2(path, _raw_payload(), acquisition=True)
        return path

    def test_world_to_brain_composed(
        self, scan_v2_acq_path: Path, bps_path: Path
    ) -> None:
        """bps_path adds a world_to_brain affine mapping world (already lab) to brain.

        ConfUSIus world coordinates are already lab space (probe_to_lab is folded
        into the primary voxel-to-world affine at construction), so world_to_brain
        no longer composes with a separately stored probe_to_lab.
        """
        da = load_scan(scan_v2_acq_path, bps_path=bps_path)
        expected = np.linalg.inv(load_bps(bps_path))
        np.testing.assert_allclose(
            da.attrs["affines"]["world_to_brain"], expected, atol=1e-12
        )

    def test_implausible_pose_rejected_before_bps(
        self, tmp_path: Path, bps_path: Path
    ) -> None:
        """v2 files with implausible pose fail loading, with or without bps_path."""
        path = tmp_path / "scan_v2_noaffine.scan"
        _write_scan_v2(
            path, _raw_payload(), acquisition=True, corrupt_acquisition="pose"
        )
        with pytest.raises(ValueError, match="probe-pose block"):
            load_scan(path, bps_path=bps_path)


class TestLoadScanV2Errors:
    """Tests for v2 error handling and dispatch."""

    def test_unrecognized_format_raises(self, tmp_path: Path) -> None:
        """A non-HDF5 file without the SCAN magic raises a descriptive error."""
        path = tmp_path / "mystery.scan"
        path.write_bytes(b"XXXX not a scan file")
        with pytest.raises(ValueError, match="not a SCAN file recognized"):
            load_scan(path)

    def test_payload_size_mismatch_raises(self, tmp_path: Path) -> None:
        """A payload-size field inconsistent with the dimensions raises."""
        path = tmp_path / "bad_size.scan"
        _write_scan_v2(path, _raw_payload(), payload_bytes_override=12345)
        with pytest.raises(ValueError, match="does not match the product"):
            load_scan(path)

    def test_truncated_header_raises(self, tmp_path: Path) -> None:
        """An implausibly large n_time (header too short for it) raises."""
        path = tmp_path / "truncated.scan"
        _write_scan_v2(path, _raw_payload())
        _patch_u64(path, _SCAN_V2_OFFSETS["n_time"], 10_000_000)
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        # The wrapped parse error carries the underlying cause message.
        assert "implausible timing/pose block" in str(excinfo.value)

    def test_nonpositive_dimension_raises(self, tmp_path: Path) -> None:
        """A zero dimension in the header raises."""
        path = tmp_path / "zero_dim.scan"
        _write_scan_v2(path, _raw_payload())
        _patch_u64(path, _SCAN_V2_OFFSETS["size_x"], 0)
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "non-positive" in str(excinfo.value)

    def test_short_header_raises(self, tmp_path: Path) -> None:
        """A header shorter than the dimension block raises."""
        path = tmp_path / "short_header.scan"
        header = bytearray(100)
        header[: len(SCAN_V2_MAGIC)] = SCAN_V2_MAGIC
        struct.pack_into("<Q", header, _SCAN_V2_OFFSETS["total_header_bytes"], 100)
        path.write_bytes(header)
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "truncated before the dimension block" in str(excinfo.value)

    def test_nonpositive_dim6_count_raises(self, tmp_path: Path) -> None:
        """A zero dim6 count in the header raises."""
        path = tmp_path / "zero_dim6.scan"
        _write_scan_v2(path, _raw_payload())
        _patch_u64(path, _SCAN_V2_OFFSETS["dim6_count"], 0)
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "non-positive dim6" in str(excinfo.value)

    def test_unsupported_dim6_intent_raises(self, tmp_path: Path) -> None:
        """Unsupported dim6 intents raise a descriptive parse error."""
        path = tmp_path / "bad_dim6.scan"
        _write_scan_v2(path, _raw_payload(), dim6_intents=[99])
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "Unsupported SCAN v2 dim6 intent" in str(excinfo.value)

    def test_truncated_angle_block_raises(self, tmp_path: Path) -> None:
        """An angle count that points beyond the header raises."""
        path = tmp_path / "bad_angles.scan"
        _write_scan_v2(path, _raw_payload(), corrupt_acquisition="angles")
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "plane-wave angle block" in str(excinfo.value)

    def test_truncated_string_field_raises(self, tmp_path: Path) -> None:
        """A string length that points beyond the header raises."""
        path = tmp_path / "bad_string.scan"
        _write_scan_v2(path, _raw_payload())
        data = bytearray(path.read_bytes())
        text_offset = data.index(b"default sequence")
        struct.pack_into("<L", data, text_offset - 4, 99_999)
        path.write_bytes(data)
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "string field is truncated" in str(excinfo.value)

    def test_truncated_toggle_block_raises(self, tmp_path: Path) -> None:
        """A stimulation-toggle count that points beyond the header raises."""
        path = tmp_path / "bad_toggles.scan"
        _write_scan_v2(path, _raw_payload(), corrupt_acquisition="toggles")
        with pytest.raises(ValueError, match="could not be parsed") as excinfo:
            load_scan(path)
        assert "stimulation-toggle block" in str(excinfo.value)
