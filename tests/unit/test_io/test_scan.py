"""Unit tests for confusius.io.scan module."""

from pathlib import Path

import dask.array as dask_array
import h5py
import numpy as np
import pytest
import xarray as xr

from confusius.io.scan import load_scan

_RNG = np.random.default_rng(42)

_SIZE_X = 8
_SIZE_Y = 1
_SIZE_Z = 12
_NPOSE = 3
_T = 5
_FRAME_DURATION = 0.05
_POSE_DURATION = 0.1
_DX = 0.0003
_DY = 0.0004
_DZ = 0.0002
_OX = -_DX * _SIZE_X / 2
_OY = 0.0
_OZ = -0.003

_VOXELS_TO_PROBE = np.array(
    [
        [_DX, 0.0, 0.0, _OX],
        [0.0, _DY, 0.0, _OY],
        [0.0, 0.0, -_DZ, _OZ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_PROBE_TO_LAB_SINGLE = np.eye(4, dtype=np.float64)
_PROBE_TO_LAB_SINGLE[:3, 3] = [0.001, 0.002, 0.003]

_PROBE_TO_LAB_MULTI = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array([[0, 0, 0, i * 0.001], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        for i in range(_NPOSE)
    ]
)

_PROBE_TO_LAB_ROTATED = np.array(
    [
        [0.0, 0.0, 1.0, 0.005],
        [0.0, 1.0, 0.0, 0.002],
        [-1.0, 0.0, 0.0, 0.003],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# BrainToLab: identity rotation, translation in metres on the Iconeus lab side
# (xyz = lateral, elevation, axial). Used to build a synthetic .bps fixture.
_BRAIN_TO_LAB = np.eye(4, dtype=np.float64)
_BRAIN_TO_LAB[:3, 3] = [0.010, 0.020, 0.030]


@pytest.fixture
def bps_path(tmp_path: Path) -> Path:
    """Create a synthetic BPS HDF5 file with a known BrainToLab affine."""
    path = tmp_path / "test.bps"
    with h5py.File(path, "w") as f:
        f.create_dataset("BrainToLab", data=_BRAIN_TO_LAB)
    return path


def _end_referenced_times(n: int, duration: float) -> np.ndarray:
    """Return regularly spaced end-referenced timestamps."""
    return duration + np.arange(n, dtype=np.float64) * duration


def _write_scan_metadata(
    f: h5py.File,
    mode: str,
    size_x: int,
    size_y: int,
    size_z: int,
    npose: int,
    nscan_repeat: int | None,
    nblock_repeat: int | None,
    voxels_to_probe: np.ndarray,
    probe_to_lab: np.ndarray,
    time_data: np.ndarray,
) -> None:
    """Write common metadata groups to a synthetic SCAN HDF5 file."""
    acq = f.create_group("acqMetaData")
    acq.create_dataset(
        "acquisitionMode", data=np.array([[mode.encode()]], dtype=object)
    )

    img = acq.create_group("imgDim")
    img.create_dataset("sizeX", data=np.array([[float(size_x)]]))
    img.create_dataset("sizeY", data=np.array([[float(size_y)]]))
    img.create_dataset("sizeZ", data=np.array([[float(size_z)]]))
    img.create_dataset("npose", data=np.array([[float(npose)]]))
    img.create_dataset(
        "nscanRepeat", data=np.array([[float(nscan_repeat if nscan_repeat else 1)]])
    )
    img.create_dataset(
        "nblockRepeat",
        data=np.array([[float(nblock_repeat if nblock_repeat else 1)]]),
    )
    img.create_dataset("dim7", data=np.array([[1.0]]))

    vox = acq.create_group("voxDim")
    vox.create_dataset("dx", data=np.array([[_DX]]))
    vox.create_dataset("dy", data=np.array([[_DY]]))
    vox.create_dataset("dz", data=np.array([[_DZ]]))
    vox.create_dataset("dt", data=np.array([[0.05]]))

    acq.create_dataset("time", data=time_data)
    acq.create_dataset("timeOriginal", data=time_data)
    acq.create_dataset("probeToLab", data=probe_to_lab)
    acq.create_dataset("voxelsToProbe", data=voxels_to_probe)

    meta = f.create_group("scanMetaData")
    for key, val in [
        ("Subject_tag", "sub-01"),
        ("Session_tag", "ses-01"),
        ("Scan_tag", "scan-01"),
        ("Project_tag", "proj-01"),
        ("Date", "2025-01-01"),
        ("Neuroscan_version", "1.0"),
        ("Machine_SN", "SN-0001"),
        ("User_name", "user"),
        ("Type", "source "),
        ("Code", " neuroSoft"),
        ("Comment", ""),
        ("Tag", ""),
    ]:
        meta.create_dataset(key, data=np.array([[val.encode()]], dtype=object))


# ---------------------------------------------------------------------------
# Tests: 2Dscan
# ---------------------------------------------------------------------------


class TestLoadScan2D:
    """Tests for load_scan with 2Dscan files."""

    def test_dims(self, scan_2d: xr.DataArray) -> None:
        """2Dscan produces DataArray with dims (time, z, y, x)."""
        assert scan_2d.dims == ("time", "z", "y", "x")

    def test_shape(self, scan_2d: xr.DataArray) -> None:
        """2Dscan shape matches (T, sizeY, sizeZ, sizeX)."""
        assert scan_2d.shape == (_T, _SIZE_Y, _SIZE_Z, _SIZE_X)

    def test_dtype_float64(self, scan_2d: xr.DataArray) -> None:
        """2Dscan data is float64 — no silent downcasting."""
        assert scan_2d.dtype == np.float64

    def test_lazy(self, scan_2d: xr.DataArray) -> None:
        """2Dscan returns a lazy Dask-backed DataArray."""
        assert isinstance(scan_2d.data, dask_array.Array)

    def test_time_coord(self, scan_2d: xr.DataArray) -> None:
        """2Dscan time coordinate is 1D, length T, units seconds."""
        assert "time" in scan_2d.coords
        assert scan_2d.coords["time"].dims == ("time",)
        assert len(scan_2d.coords["time"]) == _T
        assert scan_2d.coords["time"].attrs.get("units") == "s"
        assert scan_2d.coords["time"].attrs.get("volume_acquisition_reference") == "end"

    def test_spatial_coords_present(self, scan_2d: xr.DataArray) -> None:
        """2Dscan has x, y, z spatial coordinates."""
        for dim in ("x", "y", "z"):
            assert dim in scan_2d.coords

    def test_spatial_coords_units_mm(self, scan_2d: xr.DataArray) -> None:
        """Spatial coordinates have units='mm'."""
        for dim in ("x", "y", "z"):
            assert scan_2d.coords[dim].attrs.get("units") == "mm"

    def test_spatial_coords_voxdim(self, scan_2d: xr.DataArray) -> None:
        """Spatial coordinates carry voxdim attribute in mm."""
        np.testing.assert_allclose(
            scan_2d.coords["x"].attrs["voxdim"], 1e3 * _DX, rtol=1e-10
        )
        np.testing.assert_allclose(
            scan_2d.coords["y"].attrs["voxdim"], 1e3 * _DZ, rtol=1e-10
        )
        np.testing.assert_allclose(
            scan_2d.coords["z"].attrs["voxdim"], 1e3 * _DY, rtol=1e-10
        )

    def test_x_coord_values(self, scan_2d: xr.DataArray) -> None:
        """x coordinate matches expected lateral positions in mm (MATLAB 1-indexed)."""
        # voxelsToProbe is MATLAB-based (1-indexed), so we add 1 to zero-indexed Python.
        expected = 1e3 * (
            _VOXELS_TO_PROBE[0, 0] * (np.arange(_SIZE_X) + 1) + _VOXELS_TO_PROBE[0, 3]
        )
        np.testing.assert_allclose(scan_2d.coords["x"].values, expected, rtol=1e-10)

    def test_y_coord_values(self, scan_2d: xr.DataArray) -> None:
        """y coordinate matches expected depth positions (sign-flipped probe z)."""
        # voxelsToProbe is MATLAB-based (1-indexed), so we add 1 to zero-indexed Python.
        expected = 1e3 * (
            -(
                _VOXELS_TO_PROBE[2, 2] * (np.arange(_SIZE_Z) + 1)
                + _VOXELS_TO_PROBE[2, 3]
            )
        )
        np.testing.assert_allclose(scan_2d.coords["y"].values, expected, rtol=1e-10)

    def test_provenance_attrs(self, scan_2d: xr.DataArray) -> None:
        """2Dscan attrs contain all provenance fields with correct values."""
        # BIDS-compatible fields
        for key in ("device_serial_number", "software_version"):
            assert key in scan_2d.attrs
        # Iconeus-specific fields
        for key in (
            "iconeus_scan_mode",
            "iconeus_subject",
            "iconeus_session",
            "iconeus_scan",
            "iconeus_project",
            "iconeus_date",
        ):
            assert key in scan_2d.attrs
        # Spot-check a few values against the fixture to guard against empty-string bugs.
        assert scan_2d.attrs["iconeus_subject"] == "sub-01"
        assert scan_2d.attrs["iconeus_date"] == "2025-01-01"
        assert scan_2d.attrs["device_serial_number"] == "SN-0001"

    def test_name_from_scan_tag(self, scan_2d: xr.DataArray) -> None:
        """DataArray name is taken from iconeus_scan (Scan_tag)."""
        assert scan_2d.name == "scan-01"

    def test_scan_mode_attr(self, scan_2d: xr.DataArray) -> None:
        """iconeus_scan_mode attr equals '2Dscan'."""
        assert scan_2d.attrs["iconeus_scan_mode"] == "2Dscan"

    def test_physical_to_lab_shape(self, scan_2d: xr.DataArray) -> None:
        """physical_to_lab affine has shape (4, 4) for 2Dscan."""
        A = np.asarray(scan_2d.attrs["affines"]["physical_to_lab"])
        assert A.shape == (4, 4)

    def test_physical_to_lab_translation_in_mm(self, scan_2d: xr.DataArray) -> None:
        """physical_to_lab translation column is in mm and in ConfUSIus axis order."""
        A = np.asarray(scan_2d.attrs["affines"]["physical_to_lab"])
        # probeToLab translation is (x_lab, y_lab, z_lab) = (lateral, elevation, axial)
        # in metres. After P^T @ probeToLab @ P the translation is reordered to
        # ConfUSIus order (z_conf, y_conf, x_conf) = (elevation, -axial, lateral) and
        # scaled to mm.
        P = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        expected_t = (P.T @ _PROBE_TO_LAB_SINGLE)[:3, 3] * 1e3
        np.testing.assert_allclose(A[:3, 3], expected_t, rtol=1e-10)

    def test_data_values_preserved(self, scan_2d_path: Path) -> None:
        """Data values are unchanged after transpose (no silent modification)."""
        with h5py.File(scan_2d_path, "r") as f:
            raw = np.array(f["/Data"][()])
        expected = np.transpose(raw, [0, 2, 1, 3])
        da = load_scan(scan_2d_path)
        np.testing.assert_array_equal(da.values, expected)

    def test_no_pose_coord(self, scan_2d: xr.DataArray) -> None:
        """2Dscan has no pose coordinate."""
        assert "pose" not in scan_2d.coords

    def test_no_pose_time_coord(self, scan_2d: xr.DataArray) -> None:
        """2Dscan has no pose_time coordinate."""
        assert "pose_time" not in scan_2d.coords

    def test_linear_probe_z_singleton(self, scan_2d: xr.DataArray) -> None:
        """2Dscan with linear probe (sizeY=1) has a singleton z dimension."""
        assert scan_2d.sizes["z"] == _SIZE_Y == 1


class TestLoadScan2DTransposedTime:
    """Tests for load_scan with 2Dscan files whose time array has shape (T, 1)."""

    def test_time_coord_transposed_layout(
        self, scan_2d_transposed_time: xr.DataArray
    ) -> None:
        """2Dscan time coord is correct when time is stored as (T, 1) instead of (1, T)."""
        expected = _end_referenced_times(_T, _FRAME_DURATION)
        np.testing.assert_allclose(
            scan_2d_transposed_time.coords["time"].values, expected, rtol=1e-10
        )

    def test_time_coord_shape_transposed_layout(
        self, scan_2d_transposed_time: xr.DataArray
    ) -> None:
        """Time coordinate is 1D even when stored as (T, 1) in the file."""
        assert scan_2d_transposed_time.coords["time"].dims == ("time",)
        assert len(scan_2d_transposed_time.coords["time"]) == _T


# ---------------------------------------------------------------------------
# Tests: 3Dscan
# ---------------------------------------------------------------------------


class TestLoadScan3D:
    """Tests for load_scan with 3Dscan files."""

    def test_dims(self, scan_3d: xr.DataArray) -> None:
        """3Dscan produces DataArray with dims (pose, z, y, x)."""
        assert scan_3d.dims == ("pose", "z", "y", "x")

    def test_shape(self, scan_3d: xr.DataArray) -> None:
        """3Dscan shape matches (npose, sizeY, sizeZ, sizeX)."""
        assert scan_3d.shape == (_NPOSE, _SIZE_Y, _SIZE_Z, _SIZE_X)

    def test_dtype_float64(self, scan_3d: xr.DataArray) -> None:
        """3Dscan data is float64."""
        assert scan_3d.dtype == np.float64

    def test_lazy(self, scan_3d: xr.DataArray) -> None:
        """3Dscan returns a lazy Dask-backed DataArray."""
        assert isinstance(scan_3d.data, dask_array.Array)

    def test_pose_coord(self, scan_3d: xr.DataArray) -> None:
        """3Dscan has integer pose coordinate from 0 to npose-1."""
        assert "pose" in scan_3d.coords
        np.testing.assert_array_equal(scan_3d.coords["pose"].values, np.arange(_NPOSE))

    def test_no_time_coord(self, scan_3d: xr.DataArray) -> None:
        """3Dscan has no time coordinate."""
        assert "time" not in scan_3d.coords

    def test_physical_to_lab_shape(self, scan_3d: xr.DataArray) -> None:
        """physical_to_lab affine has shape (npose, 4, 4) for 3Dscan."""
        A = np.asarray(scan_3d.attrs["affines"]["physical_to_lab"])
        assert A.shape == (_NPOSE, 4, 4)

    def test_data_values_preserved(self, scan_3d_path: Path) -> None:
        """Data values are unchanged after squeeze + transpose."""
        with h5py.File(scan_3d_path, "r") as f:
            raw = np.array(f["/Data"][()])
        expected = np.transpose(raw.squeeze(axis=1), [0, 2, 1, 3])
        da = load_scan(scan_3d_path)
        np.testing.assert_array_equal(da.values, expected)

    def test_scan_mode_attr(self, scan_3d: xr.DataArray) -> None:
        """iconeus_scan_mode attr equals '3Dscan'."""
        assert scan_3d.attrs["iconeus_scan_mode"] == "3Dscan"

    def test_matrix_probe_z_dim(self, scan_3d_matrix: xr.DataArray) -> None:
        """3Dscan with matrix probe (sizeY=4) has z dimension size 4."""
        assert scan_3d_matrix.dims == ("pose", "z", "y", "x")
        assert scan_3d_matrix.sizes["z"] == 4

    def test_linear_probe_z_singleton(self, scan_3d: xr.DataArray) -> None:
        """3Dscan with linear probe (sizeY=1) has a singleton z dimension."""
        assert scan_3d.sizes["z"] == _SIZE_Y == 1


# ---------------------------------------------------------------------------
# Tests: 4Dscan
# ---------------------------------------------------------------------------


class TestLoadScan4D:
    """Tests for load_scan with 4Dscan files."""

    def test_dims(self, scan_4d: xr.DataArray) -> None:
        """4Dscan produces DataArray with dims (time, pose, z, y, x)."""
        assert scan_4d.dims == ("time", "pose", "z", "y", "x")

    def test_shape(self, scan_4d: xr.DataArray) -> None:
        """4Dscan shape matches (nscanRepeat, npose, sizeY, sizeZ, sizeX)."""
        assert scan_4d.shape == (_T, _NPOSE, _SIZE_Y, _SIZE_Z, _SIZE_X)

    def test_dtype_float64(self, scan_4d: xr.DataArray) -> None:
        """4Dscan data is float64."""
        assert scan_4d.dtype == np.float64

    def test_lazy(self, scan_4d: xr.DataArray) -> None:
        """4Dscan returns a lazy Dask-backed DataArray."""
        assert isinstance(scan_4d.data, dask_array.Array)

    def test_time_coord(self, scan_4d: xr.DataArray) -> None:
        """4Dscan time coordinate is 1D, length nscanRepeat, units seconds."""
        assert "time" in scan_4d.coords
        assert scan_4d.coords["time"].dims == ("time",)
        assert len(scan_4d.coords["time"]) == _T
        assert scan_4d.coords["time"].attrs.get("units") == "s"
        assert scan_4d.coords["time"].attrs.get("volume_acquisition_reference") == "end"

    def test_time_coord_duration_inferred_from_first_timestamp(
        self, scan_4d_offset_path: Path
    ) -> None:
        """4Dscan infers end-reference duration from the first block timestamp."""
        scan_4d = load_scan(scan_4d_offset_path)

        assert scan_4d.coords["time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.4)

    def test_time_coord_latest_per_block(self, scan_4d: xr.DataArray) -> None:
        """4Dscan time coordinate equals latest (end-referenced) timestamp per block."""
        time_flat = _end_referenced_times(_T * _NPOSE, _POSE_DURATION).reshape(
            _T * _NPOSE, 1
        )
        time_mat = time_flat.reshape(_T, _NPOSE)
        expected = time_mat.max(axis=1)
        np.testing.assert_allclose(scan_4d.coords["time"].values, expected, rtol=1e-10)

    def test_pose_coord(self, scan_4d: xr.DataArray) -> None:
        """4Dscan has integer pose coordinate from 0 to npose-1."""
        assert "pose" in scan_4d.coords
        np.testing.assert_array_equal(scan_4d.coords["pose"].values, np.arange(_NPOSE))

    def test_pose_time_coord_present(self, scan_4d: xr.DataArray) -> None:
        """4Dscan has a pose_time non-dimension coordinate."""
        assert "pose_time" in scan_4d.coords

    def test_pose_time_coord_shape(self, scan_4d: xr.DataArray) -> None:
        """pose_time has dims (time, pose) with shape (nscanRepeat, npose)."""
        pt = scan_4d.coords["pose_time"]
        assert pt.dims == ("time", "pose")
        assert pt.shape == (_T, _NPOSE)

    def test_pose_time_coord_units(self, scan_4d: xr.DataArray) -> None:
        """pose_time has units='s'."""
        assert scan_4d.coords["pose_time"].attrs.get("units") == "s"
        assert (
            scan_4d.coords["pose_time"].attrs.get("volume_acquisition_reference")
            == "end"
        )

    def test_pose_time_duration_inferred_from_first_timestamp(
        self, scan_4d_offset_path: Path
    ) -> None:
        """pose_time carries the same inferred timing metadata as time."""
        scan_4d = load_scan(scan_4d_offset_path)

        assert (
            scan_4d.coords["pose_time"].attrs["volume_acquisition_reference"] == "end"
        )
        assert scan_4d.coords["pose_time"].attrs[
            "volume_acquisition_duration"
        ] == pytest.approx(0.4)

    def test_pose_time_coord_values(self, scan_4d: xr.DataArray) -> None:
        """pose_time values match the reshaped raw time array."""
        time_flat = _end_referenced_times(_T * _NPOSE, _POSE_DURATION).reshape(
            _T * _NPOSE, 1
        )
        expected = time_flat.reshape(_T, _NPOSE)
        np.testing.assert_allclose(
            scan_4d.coords["pose_time"].values, expected, rtol=1e-10
        )

    def test_pose_time_isel_time(self, scan_4d: xr.DataArray) -> None:
        """isel on time slices pose_time correctly to shape (pose,)."""
        sliced = scan_4d.isel(time=0)
        assert sliced.coords["pose_time"].dims == ("pose",)
        assert sliced.coords["pose_time"].shape == (_NPOSE,)

    def test_pose_time_isel_pose(self, scan_4d: xr.DataArray) -> None:
        """isel on pose slices pose_time correctly to shape (time,)."""
        sliced = scan_4d.isel(pose=0)
        assert sliced.coords["pose_time"].dims == ("time",)
        assert sliced.coords["pose_time"].shape == (_T,)

    def test_physical_to_lab_shape(self, scan_4d: xr.DataArray) -> None:
        """physical_to_lab affine has shape (npose, 4, 4) for 4Dscan."""
        A = np.asarray(scan_4d.attrs["affines"]["physical_to_lab"])
        assert A.shape == (_NPOSE, 4, 4)

    def test_data_values_preserved(self, scan_4d_path: Path) -> None:
        """Data values are unchanged after squeeze + transpose."""
        with h5py.File(scan_4d_path, "r") as f:
            raw = np.array(f["/Data"][()])
        expected = np.transpose(raw.squeeze(axis=2), [0, 1, 3, 2, 4])
        da = load_scan(scan_4d_path)
        np.testing.assert_array_equal(da.values, expected)

    def test_scan_mode_attr(self, scan_4d: xr.DataArray) -> None:
        """iconeus_scan_mode attr equals '4Dscan'."""
        assert scan_4d.attrs["iconeus_scan_mode"] == "4Dscan"

    def test_multiblock_repeats_are_merged_into_time(
        self, scan_4d_multiblock_path: Path, scan_4d_multiblock: xr.DataArray
    ) -> None:
        """`nblockRepeat > 1` is folded into the output time dimension."""
        with h5py.File(scan_4d_multiblock_path, "r") as f:
            raw = np.array(f["/Data"][()])
        expected = np.transpose(raw, [0, 2, 1, 4, 3, 5]).reshape(
            6, _NPOSE, _SIZE_Y, _SIZE_Z, _SIZE_X
        )

        assert scan_4d_multiblock.shape == (6, _NPOSE, _SIZE_Y, _SIZE_Z, _SIZE_X)
        np.testing.assert_array_equal(scan_4d_multiblock.values, expected)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestLoadScanErrors:
    """Tests for error handling in load_scan."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """load_scan raises ValueError for a non-existent file."""
        with pytest.raises(ValueError):
            load_scan(tmp_path / "nonexistent.scan")

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        """load_scan raises ValueError for an unknown acquisitionMode."""
        path = tmp_path / "bad_mode.scan"
        time_data = np.array([[float(_FRAME_DURATION)]])
        data = _RNG.random((_T, _SIZE_Z, _SIZE_Y, _SIZE_X), dtype=np.float64)
        with h5py.File(path, "w") as f:
            f.create_dataset("/Data", data=data)
            _write_scan_metadata(
                f,
                mode="BadMode",
                size_x=_SIZE_X,
                size_y=_SIZE_Y,
                size_z=_SIZE_Z,
                npose=1,
                nscan_repeat=None,
                nblock_repeat=_T,
                voxels_to_probe=_VOXELS_TO_PROBE,
                probe_to_lab=_PROBE_TO_LAB_SINGLE,
                time_data=time_data,
            )

        with pytest.raises(ValueError, match="Unknown acquisitionMode"):
            load_scan(path)


# ---------------------------------------------------------------------------
# Tests: physical_to_lab correctness
# ---------------------------------------------------------------------------


class TestPhysicalToLab:
    """Tests for the physical_to_lab affine computation."""

    def test_physical_to_lab_maps_coords_to_lab(self, scan_2d: xr.DataArray) -> None:
        """physical_to_lab maps (z_mm, y_mm, x_mm) to ConfUSIus-ordered lab space (mm)."""
        A = np.asarray(scan_2d.attrs["affines"]["physical_to_lab"])

        # physical_to_lab = P^T @ probeToLab @ P, with output in ConfUSIus axis order.
        P = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        expected = P.T @ _PROBE_TO_LAB_SINGLE @ P
        expected[:3, 3] *= 1e3

        np.testing.assert_allclose(A, expected, rtol=1e-10)

    def test_physical_to_lab_per_pose_values(self, scan_3d: xr.DataArray) -> None:
        """Each per-pose physical_to_lab slice matches the corresponding probeToLab."""
        A = np.asarray(scan_3d.attrs["affines"]["physical_to_lab"])
        P = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        for i in range(_NPOSE):
            expected = P.T @ _PROBE_TO_LAB_MULTI[i] @ P
            expected[:3, 3] *= 1e3
            np.testing.assert_allclose(A[i], expected, rtol=1e-10)

    def test_physical_to_lab_non_identity_rotation(
        self, scan_2d_rotated: xr.DataArray
    ) -> None:
        """physical_to_lab correctly applies P^T @ R @ P for a non-trivial rotation.

        With a 90° Y-rotation in probeToLab, the rotation block of physical_to_lab
        must equal P^T @ R_y90 @ P, confirming that non-identity rotations are handled
        correctly and the output remains in ConfUSIus axis order.
        """
        A = np.asarray(scan_2d_rotated.attrs["affines"]["physical_to_lab"])

        P = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        expected = P.T @ _PROBE_TO_LAB_ROTATED @ P
        expected[:3, 3] *= 1e3

        np.testing.assert_allclose(A, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: BPS sidecar (physical_to_brain affine)
# ---------------------------------------------------------------------------


class TestLoadScanWithBPS:
    """Tests for load_scan when a BPS file is provided via `bps_path`."""

    def test_no_bps_omits_physical_to_brain(self, scan_2d: xr.DataArray) -> None:
        """Loading without `bps_path` does not add physical_to_brain to attrs."""
        assert "physical_to_brain" not in scan_2d.attrs["affines"]

    def test_bps_adds_physical_to_brain(
        self, scan_2d_path: Path, bps_path: Path
    ) -> None:
        """Loading with `bps_path` adds physical_to_brain alongside physical_to_lab."""
        da = load_scan(scan_2d_path, bps_path=bps_path)
        assert "physical_to_brain" in da.attrs["affines"]
        assert "physical_to_lab" in da.attrs["affines"]

    def test_physical_to_brain_shape_2d(
        self, scan_2d_path: Path, bps_path: Path
    ) -> None:
        """physical_to_brain has shape (4, 4) for 2Dscan."""
        da = load_scan(scan_2d_path, bps_path=bps_path)
        A = np.asarray(da.attrs["affines"]["physical_to_brain"])
        assert A.shape == (4, 4)

    def test_physical_to_brain_shape_multipose(
        self, scan_3d_path: Path, scan_4d_path: Path, bps_path: Path
    ) -> None:
        """physical_to_brain has shape (npose, 4, 4) for 3Dscan and 4Dscan."""
        for scan_path in (scan_3d_path, scan_4d_path):
            da = load_scan(scan_path, bps_path=bps_path)
            A = np.asarray(da.attrs["affines"]["physical_to_brain"])
            assert A.shape == (_NPOSE, 4, 4)

    def test_physical_to_brain_maps_origin_to_expected_brain_point(
        self, scan_2d_path: Path, bps_path: Path
    ) -> None:
        """Physical origin (0, 0, 0) mm maps to the analytically expected brain point.

        With BrainToLab = I + translation t (in metres, on the Iconeus xyz lab
        side), the chain unwinds as:

        - physical (z, y, x) = (0, 0, 0) mm -> ConfUSIus-ordered lab equals the
          translation column of physical_to_lab in mm.
        - Convert that lab point to Iconeus-ordered xyz metres, then subtract t to
          get the brain coordinate (BrainToLab is identity-plus-translation).
        """
        da = load_scan(scan_2d_path, bps_path=bps_path)
        A = np.asarray(da.attrs["affines"]["physical_to_brain"])

        # Lab point corresponding to physical origin, in ConfUSIus zyx mm.
        lab_zyx_mm = np.asarray(da.attrs["affines"]["physical_to_lab"])[:3, 3]
        # Convert to Iconeus xyz metres: x_lab=x_conf, y_lab=z_conf, z_lab=-y_conf.
        lab_xyz_m = (
            np.array([lab_zyx_mm[2], lab_zyx_mm[0], -lab_zyx_mm[1]], dtype=np.float64)
            * 1e-3
        )
        expected_brain = lab_xyz_m - _BRAIN_TO_LAB[:3, 3]

        result = A @ np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result[:3], expected_brain, rtol=1e-10, atol=1e-12)

    def test_missing_bps_path_raises(self, scan_2d_path: Path, tmp_path: Path) -> None:
        """A non-existent `bps_path` raises ValueError before opening the SCAN."""
        with pytest.raises(ValueError):
            load_scan(scan_2d_path, bps_path=tmp_path / "nonexistent.bps")
