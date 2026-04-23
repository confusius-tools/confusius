"""Shared fixtures for unit tests."""

from pathlib import Path

import h5py
import matplotlib
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from confusius.io.scan import load_scan

_SCAN_RNG = np.random.default_rng(42)

_SCAN_SIZE_X = 8
_SCAN_SIZE_Y = 1
_SCAN_SIZE_Z = 12
_SCAN_NPOSE = 3
_SCAN_T = 5
_SCAN_FRAME_DURATION = 0.05
_SCAN_POSE_DURATION = 0.1
_SCAN_DX = 0.0003
_SCAN_DY = 0.0004
_SCAN_DZ = 0.0002
_SCAN_OX = -_SCAN_DX * _SCAN_SIZE_X / 2
_SCAN_OY = 0.0
_SCAN_OZ = -0.003

_SCAN_VOXELS_TO_PROBE = np.array(
    [
        [_SCAN_DX, 0.0, 0.0, _SCAN_OX],
        [0.0, _SCAN_DY, 0.0, _SCAN_OY],
        [0.0, 0.0, -_SCAN_DZ, _SCAN_OZ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_SCAN_PROBE_TO_LAB_SINGLE = np.eye(4, dtype=np.float64)
_SCAN_PROBE_TO_LAB_SINGLE[:3, 3] = [0.001, 0.002, 0.003]

_SCAN_PROBE_TO_LAB_MULTI = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array([[0, 0, 0, i * 0.001], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        for i in range(_SCAN_NPOSE)
    ]
)

_SCAN_PROBE_TO_LAB_ROTATED = np.array(
    [
        [0.0, 0.0, 1.0, 0.005],
        [0.0, 1.0, 0.0, 0.002],
        [-1.0, 0.0, 0.0, 0.003],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_SCAN_PROBE_TO_LAB_IRREGULAR = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array([[0, 0, 0, t], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        for t in [0.0, 0.001, 0.005]
    ]
)

_SCAN_ANGLE = np.pi / 6
_SCAN_PROBE_TO_LAB_VARYING_ROTATION = np.stack(
    [
        np.array(
            [
                [np.cos(i * _SCAN_ANGLE), 0, np.sin(i * _SCAN_ANGLE), i * 0.001],
                [0, 1, 0, 0],
                [-np.sin(i * _SCAN_ANGLE), 0, np.cos(i * _SCAN_ANGLE), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        for i in range(_SCAN_NPOSE)
    ]
)

_SCAN_PROBE_TO_LAB_2D_SWEEP = np.stack(
    [
        np.eye(4, dtype=np.float64)
        + np.array(
            [
                [0, 0, 0, i * 0.001],
                [0, 0, 0, i * 0.001],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        for i in range(_SCAN_NPOSE)
    ]
)


def _scan_end_referenced_times(n: int, duration: float) -> npt.NDArray[np.float64]:
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
    voxels_to_probe: npt.NDArray[np.float64],
    probe_to_lab: npt.NDArray[np.float64],
    time_data: npt.NDArray[np.float64],
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
    vox.create_dataset("dx", data=np.array([[_SCAN_DX]]))
    vox.create_dataset("dy", data=np.array([[_SCAN_DY]]))
    vox.create_dataset("dz", data=np.array([[_SCAN_DZ]]))
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


@pytest.fixture
def scan_2d_path(tmp_path: Path) -> Path:
    """Create a synthetic 2Dscan HDF5 file."""
    path = tmp_path / "test_2dscan.scan"
    time_data = _scan_end_referenced_times(_SCAN_T, _SCAN_FRAME_DURATION).reshape(
        1, _SCAN_T
    )
    data = _SCAN_RNG.random(
        (_SCAN_T, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X), dtype=np.float64
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="2Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=1,
            nscan_repeat=None,
            nblock_repeat=_SCAN_T,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_SINGLE,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_2d_transposed_time_path(tmp_path: Path) -> Path:
    """Create a synthetic 2Dscan HDF5 file with transposed time layout."""
    path = tmp_path / "test_2dscan_transposed_time.scan"
    time_data = _scan_end_referenced_times(_SCAN_T, _SCAN_FRAME_DURATION).reshape(
        _SCAN_T, 1
    )
    data = _SCAN_RNG.random(
        (_SCAN_T, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X), dtype=np.float64
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="2Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=1,
            nscan_repeat=None,
            nblock_repeat=_SCAN_T,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_SINGLE,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file."""
    path = tmp_path / "test_3dscan.scan"
    time_data = _scan_end_referenced_times(_SCAN_NPOSE, _SCAN_POSE_DURATION).reshape(
        _SCAN_NPOSE, 1
    )
    data = _SCAN_RNG.random(
        (_SCAN_NPOSE, 1, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_MULTI,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_4d_path(tmp_path: Path) -> Path:
    """Create a synthetic 4Dscan HDF5 file."""
    path = tmp_path / "test_4dscan.scan"
    time_flat = _scan_end_referenced_times(
        _SCAN_T * _SCAN_NPOSE, _SCAN_POSE_DURATION
    ).reshape(_SCAN_T * _SCAN_NPOSE, 1)
    data = _SCAN_RNG.random(
        (_SCAN_T, _SCAN_NPOSE, 1, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="4Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=_SCAN_T,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_MULTI,
            time_data=time_flat,
        )
    return path


@pytest.fixture
def scan_4d_offset_path(tmp_path: Path) -> Path:
    """Create a synthetic 4Dscan file with a larger non-zero first timestamp."""
    path = tmp_path / "test_4dscan_offset.scan"
    time_flat = (
        0.4
        + np.arange(_SCAN_T * _SCAN_NPOSE, dtype=np.float64).reshape(
            _SCAN_T * _SCAN_NPOSE, 1
        )
        * 0.6
    )
    data = _SCAN_RNG.random(
        (_SCAN_T, _SCAN_NPOSE, 1, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="4Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=_SCAN_T,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_MULTI,
            time_data=time_flat,
        )
    return path


@pytest.fixture
def scan_4d_multiblock_path(tmp_path: Path) -> Path:
    """Create a synthetic 4Dscan HDF5 file with repeated blocks."""
    path = tmp_path / "test_4dscan_multiblock.scan"
    nscan_repeat = 2
    nblock_repeat = 3
    n_time = nscan_repeat * nblock_repeat
    time_flat = _scan_end_referenced_times(
        n_time * _SCAN_NPOSE, _SCAN_POSE_DURATION
    ).reshape(n_time * _SCAN_NPOSE, 1)
    data = _SCAN_RNG.random(
        (
            nscan_repeat,
            _SCAN_NPOSE,
            nblock_repeat,
            _SCAN_SIZE_Z,
            _SCAN_SIZE_Y,
            _SCAN_SIZE_X,
        ),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="4Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=nscan_repeat,
            nblock_repeat=nblock_repeat,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_MULTI,
            time_data=time_flat,
        )
    return path


@pytest.fixture
def scan_2d_rotated_path(tmp_path: Path) -> Path:
    """Create a synthetic 2Dscan HDF5 file with a rotated probe transform."""
    path = tmp_path / "test_2dscan_rotated.scan"
    time_data = _scan_end_referenced_times(_SCAN_T, _SCAN_FRAME_DURATION).reshape(
        1, _SCAN_T
    )
    data = _SCAN_RNG.random(
        (_SCAN_T, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X), dtype=np.float64
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="2Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=1,
            nscan_repeat=None,
            nblock_repeat=_SCAN_T,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_ROTATED,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_matrix_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with a matrix probe."""
    size_y_matrix = 4
    path = tmp_path / "test_3dscan_matrix.scan"
    time_data = _scan_end_referenced_times(_SCAN_NPOSE, _SCAN_POSE_DURATION).reshape(
        _SCAN_NPOSE, 1
    )
    data = _SCAN_RNG.random(
        (_SCAN_NPOSE, 1, _SCAN_SIZE_Z, size_y_matrix, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=size_y_matrix,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_MULTI,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_irregular_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with irregular pose spacing."""
    path = tmp_path / "test_3dscan_irregular.scan"
    time_data = _scan_end_referenced_times(_SCAN_NPOSE, _SCAN_POSE_DURATION).reshape(
        _SCAN_NPOSE, 1
    )
    data = _SCAN_RNG.random(
        (_SCAN_NPOSE, 1, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_IRREGULAR,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_varying_rotation_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with varying pose rotations."""
    path = tmp_path / "test_3dscan_varying_rotation.scan"
    time_data = _scan_end_referenced_times(_SCAN_NPOSE, _SCAN_POSE_DURATION).reshape(
        _SCAN_NPOSE, 1
    )
    data = _SCAN_RNG.random(
        (_SCAN_NPOSE, 1, _SCAN_SIZE_Z, _SCAN_SIZE_Y, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=_SCAN_SIZE_Y,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_VARYING_ROTATION,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_3d_2d_sweep_path(tmp_path: Path) -> Path:
    """Create a synthetic 3Dscan HDF5 file with a non-1D sweep."""
    size_y_matrix = 4
    path = tmp_path / "test_3dscan_2d_sweep.scan"
    time_data = _scan_end_referenced_times(_SCAN_NPOSE, _SCAN_POSE_DURATION).reshape(
        _SCAN_NPOSE, 1
    )
    data = _SCAN_RNG.random(
        (_SCAN_NPOSE, 1, _SCAN_SIZE_Z, size_y_matrix, _SCAN_SIZE_X),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("/Data", data=data)
        _write_scan_metadata(
            f,
            mode="3Dscan",
            size_x=_SCAN_SIZE_X,
            size_y=size_y_matrix,
            size_z=_SCAN_SIZE_Z,
            npose=_SCAN_NPOSE,
            nscan_repeat=None,
            nblock_repeat=None,
            voxels_to_probe=_SCAN_VOXELS_TO_PROBE,
            probe_to_lab=_SCAN_PROBE_TO_LAB_2D_SWEEP,
            time_data=time_data,
        )
    return path


@pytest.fixture
def scan_2d(scan_2d_path: Path) -> xr.DataArray:
    """Load a synthetic 2Dscan DataArray."""
    return load_scan(scan_2d_path)


@pytest.fixture
def scan_2d_transposed_time(scan_2d_transposed_time_path: Path) -> xr.DataArray:
    """Load a synthetic 2Dscan DataArray with transposed time layout."""
    return load_scan(scan_2d_transposed_time_path)


@pytest.fixture
def scan_2d_rotated(scan_2d_rotated_path: Path) -> xr.DataArray:
    """Load a synthetic 2Dscan DataArray with rotated probe transform."""
    return load_scan(scan_2d_rotated_path)


@pytest.fixture
def scan_3d(scan_3d_path: Path) -> xr.DataArray:
    """Load a synthetic 3Dscan DataArray."""
    return load_scan(scan_3d_path)


@pytest.fixture
def scan_3d_matrix(scan_3d_matrix_path: Path) -> xr.DataArray:
    """Load a synthetic 3Dscan DataArray with a matrix probe."""
    return load_scan(scan_3d_matrix_path)


@pytest.fixture
def scan_4d(scan_4d_path: Path) -> xr.DataArray:
    """Load a synthetic 4Dscan DataArray."""
    return load_scan(scan_4d_path)


@pytest.fixture
def scan_4d_multiblock(scan_4d_multiblock_path: Path) -> xr.DataArray:
    """Load a synthetic 4Dscan DataArray with repeated blocks."""
    return load_scan(scan_4d_multiblock_path)


matplotlib.use("Agg", force=True)


@pytest.fixture
def sample_3d_volume(rng):
    """3D spatial volume (z, y, x) with consistent spatial coordinates.

    Shape: (4, 6, 8) - small enough for fast tests.
    Includes time as a scalar coordinate for consistency with 4D volumes.
    Includes name and metadata attributes for testing labels and units.
    """
    shape = (4, 6, 8)
    data = rng.random(shape)
    da = xr.DataArray(
        data,
        name="power_doppler",
        dims=["z", "y", "x"],
        coords={
            "z": xr.DataArray(
                1.0 + np.arange(4) * 0.2,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                2.0 + np.arange(6) * 0.1,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                3.0 + np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
            "time": 0.0,  # Scalar coord for consistency with 4D volumes.
        },
        attrs={
            "long_name": "Intensity",
            "units": "a.u.",
        },
    )
    return da


@pytest.fixture
def sample_4d_volume(rng):
    """4D volume (time, z, y, x) with consistent coordinates.

    Shape: (10, 4, 6, 8) - small enough for fast tests.
    Spatial coordinates match sample_3d_volume exactly.
    Includes name and metadata attributes for testing labels and units.
    """
    shape = (10, 4, 6, 8)
    data = rng.random(shape)
    da = xr.DataArray(
        data,
        name="power_doppler",
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                10.0 + np.arange(10) * 0.5,
                dims=["time"],
                attrs={"units": "s"},
            ),
            "z": xr.DataArray(
                1.0 + np.arange(4) * 0.2,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                2.0 + np.arange(6) * 0.1,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                3.0 + np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
        },
        attrs={
            "long_name": "Intensity",
            "units": "a.u.",
        },
    )
    return da


@pytest.fixture
def sample_4d_volume_complex(rng):
    """Complex-valued 4D volume (time, z, y, x) for IQ processing tests.

    Shape: (10, 4, 6, 8) - matches sample_4d_volume spatial dimensions.
    Includes name and metadata attributes for testing labels and units.
    """
    shape = (10, 4, 6, 8)
    data = rng.random(shape) + 1j * rng.random(shape)
    da = xr.DataArray(
        data,
        name="iq",
        dims=["time", "z", "y", "x"],
        coords={
            "time": xr.DataArray(
                np.arange(10) * 0.1,
                dims=["time"],
                attrs={"units": "s"},
            ),
            "z": xr.DataArray(
                np.arange(4) * 0.1,
                dims=["z"],
                attrs={"units": "mm"},
            ),
            "y": xr.DataArray(
                np.arange(6) * 0.05,
                dims=["y"],
                attrs={"units": "mm"},
            ),
            "x": xr.DataArray(
                np.arange(8) * 0.05,
                dims=["x"],
                attrs={"units": "mm"},
            ),
        },
        attrs={
            "long_name": "Complex Signal",
            "units": "a.u.",
        },
    )
    return da


@pytest.fixture
def sample_timeseries(rng):
    """Factory fixture for 2D time-series data (time, voxels).

    Creates DataArray with proper time coordinates.
    """

    def _make(
        n_time=100,
        n_voxels=50,
        sampling_rate=100.0,
    ):
        data = rng.normal(size=(n_time, n_voxels))
        return xr.DataArray(
            data,
            dims=["time", "space"],
            coords={
                "time": np.arange(n_time) / sampling_rate,
                "space": np.arange(n_voxels),
            },
        )

    return _make


@pytest.fixture
def spatial_mask(rng, sample_4d_volume):
    """Boolean spatial mask matching (z, y, x) of sample volumes."""
    _, z, y, x = sample_4d_volume.shape
    return rng.random((z, y, x)) > 0.5


@pytest.fixture
def matplotlib_pyplot():
    """Set up and teardown fixture for matplotlib.

    Returns the pyplot module and ensures all figures are closed after the test.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    yield plt
    plt.close("all")
