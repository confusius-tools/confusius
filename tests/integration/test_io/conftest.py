"""Shared fixtures for integration tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest


def _create_autc_dat_file(
    path: Path,
    n_blocks: int = 2,
    n_x: int = 4,
    n_z: int = 6,
    n_frames: int = 3,
    start_index: int = 1,
) -> None:
    """Create a synthetic AUTC DAT file for testing.

    Generates deterministic complex data for reproducible tests:
    - Block 0: values 1+1j, 2+2j, ..., n_frames+n_frames*j
    - Block 1: values 101+101j, 102+102j, ...
    - Block 2: values 201+201j, 202+202j, ...

    Parameters
    ----------
    path : Path
        Path to create the DAT file.
    n_blocks : int, default: 2
        Number of blocks in the file.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of depth samples.
    n_frames : int, default: 3
        Number of frames per block.
    start_index : int, default: 1
        Starting acquisition block index (1-based in file).
    """
    header_dtype = np.dtype(
        [
            ("acquisition_block_index", np.int32),
            ("n_data_items", np.int32),
            ("n_z", np.int32),
            ("n_x", np.int32),
            ("n_frames", np.int32),
            ("field5", np.int32),
            ("field6", np.int32),
            ("field7", np.int32),
            ("field8", np.int32),
            ("field9", np.int32),
        ]
    )

    n_data_items = n_x * n_z * n_frames

    with open(path, "wb") as f:
        for block_idx in range(n_blocks):
            acquisition_index = start_index + block_idx * 4

            header = np.array(
                [
                    (
                        acquisition_index,
                        n_data_items,
                        n_z,
                        n_x,
                        n_frames,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                ],
                dtype=header_dtype,
            )

            base_value = block_idx * 100 + 1
            data = np.zeros((n_x, n_z, n_frames), dtype=np.complex64)
            for frame in range(n_frames):
                complex_val = base_value + frame
                data[:, :, frame] = complex(complex_val, complex_val)

            f.write(header.tobytes())
            f.write(data.tobytes())


def _create_echoframe_mat_metadata(
    path: Path,
    n_x: int = 4,
    n_z: int = 6,
    n_volumes: int = 3,
    crop: bool = False,
    cropping_roi: list | None = None,
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
        # 200 µs transmit-receive time -> PRF = 5000 Hz, CSF = 1000 Hz (5 angles).
        receive_spec["transmitReceiveTimeMus"] = np.array([200.0])

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
) -> None:
    """Create a synthetic EchoFrame DAT file.

    Generates deterministic complex data for reproducible tests:
    - Block 0: values 1+1j, 2+2j, ..., n_volumes+n_volumes*j
    - Block 1: values 101+101j, 102+102j, ...
    - Block 2: values 201+201j, 202+202j, ...

    Parameters
    ----------
    path : Path
        Path to create the DAT file.
    n_blocks : int, default: 2
        Number of blocks in the file.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of depth samples.
    n_volumes : int, default: 3
        Number of volumes per block.
    padding_bytes : int, default: 0
        Padding bytes per block (0 for cleaned files).
    """
    header_dtype = np.uint64
    n_header_items = 5
    data_size = n_x * n_volumes * n_z

    with open(path, "wb") as f:
        header = np.array(
            [
                1,
                n_header_items * 8,
                n_blocks,
                data_size,
                padding_bytes,
            ],
            dtype=header_dtype,
        )
        header.tofile(f)

        for block_idx in range(n_blocks):
            base_value = block_idx * 100 + 1
            data = np.zeros((n_volumes, n_x, 1, n_z), dtype=np.complex64)
            for volume in range(n_volumes):
                complex_val = base_value + volume
                data[volume, :, :, :] = complex(complex_val, complex_val)
            data.tofile(f)

            if padding_bytes > 0:
                padding = np.zeros(padding_bytes, dtype=np.uint8)
                padding.tofile(f)


@pytest.fixture
def synthetic_autc_session(tmp_path):
    """Create a synthetic AUTC session with deterministic data."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    file1 = session_dir / "bf0part000.dat"
    file2 = session_dir / "bf0part001.dat"

    _create_autc_dat_file(file1, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=1)
    _create_autc_dat_file(file2, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=9)

    return session_dir


@pytest.fixture
def synthetic_echoframe_session(tmp_path):
    """Create a synthetic EchoFrame session with deterministic data."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    dat_path = session_dir / "fUSi_BF.dat"
    meta_path = session_dir / "ScanParameters.mat"

    _create_echoframe_dat_file(
        dat_path, n_blocks=2, n_x=4, n_z=6, n_volumes=3, padding_bytes=0
    )
    _create_echoframe_mat_metadata(meta_path, n_x=4, n_z=6, n_volumes=3)

    return session_dir


@pytest.fixture
def synthetic_echoframe_session_cropped(tmp_path):
    """Create synthetic EchoFrame session with cropping enabled."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    dat_path = session_dir / "fUSi_BF.dat"
    meta_path = session_dir / "ScanParameters.mat"

    _create_echoframe_dat_file(
        dat_path, n_blocks=1, n_x=6, n_z=6, n_volumes=3, padding_bytes=0
    )
    _create_echoframe_mat_metadata(
        meta_path, n_x=6, n_z=6, n_volumes=3, crop=True, cropping_roi=[1, 4, 1, 3]
    )

    return session_dir


@pytest.fixture
def synthetic_echoframe_session_with_padding(tmp_path):
    """Create synthetic EchoFrame session with padding (uncleaned)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    dat_path = session_dir / "fUSi_BF.dat"
    meta_path = session_dir / "ScanParameters.mat"

    _create_echoframe_dat_file(
        dat_path, n_blocks=2, n_x=4, n_z=6, n_volumes=3, padding_bytes=16
    )
    _create_echoframe_mat_metadata(meta_path, n_x=4, n_z=6, n_volumes=3)

    return session_dir
