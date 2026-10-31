"""Shared fixtures for integration tests."""

from pathlib import Path

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
