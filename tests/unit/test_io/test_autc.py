"""Unit tests for confusius.io.autc module."""

from pathlib import Path

import numpy as np
import pytest

from confusius.io.autc import AUTCDAT, AUTCDATsLoader


def _create_autc_dat_file(
    path: Path,
    n_blocks: int = 1,
    n_x: int = 4,
    n_z: int = 6,
    n_frames: int = 3,
    start_index: int = 1,
    known_pattern: bool = False,
) -> None:
    """Create a synthetic AUTC DAT file for testing.

    Parameters
    ----------
    path : Path
        Path to create the DAT file.
    n_blocks : int, default: 1
        Number of blocks in the file.
    n_x : int, default: 4
        Number of lateral samples.
    n_z : int, default: 6
        Number of depth samples.
    n_frames : int, default: 3
        Number of frames per block.
    start_index : int, default: 1
        Starting acquisition block index (1-based in file).
    known_pattern : bool, default: False
        If True, fill each block with a constant complex value equal to
        ``complex(block_idx + 1, 1)`` for deterministic value checks.
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

            data = np.random.randn(n_x, n_z, n_frames).astype(np.complex64) * (1 + 1j)
            if known_pattern:
                data = np.full(
                    (n_x, n_z, n_frames),
                    complex(block_idx + 1.0, 1.0),
                    dtype=np.complex64,
                )

            f.write(header.tobytes())
            f.write(data.tobytes())


@pytest.fixture
def single_autc_dat_file(tmp_path):
    """Create a single synthetic AUTC DAT file."""
    dat_path = tmp_path / "bf0part000.dat"
    _create_autc_dat_file(dat_path, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=1)
    return dat_path


@pytest.fixture
def single_autc_dat_file_known(tmp_path):
    """Create a single synthetic AUTC DAT file with a known pattern."""
    dat_path = tmp_path / "bf0part000.dat"
    _create_autc_dat_file(
        dat_path,
        n_blocks=2,
        n_x=4,
        n_z=6,
        n_frames=3,
        start_index=1,
        known_pattern=True,
    )
    return dat_path


@pytest.fixture
def multi_autc_dat_files(tmp_path):
    """Create multiple synthetic AUTC DAT files."""
    file1 = tmp_path / "bf0part000.dat"
    file2 = tmp_path / "bf0part001.dat"
    file3 = tmp_path / "bf0part002.dat"

    _create_autc_dat_file(file1, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=1)
    _create_autc_dat_file(file2, n_blocks=2, n_x=4, n_z=6, n_frames=3, start_index=9)
    _create_autc_dat_file(file3, n_blocks=1, n_x=4, n_z=6, n_frames=3, start_index=17)

    return tmp_path


@pytest.fixture
def empty_autc_dat_file(tmp_path):
    """Create an empty AUTC DAT file (0 bytes)."""
    dat_path = tmp_path / "bf0part000.dat"
    dat_path.touch()
    return dat_path


class TestAUTCDAT:
    """Tests for `AUTCDAT` class."""

    def test_autcdat_loads_file(self, single_autc_dat_file):
        """`AUTCDAT` loads a valid DAT file successfully."""
        autc = AUTCDAT(single_autc_dat_file)

        assert autc.path.is_file()
        assert autc.n_blocks == 2
        assert autc.n_frames_per_block == 3

    def test_autcdat_shape_property(self, single_autc_dat_file):
        """`AUTCDAT` shape property returns correct tuple."""
        autc = AUTCDAT(single_autc_dat_file)

        assert autc.shape == (2, 4, 6, 3)  # (n_blocks, n_x, n_z, n_frames)

    def test_autcdat_len(self, single_autc_dat_file):
        """`AUTCDAT` `__len__` returns number of blocks."""
        autc = AUTCDAT(single_autc_dat_file)

        assert len(autc) == 2

    def test_autcdat_indexing_single_block(self, single_autc_dat_file):
        """`AUTCDAT` indexing with int returns single block."""
        autc = AUTCDAT(single_autc_dat_file)

        data = autc[0]

        assert data.shape == (4, 6, 3)
        assert data.dtype == np.complex64

    def test_autcdat_indexing_slice(self, single_autc_dat_file):
        """`AUTCDAT` indexing with slice returns multiple blocks."""
        autc = AUTCDAT(single_autc_dat_file)

        data = autc[0:2]

        assert data.shape == (2, 4, 6, 3)

    def test_autcdat_indexing_tuple(self, single_autc_dat_file):
        """`AUTCDAT` indexing with tuple returns sliced data."""
        autc = AUTCDAT(single_autc_dat_file)

        data = autc[0, :, :2, :2]

        assert data.shape == (4, 2, 2)

    def test_autcdat_acquisition_indices(self, single_autc_dat_file):
        """`AUTCDAT` acquisition_block_indices property works."""
        autc = AUTCDAT(single_autc_dat_file)

        indices = autc.acquisition_block_indices

        assert len(indices) == 2
        # Indices are returned as stored in file (1-based).
        # With start_index=1 and step=4: [1, 5].
        assert indices[0] == 1
        assert indices[1] == 5

    def test_autcdat_nonexistent_file(self, tmp_path):
        """`AUTCDAT` raises `ValueError` for non-existent file."""
        non_existent = tmp_path / "nonexistent.dat"

        with pytest.raises(ValueError):
            AUTCDAT(non_existent)

    def test_autcdat_indexing_values_known_pattern(self, single_autc_dat_file_known):
        """`AUTCDAT` indexing returns correct values for a known pattern."""
        autc = AUTCDAT(single_autc_dat_file_known)

        # Block 0 was filled with complex(1, 1), block 1 with complex(2, 1).
        block0 = autc[0]
        block1 = autc[1]

        assert block0[0, 0, 0] == complex(1.0, 1.0), "Block 0 value corrupted"
        assert block1[0, 0, 0] == complex(2.0, 1.0), "Block 1 value corrupted"


class TestAUTCDATsLoader:
    """Tests for `AUTCDATsLoader` class."""

    def test_autcdats_loader_loads_multiple_files(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexes multiple DAT files."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        assert len(loader.dats) == 3
        assert loader.shape[0] == 5  # Total blocks: 2 + 2 + 1

    def test_autcdats_loader_single_file(self, single_autc_dat_file):
        """`AUTCDATsLoader` works with single DAT file in a directory."""
        # AUTCDATsLoader expects a directory, so we use the parent directory
        loader = AUTCDATsLoader(single_autc_dat_file.parent, show_progress=False)

        assert len(loader.dats) == 1
        assert loader.shape[0] == 2

    def test_autcdats_loader_blocks_property(self, multi_autc_dat_files):
        """`AUTCDATsLoader` blocks property returns sorted acquisition indices."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        blocks = loader.blocks

        assert len(blocks) == 5
        # Blocks are returned as stored in file (1-based).
        # File 1: [1, 5], File 2: [9, 13], File 3: [17].
        assert np.array_equal(blocks, [1, 5, 9, 13, 17])

    def test_autcdats_loader_indexing_single_block(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexing with int returns single block."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        data = loader[0]

        assert data.shape == (1, 4, 6, 3)

    def test_autcdats_loader_indexing_slice(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexing with slice returns multiple blocks."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        data = loader[0:3]

        assert data.shape == (3, 4, 6, 3)

    def test_autcdats_loader_indexing_across_files(self, multi_autc_dat_files):
        """`AUTCDATsLoader` indexing across file boundaries works."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        data = loader[1:4]  # Block 1 (from file 1), blocks 8, 12 (from file 2).

        assert data.shape == (3, 4, 6, 3)

    def test_autcdats_loader_empty_file_warns(self, multi_autc_dat_files, monkeypatch):
        """`AUTCDATsLoader` skips empty DAT files with warning."""
        import warnings

        empty_file = multi_autc_dat_files / "bf0part003.dat"
        empty_file.touch()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

            assert len(loader.dats) == 3  # Empty file skipped.
            assert any(
                "Skipping empty AUTC DAT file" in str(warning.message) for warning in w
            )

    def test_autcdats_loader_no_matching_files(self, tmp_path):
        """`AUTCDATsLoader` raises `FileNotFoundError` when no files match pattern."""
        with pytest.raises(FileNotFoundError, match="No AUTC DAT files found"):
            AUTCDATsLoader(tmp_path, show_progress=False)

    def test_autcdats_loader_nonexistent_root(self, tmp_path):
        """`AUTCDATsLoader` raises `ValueError` for non-existent directory."""
        non_existent = tmp_path / "nonexistent"

        with pytest.raises(ValueError):
            AUTCDATsLoader(non_existent, show_progress=False)

    def test_autcdats_loader_invalid_indexing_type(self, multi_autc_dat_files):
        """`AUTCDATsLoader` raises `TypeError` for invalid indexing type."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        with pytest.raises(
            TypeError, match="slice_object must be an int, slice, or tuple"
        ):
            loader["invalid"]  # type: ignore

    def test_autcdats_loader_indexing_with_list(self, multi_autc_dat_files):
        """`AUTCDATsLoader` raises `TypeError` for list indexing."""
        loader = AUTCDATsLoader(multi_autc_dat_files, show_progress=False)

        with pytest.raises(TypeError):
            loader[[0, 1, 2]]  # type: ignore
