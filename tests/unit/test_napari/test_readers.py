"""Unit tests for confusius._napari._readers module.

Reader functions are plain Python, no napari viewer required. Gate tests verify the
None-vs-callable contract; LayerData tests verify that the
scale/translate/axis_labels/units/colormap values produced for napari are physically
correct.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius._napari._io._readers import read_nifti, read_scan, read_zarr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nifti_path(tmp_path: Path) -> Path:
    """3D NIfTI file (.nii.gz)."""
    data = np.random.default_rng(0).random((4, 6, 8)).astype(np.float32)
    path = tmp_path / "vol.nii.gz"
    nib.Nifti1Image(data, np.eye(4)).to_filename(path)
    return path


@pytest.fixture
def scan_path(tmp_path: Path) -> Path:
    """Placeholder .scan file — content is irrelevant for gate tests."""
    path = tmp_path / "data.scan"
    path.touch()
    return path


@pytest.fixture
def zarr_3d_path(tmp_path: Path, sample_3d_volume: xr.DataArray) -> Path:
    """Zarr store built from the shared sample_3d_volume fixture."""
    path = tmp_path / "vol3d.zarr"
    xr.Dataset({"data": sample_3d_volume}).to_zarr(path, zarr_format=2)
    return path


@pytest.fixture
def zarr_4d_path(tmp_path: Path, sample_4d_volume: xr.DataArray) -> Path:
    """Zarr store built from the shared sample_4d_volume fixture."""
    path = tmp_path / "vol4d.zarr"
    xr.Dataset({"data": sample_4d_volume}).to_zarr(path, zarr_format=2)
    return path


# ---------------------------------------------------------------------------
# Gate logic: None vs callable
# ---------------------------------------------------------------------------


class TestReadNiftiGating:
    """read_nifti returns None for invalid inputs, a callable for valid files."""

    def test_returns_none_for_list(self, nifti_path: Path) -> None:
        """Path list is rejected — napari should try other readers."""
        assert read_nifti([str(nifti_path)]) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Non-existent path is rejected."""
        assert read_nifti(str(tmp_path / "missing.nii.gz")) is None

    def test_returns_callable_for_valid_nifti(self, nifti_path: Path) -> None:
        """Existing .nii.gz file returns a reader function."""
        assert callable(read_nifti(str(nifti_path)))


class TestReadScanGating:
    """read_scan returns None for invalid inputs, a callable for valid files."""

    def test_returns_none_for_list(self, scan_path: Path) -> None:
        """Path list is rejected."""
        assert read_scan([str(scan_path)]) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Non-existent path is rejected."""
        assert read_scan(str(tmp_path / "missing.scan")) is None

    def test_returns_callable_for_existing_scan_file(self, scan_path: Path) -> None:
        """Existing .scan file returns a reader function."""
        assert callable(read_scan(str(scan_path)))


class TestReadZarrGating:
    """read_zarr returns None for invalid inputs, a callable for valid stores."""

    def test_returns_none_for_list(self, zarr_3d_path: Path) -> None:
        """Path list is rejected."""
        assert read_zarr([str(zarr_3d_path)]) is None

    def test_returns_none_for_missing_path(self, tmp_path: Path) -> None:
        """Non-existent path is rejected."""
        assert read_zarr(str(tmp_path / "missing.zarr")) is None

    def test_returns_none_for_dir_without_zarr_indicators(self, tmp_path: Path) -> None:
        """Plain directory with no zarr metadata files is rejected."""
        plain_dir = tmp_path / "not_a_store.zarr"
        plain_dir.mkdir()
        assert read_zarr(str(plain_dir)) is None

    def test_returns_callable_for_valid_zarr_store(self, zarr_3d_path: Path) -> None:
        """Valid zarr store returns a reader function."""
        assert callable(read_zarr(str(zarr_3d_path)))


# ---------------------------------------------------------------------------
# LayerData structure
# ---------------------------------------------------------------------------


class TestReaderLayerData:
    """The ReaderFunction returns physically correct napari LayerData."""

    def test_3d_scale_and_translate(self, zarr_3d_path: Path) -> None:
        """3D scale matches fusi.spacing; translate matches fusi.origin; layer type is image."""
        reader = read_zarr(str(zarr_3d_path))
        assert reader is not None
        _, kwargs, layer_type = reader(str(zarr_3d_path))[0]

        assert layer_type == "image"

        # Coords: z=[1.0..1.6 step 0.2], y=[2.0..2.5 step 0.1], x=[3.0..3.35 step 0.05]
        npt.assert_allclose(kwargs["scale"], [0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(kwargs["translate"], [1.0, 2.0, 3.0], rtol=1e-5)
        assert kwargs["axis_labels"] == ["z", "y", "x"]

    def test_4d_scale_uses_time_spacing(self, zarr_4d_path: Path) -> None:
        """4D scale uses fusi.spacing for all dims, including time."""
        reader = read_zarr(str(zarr_4d_path))
        assert reader is not None
        _, kwargs, _ = reader(str(zarr_4d_path))[0]

        # time: origin=10.0 spacing=0.5; z: origin=1.0 spacing=0.2;
        # y: origin=2.0 spacing=0.1; x: origin=3.0 spacing=0.05
        npt.assert_allclose(kwargs["scale"], [0.5, 0.2, 0.1, 0.05], rtol=1e-5)
        npt.assert_allclose(kwargs["translate"], [10.0, 1.0, 2.0, 3.0], rtol=1e-5)
        assert kwargs["axis_labels"] == ["time", "z", "y", "x"]

    def test_time_last_dim_order(self, tmp_path: Path) -> None:
        """scale/translate/units follow the actual dim order when time is last (e.g. NIfTI)."""
        da = xr.DataArray(
            np.zeros((8, 6, 4, 10), dtype=np.float32),
            dims=["x", "y", "z", "time"],
            coords={
                "x": xr.DataArray(
                    1.0 + np.arange(8) * 0.05, dims=["x"], attrs={"units": "mm"}
                ),
                "y": xr.DataArray(
                    2.0 + np.arange(6) * 0.10, dims=["y"], attrs={"units": "mm"}
                ),
                "z": xr.DataArray(
                    3.0 + np.arange(4) * 0.20, dims=["z"], attrs={"units": "mm"}
                ),
                "time": xr.DataArray(
                    10.0 + np.arange(10) * 0.5, dims=["time"], attrs={"units": "s"}
                ),
            },
        )
        path = tmp_path / "time_last.zarr"
        xr.Dataset({"data": da}).to_zarr(path, zarr_format=2)

        reader = read_zarr(str(path))
        assert reader is not None
        _, kwargs, _ = reader(str(path))[0]

        # scale and translate must be in (x, y, z, time) order, not time-prepended.
        npt.assert_allclose(kwargs["scale"], [0.05, 0.10, 0.20, 0.5], rtol=1e-5)
        npt.assert_allclose(kwargs["translate"], [1.0, 2.0, 3.0, 10.0], rtol=1e-5)
        assert kwargs["axis_labels"] == ["x", "y", "z", "time"]
        assert kwargs["units"] == ["mm", "mm", "mm", "s"]

    def test_units_from_coord_attrs(self, zarr_3d_path: Path) -> None:
        """Units are read from coordinate attrs and passed to napari."""
        reader = read_zarr(str(zarr_3d_path))
        assert reader is not None
        _, kwargs, _ = reader(str(zarr_3d_path))[0]

        assert kwargs["units"] == ["mm", "mm", "mm"]

    def test_default_colormap_is_gray(self, zarr_3d_path: Path) -> None:
        """Colormap defaults to gray when da.attrs has no 'cmap' key."""
        reader = read_zarr(str(zarr_3d_path))
        assert reader is not None
        _, kwargs, _ = reader(str(zarr_3d_path))[0]

        assert kwargs["colormap"] == "gray"

    def test_colormap_from_attrs(self, tmp_path: Path) -> None:
        """Colormap is read from da.attrs['cmap'] when present."""
        da = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["z", "x"],
            coords={
                "z": np.arange(4) * 0.1,
                "x": np.arange(6) * 0.05,
            },
            attrs={"cmap": "viridis"},
        )
        path = tmp_path / "colored.zarr"
        xr.Dataset({"data": da}).to_zarr(path, zarr_format=2)

        reader = read_zarr(str(path))
        assert reader is not None
        _, kwargs, _ = reader(str(path))[0]

        assert kwargs["colormap"] == "viridis"

    def test_xarray_stored_in_metadata(self, zarr_3d_path: Path) -> None:
        """The original DataArray is stored in metadata for downstream access."""
        reader = read_zarr(str(zarr_3d_path))
        assert reader is not None
        _, kwargs, _ = reader(str(zarr_3d_path))[0]

        assert isinstance(kwargs["metadata"]["xarray"], xr.DataArray)
