"""Unit tests for confusius.io.loadsave module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from confusius._utils.geometry import add_physical_coords_from_voxel_affine
from confusius.io.loadsave import load, save


class TestLoadDispatch:
    """Extension-based dispatch correctness."""

    def test_nii_gz_dispatches_to_load_nifti(self, tmp_path):
        """.nii.gz extension calls load_nifti."""
        path = tmp_path / "data.nii.gz"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve(), coordinate_model="legacy")
        assert result is mock_da

    def test_compound_nii_gz_extension(self, tmp_path):
        """.source.nii.gz compound extension calls load_nifti."""
        path = tmp_path / "data.source.nii.gz"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve(), coordinate_model="legacy")
        assert result is mock_da

    def test_nii_dispatches_to_load_nifti(self, tmp_path):
        """.nii extension calls load_nifti."""
        path = tmp_path / "data.nii"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve(), coordinate_model="legacy")
        assert result is mock_da

    def test_scan_dispatches_to_load_scan(self, tmp_path):
        """.scan extension calls load_scan."""
        path = tmp_path / "data.scan"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.scan.load_scan", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve(), coordinate_model="legacy")
        assert result is mock_da

    def test_compound_scan_extension(self, tmp_path):
        """.source.scan compound extension calls load_scan."""
        path = tmp_path / "data.source.scan"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.scan.load_scan", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve(), coordinate_model="legacy")
        assert result is mock_da

    def test_kwargs_forwarded_to_loader(self, tmp_path):
        """Extra kwargs are forwarded to the underlying loader."""
        path = tmp_path / "data.nii.gz"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            load(path, chunks=None)
        mock.assert_called_once_with(
            path.resolve(), coordinate_model="legacy", chunks=None
        )

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extension raises ValueError."""
        path = tmp_path / "data.hdf5"
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load(path)


class TestSaveDispatch:
    """Extension-based dispatch correctness for save()."""

    def test_nii_gz_dispatches_to_save_nifti(self, tmp_path):
        """.nii.gz extension calls save_nifti."""
        path = tmp_path / "data.nii.gz"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        mock.assert_called_once_with(da, path.resolve())

    def test_compound_nii_gz_save_extension(self, tmp_path):
        """.source.nii.gz compound extension calls save_nifti."""
        path = tmp_path / "data.source.nii.gz"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        mock.assert_called_once_with(da, path.resolve())

    def test_nii_dispatches_to_save_nifti(self, tmp_path):
        """.nii extension calls save_nifti."""
        path = tmp_path / "data.nii"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        mock.assert_called_once_with(da, path.resolve())

    def test_zarr_dispatches_to_to_zarr(self, tmp_path):
        """.zarr extension calls DataArray.to_zarr."""
        path = tmp_path / "data.zarr"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti"):
            save(da, path)
        da.to_zarr.assert_called_once_with(path.resolve())

    def test_compound_zarr_extension(self, tmp_path):
        """.source.zarr compound extension calls DataArray.to_zarr."""
        path = tmp_path / "data.source.zarr"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti"):
            save(da, path)
        da.to_zarr.assert_called_once_with(path.resolve())

    def test_kwargs_forwarded_to_saver(self, tmp_path):
        """Extra kwargs are forwarded to the underlying saver."""
        path = tmp_path / "data.nii.gz"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path, nifti_version=2)
        mock.assert_called_once_with(da, path.resolve(), nifti_version=2)

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extension raises ValueError."""
        da = MagicMock(spec=xr.DataArray)
        with pytest.raises(ValueError, match="Unsupported file extension"):
            save(da, tmp_path / "data.scan")


class TestLoadZarr:
    """Zarr variable extraction logic."""

    @pytest.fixture
    def single_var_zarr(self, tmp_path):
        """Zarr store with one variable."""
        ds = xr.Dataset({"iq": xr.DataArray(np.zeros((4, 3)))})
        path = tmp_path / "data.zarr"
        ds.to_zarr(path, zarr_format=2)
        return path

    @pytest.fixture
    def multi_var_zarr(self, tmp_path):
        """Zarr store with two variables."""
        ds = xr.Dataset(
            {
                "power": xr.DataArray(np.ones((4, 3))),
                "iq": xr.DataArray(np.zeros((4, 3))),
            }
        )
        path = tmp_path / "data.zarr"
        ds.to_zarr(path, zarr_format=2)
        return path

    def test_zarr_default_returns_first_variable(self, single_var_zarr):
        """variable=None returns the only variable as a DataArray."""
        result = load(single_var_zarr)
        assert isinstance(result, xr.DataArray)
        assert result.name == "iq"

    def test_zarr_named_variable(self, multi_var_zarr):
        """variable='iq' returns the iq DataArray."""
        result = load(multi_var_zarr, variable="iq")
        assert isinstance(result, xr.DataArray)
        assert result.name == "iq"

    def test_zarr_voxel_affine_rebuilds_coordinate_transform_index(self, tmp_path):
        """coordinate_model='voxel_affine' rebuilds CTI-backed physical coords."""
        base = xr.DataArray(
            np.arange(24, dtype=float).reshape(2, 3, 4),
            dims=["k", "j", "i"],
            coords={
                "k": [0.0, 1.0],
                "j": [0.0, 1.0, 2.0],
                "i": [0.0, 1.0, 2.0, 3.0],
            },
            name="power",
        )
        voxel_affine = add_physical_coords_from_voxel_affine(
            base,
            np.array(
                [
                    [0.4, 0.0, 0.1, 10.0],
                    [0.1, 0.3, 0.0, 20.0],
                    [0.0, 0.05, 0.25, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            voxel_dims=("k", "j", "i"),
            physical_coord_names=("z", "y", "x"),
            physical_coord_attrs={
                "z": {"units": "mm"},
                "y": {"units": "mm"},
                "x": {"units": "mm"},
            },
        )
        path = tmp_path / "voxel_affine.zarr"
        voxel_affine.to_dataset().to_zarr(path, zarr_format=2)

        loaded = load(path, variable="power", coordinate_model="voxel_affine")

        assert loaded.dims == ("k", "j", "i")
        assert loaded.coords["x"].dims == ("k", "j", "i")
        assert loaded.coords["y"].dims == ("k", "j", "i")
        assert loaded.coords["z"].dims == ("k", "j", "i")
        assert type(loaded.xindexes["x"]).__name__ == "CoordinateTransformIndex"
        np.testing.assert_allclose(
            loaded.attrs["voxel_to_physical"], voxel_affine.attrs["voxel_to_physical"]
        )
