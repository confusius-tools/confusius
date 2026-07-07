"""Unit tests for confusius.io.loadsave module."""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius.io.loadsave import load, save


class TestLoadDispatch:
    """Extension-based dispatch correctness."""

    def test_nii_gz_dispatches_to_load_nifti(self, tmp_path):
        """.nii.gz extension calls load_nifti."""
        path = tmp_path / "data.nii.gz"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve())
        assert result is mock_da

    def test_compound_nii_gz_extension(self, tmp_path):
        """.source.nii.gz compound extension calls load_nifti."""
        path = tmp_path / "data.source.nii.gz"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve())
        assert result is mock_da

    def test_nii_dispatches_to_load_nifti(self, tmp_path):
        """.nii extension calls load_nifti."""
        path = tmp_path / "data.nii"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve())
        assert result is mock_da

    def test_scan_dispatches_to_load_scan(self, tmp_path):
        """.scan extension calls load_scan."""
        path = tmp_path / "data.scan"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.scan.load_scan", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve())
        assert result is mock_da

    def test_compound_scan_extension(self, tmp_path):
        """.source.scan compound extension calls load_scan."""
        path = tmp_path / "data.source.scan"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.scan.load_scan", return_value=mock_da) as mock:
            result = load(path)
        mock.assert_called_once_with(path.resolve())
        assert result is mock_da

    def test_kwargs_forwarded_to_loader(self, tmp_path):
        """Extra kwargs are forwarded to the underlying loader."""
        path = tmp_path / "data.nii.gz"
        mock_da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da) as mock:
            load(path, chunks=None)
        mock.assert_called_once_with(path.resolve(), chunks=None)

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
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        da.to_zarr.assert_called_once_with(path.resolve())

    def test_compound_zarr_extension(self, tmp_path):
        """.source.zarr compound extension calls DataArray.to_zarr."""
        path = tmp_path / "data.source.zarr"
        da = MagicMock(spec=xr.DataArray)
        with patch("confusius.io.nifti.save_nifti") as mock:
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


class TestLoadRestoresAtlasCmapAndNorm:
    """cmap/norm are rebuilt from rgb_lookup when missing after a round-trip."""

    RGB_LOOKUP = {1: [255, 0, 0], 2: [0, 255, 0]}

    @pytest.fixture
    def atlas_like_zarr(self, tmp_path):
        """Zarr store mimicking an Atlas annotation: rgb_lookup present, no cmap/norm."""
        da = xr.DataArray(
            np.array([[0, 1], [2, 1]], dtype=np.int32),
            attrs={"rgb_lookup": self.RGB_LOOKUP},
        )
        path = tmp_path / "annotation.zarr"
        xr.Dataset({"annotation": da}).to_zarr(path, zarr_format=2)
        return path

    def test_rebuilds_cmap_and_norm_from_rgb_lookup(self, atlas_like_zarr):
        """cmap/norm reproduce the exact rgb_lookup colors, not just the right types."""
        result = load(atlas_like_zarr)

        cmap = result.attrs["cmap"]
        norm = result.attrs["norm"]
        for label_id, expected_rgb in self.RGB_LOOKUP.items():
            expected_rgba = tuple(c / 255 for c in expected_rgb) + (1.0,)
            npt.assert_allclose(cmap(norm(label_id)), expected_rgba)

    def test_does_not_override_existing_cmap_and_norm(self, tmp_path):
        """Existing cmap/norm attrs are left untouched."""
        mock_da = MagicMock(spec=xr.DataArray)
        mock_da.attrs = {
            "rgb_lookup": {1: [255, 0, 0]},
            "cmap": "sentinel_cmap",
            "norm": "sentinel_norm",
        }
        path = tmp_path / "data.nii.gz"
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da):
            result = load(path)

        assert result.attrs["cmap"] == "sentinel_cmap"
        assert result.attrs["norm"] == "sentinel_norm"

    def test_no_rgb_lookup_leaves_attrs_untouched(self, tmp_path):
        """DataArrays without rgb_lookup are returned unmodified."""
        mock_da = MagicMock(spec=xr.DataArray)
        mock_da.attrs = {"task_name": "test"}
        path = tmp_path / "data.nii.gz"
        with patch("confusius.io.nifti.load_nifti", return_value=mock_da):
            result = load(path)

        assert result.attrs == {"task_name": "test"}
