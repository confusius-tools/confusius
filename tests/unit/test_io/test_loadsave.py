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

    def test_zarr_writes_readable_store(self, tmp_path):
        """.zarr extension writes a store that reloads to the same data."""
        path = tmp_path / "data.zarr"
        da = xr.DataArray(np.arange(12.0).reshape(4, 3))
        save(da, path)
        npt.assert_array_equal(load(path).values, da.values)

    def test_compound_zarr_extension(self, tmp_path):
        """.source.zarr compound extension writes a readable store."""
        path = tmp_path / "data.source.zarr"
        da = xr.DataArray(np.arange(12.0).reshape(4, 3))
        save(da, path)
        npt.assert_array_equal(load(path).values, da.values)

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


class TestSaveZarrSanitizesAttrs:
    """Non-JSON-serializable attrs are handled when saving to Zarr."""

    def test_nested_numpy_affines_round_trip(self, tmp_path):
        """`attrs["affines"]` numpy arrays survive a round-trip and reload as arrays."""
        affines = {
            "physical_to_world": np.eye(4),
            "stack": np.arange(32.0).reshape(2, 4, 4),
        }
        da = xr.DataArray(np.zeros((2, 2)), attrs={"affines": affines})
        path = tmp_path / "affines.zarr"
        save(da, path)

        loaded = load(path)
        for key, expected in affines.items():
            restored = loaded.attrs["affines"][key]
            assert isinstance(restored, np.ndarray)
            npt.assert_array_equal(restored, expected)

    def test_numpy_scalar_and_list_attrs_round_trip(self, tmp_path):
        """Numpy scalars and lists containing numpy values are kept, not dropped."""
        da = xr.DataArray(
            np.zeros((2, 2)),
            attrs={"code": np.int16(3), "angles": [np.float64(1.5), np.float64(-2.0)]},
        )
        path = tmp_path / "scalars.zarr"
        save(da, path)

        loaded = load(path)
        assert loaded.attrs["code"] == 3
        npt.assert_array_equal(loaded.attrs["angles"], [1.5, -2.0])

    def test_non_serializable_attr_dropped_with_warning(self, tmp_path):
        """Attrs that cannot be JSON-encoded are dropped, with a warning naming them."""
        da = xr.DataArray(np.zeros((2, 2)), attrs={"units": "dB", "cmap": object()})
        path = tmp_path / "drop.zarr"
        with pytest.warns(UserWarning, match="cmap"):
            save(da, path)

        loaded = load(path)
        assert loaded.attrs["units"] == "dB"
        assert "cmap" not in loaded.attrs

    def test_save_does_not_mutate_input_attrs(self, tmp_path):
        """The caller's DataArray keeps its original numpy attrs after saving."""
        da = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"m": np.eye(4)}})
        save(da, tmp_path / "nomutate.zarr")
        assert isinstance(da.attrs["affines"]["m"], np.ndarray)


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
