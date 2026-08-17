"""Unit tests for confusius.io.loadsave module."""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius._utils.geometry import (
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
)
from confusius.io.loadsave import load, save
from confusius.xarray import create_fusi_dataarray


@pytest.fixture
def saveable_volume(sample_fusi_3d: xr.DataArray) -> xr.DataArray:
    """Canonical fUSI volume accepted by public save APIs."""
    return sample_fusi_3d.copy(deep=True)


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

    def test_nii_gz_dispatches_to_save_nifti(self, tmp_path, saveable_volume):
        """.nii.gz extension calls save_nifti."""
        path = tmp_path / "data.nii.gz"
        da = saveable_volume
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        mock.assert_called_once_with(da, path.resolve())

    def test_compound_nii_gz_save_extension(self, tmp_path, saveable_volume):
        """.source.nii.gz compound extension calls save_nifti."""
        path = tmp_path / "data.source.nii.gz"
        da = saveable_volume
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        mock.assert_called_once_with(da, path.resolve())

    def test_nii_dispatches_to_save_nifti(self, tmp_path, saveable_volume):
        """.nii extension calls save_nifti."""
        path = tmp_path / "data.nii"
        da = saveable_volume
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path)
        mock.assert_called_once_with(da, path.resolve())

    def test_zarr_writes_readable_store(self, tmp_path, saveable_volume):
        """.zarr extension writes a store that reloads to the same data."""
        path = tmp_path / "data.zarr"
        da = saveable_volume
        save(da, path)
        npt.assert_array_equal(load(path).values, da.values)

    def test_zarr_roundtrips_pose_dependent_geometry(self, tmp_path):
        """A pose-stacked voxel-to-world affine survives a Zarr round trip."""
        affine = np.stack(
            [
                np.eye(4),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 100.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ]
        )
        da = create_fusi_dataarray(
            np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4),
            dims=("pose", "k", "j", "i"),
            voxel_to_world=affine,
        )
        path = tmp_path / "data.zarr"

        save(da, path)
        loaded = load(path)

        assert loaded.dims == da.dims
        npt.assert_array_equal(loaded.coords["pose"].values, da.coords["pose"].values)
        npt.assert_allclose(loaded.coords["z"].values, da.coords["z"].values)
        npt.assert_allclose(
            get_voxel_to_world_affine(loaded), get_voxel_to_world_affine(da)
        )

    def test_compound_zarr_extension(self, tmp_path, saveable_volume):
        """.source.zarr compound extension writes a readable store."""
        path = tmp_path / "data.source.zarr"
        da = saveable_volume
        save(da, path)
        npt.assert_array_equal(load(path).values, da.values)

    def test_zarr_suppresses_consolidated_metadata_warning(
        self, tmp_path, saveable_volume
    ):
        """Zarr v3 consolidated-metadata warning from xarray/zarr is hidden."""
        path = tmp_path / "data.zarr"
        da = saveable_volume

        def fake_to_zarr(*args, **kwargs) -> None:
            warnings.warn(
                "Consolidated metadata is currently not part in the Zarr format 3 specification.",
                UserWarning,
            )

        with patch.object(xr.DataArray, "to_zarr", side_effect=fake_to_zarr):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                save(da, path)

        assert not caught

    def test_kwargs_forwarded_to_saver(self, tmp_path, saveable_volume):
        """Extra kwargs are forwarded to the underlying saver."""
        path = tmp_path / "data.nii.gz"
        da = saveable_volume
        with patch("confusius.io.nifti.save_nifti") as mock:
            save(da, path, nifti_version=2)
        mock.assert_called_once_with(da, path.resolve(), nifti_version=2)

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extension is reported before fUSI data validation."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            save(xr.DataArray(np.zeros((2, 2))), tmp_path / "data.scan")


class TestSaveZarrSanitizesAttrs:
    """Non-JSON-serializable attrs are handled when saving to Zarr."""

    def test_nested_numpy_affines_round_trip(self, tmp_path, saveable_volume):
        """`attrs["affines"]` numpy arrays survive a round-trip and reload as arrays."""
        affines = {
            "world_to_custom": np.eye(4),
            "stack": np.arange(32.0).reshape(2, 4, 4),
        }
        da = saveable_volume.assign_attrs(affines=affines)
        path = tmp_path / "affines.zarr"
        save(da, path)

        loaded = load(path)
        for key, expected in affines.items():
            restored = loaded.attrs["affines"][key]
            assert isinstance(restored, np.ndarray)
            npt.assert_array_equal(restored, expected)

    def test_numpy_scalar_and_list_attrs_round_trip(self, tmp_path, saveable_volume):
        """Numpy scalars and lists containing numpy values are kept, not dropped."""
        da = saveable_volume.assign_attrs(
            code=np.int16(3), angles=[np.float64(1.5), np.float64(-2.0)]
        )
        path = tmp_path / "scalars.zarr"
        save(da, path)

        loaded = load(path)
        assert loaded.attrs["code"] == 3
        npt.assert_array_equal(loaded.attrs["angles"], [1.5, -2.0])

    def test_non_serializable_attr_dropped_with_warning(
        self, tmp_path, saveable_volume
    ):
        """Attrs that cannot be JSON-encoded are dropped, with a warning naming them."""
        da = saveable_volume.assign_attrs(units="dB", cmap=object())
        path = tmp_path / "drop.zarr"
        with pytest.warns(UserWarning, match="cmap"):
            save(da, path)

        loaded = load(path)
        assert loaded.attrs["units"] == "dB"
        assert "cmap" not in loaded.attrs

    def test_save_does_not_mutate_input_attrs(self, tmp_path, saveable_volume):
        """The caller's DataArray keeps its original numpy attrs after saving."""
        da = saveable_volume.assign_attrs(affines={"m": np.eye(4)})
        save(da, tmp_path / "nomutate.zarr")
        assert isinstance(da.attrs["affines"]["m"], np.ndarray)


class TestLoadZarr:
    """Zarr variable extraction logic."""

    @pytest.fixture
    def single_var_zarr(self, tmp_path):
        """Zarr store with one variable, written via confusius.io.save()."""
        da = create_fusi_dataarray(
            np.zeros((4, 3, 2)),
            dims=("k", "j", "i"),
            spacing=(1.0, 1.0, 1.0),
            name="iq",
        )
        path = tmp_path / "data.zarr"
        save(da, path)
        return path

    @pytest.fixture
    def multi_var_zarr(self, tmp_path):
        """Zarr store with two variables, each canonical (attrs['voxel_to_world'] set
        the same way confusius.io.save() would, since save() only writes one variable
        per store)."""
        power = create_fusi_dataarray(
            np.ones((4, 3, 2)),
            dims=("k", "j", "i"),
            spacing=(1.0, 1.0, 1.0),
            name="power",
        )
        iq = create_fusi_dataarray(
            np.zeros((4, 3, 2)),
            dims=("k", "j", "i"),
            spacing=(1.0, 1.0, 1.0),
            name="iq",
        )
        path = tmp_path / "data.zarr"
        variables = {}
        for da in (power, iq):
            voxel_to_world = get_voxel_to_world_affine(da)
            world_coord_names = get_voxel_to_world_coord_names(da)
            world_coord_attrs = {
                name: dict(da.coords[name].attrs)
                for name in world_coord_names
                if name in da.coords
            }
            da = da.drop_vars(world_coord_names)
            da.attrs = {
                **da.attrs,
                "voxel_to_world": voxel_to_world,
                "world_coord_attrs": world_coord_attrs,
            }
            variables[da.name] = da
        xr.Dataset(variables).to_zarr(path)
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

    def test_zarr_rejects_store_without_voxel_to_world(self, tmp_path):
        """A Zarr store not written by save() (no attrs['voxel_to_world']) is rejected."""
        path = tmp_path / "foreign.zarr"
        xr.Dataset({"data": xr.DataArray(np.zeros((4, 3)))}).to_zarr(path)

        with pytest.raises(ValueError, match="was not written by confusius.io.save"):
            load(path)


class TestLoadRestoresAtlasCmapAndNorm:
    """cmap/norm are rebuilt from rgb_lookup when missing after a round-trip."""

    RGB_LOOKUP = {1: [255, 0, 0], 2: [0, 255, 0]}

    @pytest.fixture
    def atlas_like_zarr(self, tmp_path):
        """Zarr store mimicking an Atlas annotation: rgb_lookup present, no cmap/norm."""
        da = create_fusi_dataarray(
            np.array([[[0, 1], [2, 1]]], dtype=np.int32),
            dims=("k", "j", "i"),
            spacing=(1.0, 1.0, 1.0),
            name="annotation",
            attrs={"rgb_lookup": self.RGB_LOOKUP},
        )
        path = tmp_path / "annotation.zarr"
        save(da, path)
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
