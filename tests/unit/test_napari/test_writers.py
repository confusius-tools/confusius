"""Unit tests for confusius._napari._writers module.

Writer functions are plain Python, no napari viewer required. Tests verify:
- Successful write + round-trip when a ConfUSIus DataArray is in layer metadata
  (layers loaded via the ConfUSIus reader — image layers).
- Successful write + round-trip when no DataArray is in metadata and one is
  reconstructed from napari layer properties (user-drawn labels layers).
- Both write_nifti and write_zarr cover both code paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from confusius._napari._io._writers import write_nifti, write_zarr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _meta_with_da(da: xr.DataArray) -> dict:
    """Build napari-style metadata carrying the original DataArray."""
    return {"metadata": {"xarray": da}}


def _meta_from_layer(
    da: xr.DataArray,
    *,
    include_units: bool = True,
) -> dict:
    """Build napari-style metadata as if the layer was drawn by a user.

    Mirrors what napari populates from a labels layer's ``_get_state()``:
    ``scale``, ``translate``, ``axis_labels``, and optionally ``units``.
    No ``"xarray"`` key is present, so the writer must reconstruct.
    """
    import confusius  # noqa: F401 — registers .fusi accessor

    scale = [da.fusi.spacing[d] or 1.0 for d in da.dims]
    translate = [float(da[d].values[0]) for d in da.dims]
    units = [da[d].attrs.get("units") for d in da.dims] if include_units else None
    return {
        "axis_labels": list(da.dims),
        "scale": scale,
        "translate": translate,
        "units": units,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# write_nifti — DataArray path (image layers loaded via reader)
# ---------------------------------------------------------------------------


class TestWriteNiftiWithDataArray:
    def test_returns_path_list(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """write_nifti returns a list containing the output path."""
        path = str(tmp_path / "out.nii.gz")
        result = write_nifti(
            path, sample_4d_volume.values, _meta_with_da(sample_4d_volume)
        )
        assert result == [path]

    def test_file_created(self, tmp_path: Path, sample_4d_volume: xr.DataArray) -> None:
        """NIfTI file is created on disk."""
        path = tmp_path / "out.nii.gz"
        write_nifti(str(path), sample_4d_volume.values, _meta_with_da(sample_4d_volume))
        assert path.exists()

    def test_roundtrip_values(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Data values are preserved through a write → load round-trip."""
        from confusius.io import load

        path = tmp_path / "out.nii.gz"
        write_nifti(str(path), sample_4d_volume.values, _meta_with_da(sample_4d_volume))
        loaded = load(path)
        npt.assert_allclose(loaded.values, sample_4d_volume.values, rtol=1e-5)

    def test_roundtrip_spatial_coordinates(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Spatial coordinates are preserved through a write → load round-trip."""
        from confusius.io import load

        path = tmp_path / "out.nii.gz"
        write_nifti(str(path), sample_4d_volume.values, _meta_with_da(sample_4d_volume))
        loaded = load(path)
        for dim in ("z", "y", "x"):
            npt.assert_allclose(
                loaded[dim].values, sample_4d_volume[dim].values, rtol=1e-4
            )


# ---------------------------------------------------------------------------
# write_nifti — reconstruction path (user-drawn labels layers)
# ---------------------------------------------------------------------------


class TestWriteNiftiFromReconstruction:
    def test_returns_path_list(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """write_nifti returns path list when reconstructing from layer state."""
        path = str(tmp_path / "out.nii.gz")
        result = write_nifti(
            path, sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        assert result == [path]

    def test_file_created(self, tmp_path: Path, sample_4d_volume: xr.DataArray) -> None:
        """NIfTI file is created when reconstructing from layer state."""
        path = tmp_path / "out.nii.gz"
        write_nifti(
            str(path), sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        assert path.exists()

    def test_roundtrip_values(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Data values are preserved when reconstructing from layer state."""
        from confusius.io import load

        path = tmp_path / "out.nii.gz"
        write_nifti(
            str(path), sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        loaded = load(path)
        npt.assert_allclose(loaded.values, sample_4d_volume.values, rtol=1e-5)

    def test_roundtrip_spatial_coordinates(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Spatial coordinates are reconstructed correctly from scale/translate."""
        from confusius.io import load

        path = tmp_path / "out.nii.gz"
        write_nifti(
            str(path), sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        loaded = load(path)
        for dim in ("z", "y", "x"):
            npt.assert_allclose(
                loaded[dim].values, sample_4d_volume[dim].values, rtol=1e-4
            )

    def test_integer_labels_preserved(self, tmp_path: Path) -> None:
        """Integer label values survive a write → load round-trip."""
        from confusius.io import load

        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(4, 6, 8), dtype=np.int32)
        meta = {
            "axis_labels": ["z", "y", "x"],
            "scale": [0.2, 0.1, 0.05],
            "translate": [1.0, 2.0, 3.0],
            "units": ["mm", "mm", "mm"],
            "metadata": {},
        }
        path = tmp_path / "labels.nii.gz"
        write_nifti(str(path), labels, meta)
        loaded = load(path)
        npt.assert_array_equal(loaded.values, labels)


# ---------------------------------------------------------------------------
# write_zarr — DataArray path (image layers loaded via reader)
# ---------------------------------------------------------------------------


class TestWriteZarrWithDataArray:
    def test_returns_path_list(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """write_zarr returns a list containing the output path."""
        path = str(tmp_path / "out.zarr")
        result = write_zarr(
            path, sample_4d_volume.values, _meta_with_da(sample_4d_volume)
        )
        assert result == [path]

    def test_store_created(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Zarr store directory is created on disk."""
        path = tmp_path / "out.zarr"
        write_zarr(str(path), sample_4d_volume.values, _meta_with_da(sample_4d_volume))
        assert path.is_dir()

    def test_roundtrip_values(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Data values are preserved through a write → load round-trip."""
        from confusius.io import load

        path = tmp_path / "out.zarr"
        write_zarr(str(path), sample_4d_volume.values, _meta_with_da(sample_4d_volume))
        loaded = load(path)
        npt.assert_allclose(loaded.values, sample_4d_volume.values, rtol=1e-6)

    def test_roundtrip_all_coordinates(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """All coordinates are preserved through a write → load round-trip."""
        from confusius.io import load

        path = tmp_path / "out.zarr"
        write_zarr(str(path), sample_4d_volume.values, _meta_with_da(sample_4d_volume))
        loaded = load(path)
        for dim in ("time", "z", "y", "x"):
            npt.assert_allclose(
                loaded[dim].values, sample_4d_volume[dim].values, rtol=1e-6
            )


# ---------------------------------------------------------------------------
# write_zarr — reconstruction path (user-drawn labels layers)
# ---------------------------------------------------------------------------


class TestWriteZarrFromReconstruction:
    def test_returns_path_list(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """write_zarr returns path list when reconstructing from layer state."""
        path = str(tmp_path / "out.zarr")
        result = write_zarr(
            path, sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        assert result == [path]

    def test_store_created(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Zarr store directory is created when reconstructing from layer state."""
        path = tmp_path / "out.zarr"
        write_zarr(
            str(path), sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        assert path.is_dir()

    def test_roundtrip_values(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Data values are preserved when reconstructing from layer state."""
        from confusius.io import load

        path = tmp_path / "out.zarr"
        write_zarr(
            str(path), sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        loaded = load(path)
        npt.assert_allclose(loaded.values, sample_4d_volume.values, rtol=1e-6)

    def test_roundtrip_spatial_coordinates(
        self, tmp_path: Path, sample_4d_volume: xr.DataArray
    ) -> None:
        """Spatial coordinates are reconstructed correctly from scale/translate."""
        from confusius.io import load

        path = tmp_path / "out.zarr"
        write_zarr(
            str(path), sample_4d_volume.values, _meta_from_layer(sample_4d_volume)
        )
        loaded = load(path)
        for dim in ("time", "z", "y", "x"):
            npt.assert_allclose(
                loaded[dim].values, sample_4d_volume[dim].values, rtol=1e-6
            )

    def test_integer_labels_preserved(self, tmp_path: Path) -> None:
        """Integer label values survive a write → load round-trip."""
        from confusius.io import load

        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(4, 6, 8), dtype=np.int32)
        meta = {
            "axis_labels": ["z", "y", "x"],
            "scale": [0.2, 0.1, 0.05],
            "translate": [1.0, 2.0, 3.0],
            "units": ["mm", "mm", "mm"],
            "metadata": {},
        }
        path = tmp_path / "labels.zarr"
        write_zarr(str(path), labels, meta)
        loaded = load(path)
        npt.assert_array_equal(loaded.values, labels)


# ---------------------------------------------------------------------------
# _compute_dataarray_from_layer — unit tests for the reconstruction helper
# ---------------------------------------------------------------------------


class TestDaFromNapariLayer:
    def test_default_dims_3d(self) -> None:
        """3D data without axis_labels gets default dims (z, y, x)."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((4, 6, 8))
        da = _compute_dataarray_from_layer(
            data, {"scale": [0.2, 0.1, 0.05], "translate": [1.0, 2.0, 3.0]}
        )
        assert list(da.dims) == ["z", "y", "x"]

    def test_default_dims_4d(self) -> None:
        """4D data without axis_labels gets default dims (time, z, y, x)."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((10, 4, 6, 8))
        da = _compute_dataarray_from_layer(data, {})
        assert list(da.dims) == ["time", "z", "y", "x"]

    def test_scale_sets_coord_spacing(self) -> None:
        """Scale values produce correct coordinate spacing."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((4, 6))
        da = _compute_dataarray_from_layer(
            data,
            {"axis_labels": ["z", "x"], "scale": [0.2, 0.05], "translate": [1.0, 3.0]},
        )
        npt.assert_allclose(np.diff(da["z"].values), 0.2, rtol=1e-10)
        npt.assert_allclose(np.diff(da["x"].values), 0.05, rtol=1e-10)

    def test_translate_sets_coord_origin(self) -> None:
        """Translate values set the first coordinate value for each dimension."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((4, 6))
        da = _compute_dataarray_from_layer(
            data,
            {"axis_labels": ["z", "x"], "scale": [0.2, 0.05], "translate": [1.0, 3.0]},
        )
        assert da["z"].values[0] == pytest.approx(1.0)
        assert da["x"].values[0] == pytest.approx(3.0)

    def test_units_stored_in_coord_attrs(self) -> None:
        """Unit strings from meta are stored in coordinate attrs."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((4, 6))
        da = _compute_dataarray_from_layer(
            data,
            {
                "axis_labels": ["z", "x"],
                "scale": [0.2, 0.05],
                "translate": [0.0, 0.0],
                "units": ["mm", "mm"],
            },
        )
        assert da["z"].attrs["units"] == "mm"
        assert da["x"].attrs["units"] == "mm"

    def test_voxdim_stored_in_coord_attrs(self) -> None:
        """voxdim is stored in coordinate attrs from scale magnitude."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((4,))
        da = _compute_dataarray_from_layer(
            data,
            {"axis_labels": ["z"], "scale": [0.2], "translate": [0.0]},
        )
        assert da["z"].attrs["voxdim"] == pytest.approx(0.2)

    def test_napari_generic_axis_labels_replaced_by_defaults(self) -> None:
        """Napari's 'axis -N' labels are replaced with ConfUSIus default dim names."""
        from confusius._napari._io._writers import _compute_dataarray_from_layer

        data = np.zeros((4, 6, 8))
        da = _compute_dataarray_from_layer(
            data, {"axis_labels": ["axis -3", "axis -2", "axis -1"]}
        )
        assert list(da.dims) == ["z", "y", "x"]

    def test_napari_pixel_units_treated_as_absent(self) -> None:
        """Pint pixel/dimensionless units from napari are not stored in coord attrs."""
        from unittest.mock import MagicMock

        from confusius._napari._io._writers import _compute_dataarray_from_layer

        # Simulate pint Unit objects as napari passes them.
        pixel_unit = MagicMock()
        pixel_unit.__str__ = lambda self: "pixel"

        data = np.zeros((4,))
        da = _compute_dataarray_from_layer(
            data,
            {"axis_labels": ["z"], "scale": [0.2], "translate": [0.0], "units": [pixel_unit]},
        )
        assert "units" not in da["z"].attrs
