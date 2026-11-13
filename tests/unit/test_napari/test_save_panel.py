"""Unit tests for the SavePanel DataArray reconstruction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import confusius._napari._data._save_panel as save_module
from confusius.plotting.napari import plot_napari
from confusius.validation import validate_voxeldata


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def save_panel(viewer):
    from confusius._napari._data._save_panel import SavePanel

    return SavePanel(viewer)


@pytest.fixture
def labels_layer(viewer, sample_voxeldata_3dt):
    """A user-drawn 3D labels layer created from the signals panel button."""
    from confusius._napari._signals._panel import SignalPanel

    plot_napari(sample_voxeldata_3dt, viewer=viewer)
    SignalPanel(viewer)._create_labels_layer()
    return viewer.layers["Labels (3D)"]


def test_save_button_requires_layer_and_output_path(
    viewer, save_panel, sample_voxeldata_3d
):
    assert not save_panel._save_btn.isEnabled()
    save_panel._path_edit.setText("output.nii.gz")
    assert not save_panel._save_btn.isEnabled()
    plot_napari(sample_voxeldata_3d, viewer=viewer)
    assert save_panel._save_btn.isEnabled()
    save_panel._path_edit.setText("   ")
    assert not save_panel._save_btn.isEnabled()
    save_panel._path_edit.setText("output.zarr")
    assert save_panel._save_btn.isEnabled()
    viewer.layers.clear()
    assert not save_panel._save_btn.isEnabled()


def test_layer_changes_do_not_enable_save_while_busy(
    viewer, save_panel, sample_voxeldata_3d, tmp_path, monkeypatch
):
    worker = MagicMock()
    save_data = MagicMock(return_value=worker)
    monkeypatch.setattr(save_module, "_save_data", save_data)
    monkeypatch.setattr(save_module, "show_info", MagicMock())
    plot_napari(sample_voxeldata_3d, viewer=viewer)
    path = tmp_path / "output.nii.gz"
    save_panel._path_edit.setText(str(path))
    save_panel._save_btn.click()
    worker.start.assert_called_once()
    assert not save_panel._save_btn.isEnabled()
    viewer.layers[0].name = "Renamed while saving"
    assert not save_panel._save_btn.isEnabled()
    viewer.layers.clear()
    assert not save_panel._save_btn.isEnabled()
    worker.returned.connect.call_args.args[0](path)
    assert not save_panel._save_btn.isEnabled()
    plot_napari(sample_voxeldata_3d, viewer=viewer)
    assert save_panel._save_btn.isEnabled()


class TestBuildDataArray:
    def test_template_from_4d_image_yields_valid_voxeldata(
        self, viewer, save_panel, labels_layer, sample_voxeldata_3dt
    ):
        save_panel._layer_combo.setCurrentText(labels_layer.name)
        save_panel._template_combo.setCurrentText(viewer.layers[0].name)
        da = save_panel._build_da()
        validate_voxeldata(da)
        assert da.dims == ("k", "j", "i")
        assert da.fusi.spacing == sample_voxeldata_3dt.isel(time=0).fusi.spacing
        assert da.fusi.origin == sample_voxeldata_3dt.isel(time=0).fusi.origin

    def test_reconstruct_without_template_yields_valid_voxeldata(
        self, save_panel, labels_layer, sample_voxeldata_3dt
    ):
        save_panel._layer_combo.setCurrentText(labels_layer.name)
        save_panel._template_combo.setCurrentIndex(-1)
        da = save_panel._build_da()
        validate_voxeldata(da)
        assert da.dims == ("k", "j", "i")
        assert da.fusi.spacing == sample_voxeldata_3dt.isel(time=0).fusi.spacing
        assert da.fusi.origin == sample_voxeldata_3dt.isel(time=0).fusi.origin
        assert (
            da.coords["x"].attrs["units"]
            == sample_voxeldata_3dt.coords["x"].attrs["units"]
        )
