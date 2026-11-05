"""Unit tests for the SavePanel DataArray reconstruction."""

from __future__ import annotations

import pytest

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


def test_template_refuses_a_layer_showing_a_transform_preview(
    viewer, save_panel, sample_voxeldata_3d, monkeypatch
):
    """Preview values must not be written under the template's committed geometry."""
    from confusius._napari._data import _save_panel as save_panel_module
    from confusius._napari._transform._model import TransformState
    from confusius._napari._transform._session import PreviewSession

    messages = []
    monkeypatch.setattr(save_panel_module, "show_error", messages.append)
    _, layer = plot_napari(sample_voxeldata_3d, viewer=viewer, name="moving")
    plot_napari(sample_voxeldata_3d, viewer=viewer, name="template")
    original = layer.data
    session = PreviewSession(layer)
    # A flip keeps the shape, so the template's shape check does not catch it.
    center = sample_voxeldata_3d.fusi.affine.geometry_center.to_numpy()
    session.show(TransformState.identity(center=center).flipped("x", frame="global").matrix())
    save_panel._layer_combo.setCurrentText("moving")
    save_panel._template_combo.setCurrentText("template")

    assert save_panel._build_da() is None
    assert len(messages) == 1
    assert "Apply" in messages[0]

    session.restore()

    da = save_panel._build_da()
    assert da is not None
    assert da.data is original
