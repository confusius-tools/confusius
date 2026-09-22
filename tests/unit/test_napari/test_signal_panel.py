"""Unit tests for the SignalPanel x-axis dimension selection.

The x-axis combo must offer exactly the *non-displayed* (slider) axes with more
than one element. Displayed axes (the two on screen) and singleton axes are
never valid x-axis choices. These tests use the `make_napari_viewer` fixture
so `viewer.dims.displayed` is populated by a real layer.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def panel(viewer):
    from confusius._napari._signals._panel import SignalPanel

    return SignalPanel(viewer)


class TestAvailableXaxisDims:
    """`_get_available_xaxis_dims` returns the non-displayed, multi-element axes."""

    def test_lists_slider_axis_for_3d_volume(self, viewer, panel, sample_voxeldata_3d):
        # (z, y, x): napari displays (y, x); only z is a slider axis.
        viewer.add_image(sample_voxeldata_3d.values, metadata={"xarray": sample_voxeldata_3d})
        assert panel._get_available_xaxis_dims() == ["k"]

    def test_lists_all_slider_axes_for_4dt_volume(self, viewer, panel, sample_voxeldata_3dt):
        # (time, z, y, x): napari displays (y, x); both time and z are sliders.
        viewer.add_image(sample_voxeldata_3dt.values, metadata={"xarray": sample_voxeldata_3dt})
        assert panel._get_available_xaxis_dims() == ["time", "k"]

    def test_excludes_singleton_slider_axis(self, viewer, panel):
        # time is a singleton slider axis and must not be offered.
        da = xr.DataArray(np.zeros((1, 4, 6, 8)), dims=["time", "k", "j", "i"])
        viewer.add_image(da.values, metadata={"xarray": da})
        assert panel._get_available_xaxis_dims() == ["k"]

    def test_combo_defaults_to_time_when_present(self, viewer, panel, sample_voxeldata_3dt):
        viewer.add_image(sample_voxeldata_3dt.values, metadata={"xarray": sample_voxeldata_3dt})
        panel._refresh_xaxis_combo()
        items = [
            panel._xaxis_combo.itemText(i) for i in range(panel._xaxis_combo.count())
        ]
        assert items == ["time", "k"]
        assert panel._xaxis_combo.currentText() == "time"


class TestCreateLayers:
    """New Points/Labels layers must share the reference image's units and axis labels."""

    @pytest.fixture
    def image_layer(self, viewer, sample_voxeldata_3dt):
        from confusius.plotting.napari import plot_napari

        _, layer = plot_napari(sample_voxeldata_3dt, viewer=viewer)
        return layer

    @pytest.mark.parametrize("method", ["_create_points_layer", "_create_labels_layer"])
    def test_new_layer_matches_reference_geometry(self, viewer, panel, image_layer, method):
        getattr(panel, method)()
        new_layer = viewer.layers[-1]
        assert new_layer.units == image_layer.units[1:]
        assert new_layer.axis_labels == image_layer.axis_labels[1:]
        assert new_layer.scale.tolist() == image_layer.scale[1:].tolist()
        assert new_layer.translate.tolist() == image_layer.translate[1:].tolist()
        # Consistent units keep napari from dropping units for rendering.
        assert viewer.layers.extent.units is not None
