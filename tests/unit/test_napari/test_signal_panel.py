"""Unit tests for the SignalPanel x-axis dimension selection.

The x-axis combo must offer exactly the *non-displayed* (slider) axes with more
than one element. Displayed axes (the two on screen) and singleton axes are
never valid x-axis choices. These tests use the ``make_napari_viewer`` fixture
so ``viewer.dims.displayed`` is populated by a real layer.
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

    def test_lists_slider_axis_for_3d_volume(self, viewer, panel, sample_3d_volume):
        # (z, y, x): napari displays (y, x); only z is a slider axis.
        viewer.add_image(sample_3d_volume.values, metadata={"xarray": sample_3d_volume})
        assert panel._get_available_xaxis_dims() == ["z"]

    def test_lists_all_slider_axes_for_4dt_volume(
        self, viewer, panel, sample_3dt_volume
    ):
        # (time, z, y, x): napari displays (y, x); both time and z are sliders.
        viewer.add_image(
            sample_3dt_volume.values, metadata={"xarray": sample_3dt_volume}
        )
        assert panel._get_available_xaxis_dims() == ["time", "z"]

    def test_excludes_singleton_slider_axis(self, viewer, panel):
        # time is a singleton slider axis and must not be offered.
        da = xr.DataArray(np.zeros((1, 4, 6, 8)), dims=["time", "z", "y", "x"])
        viewer.add_image(da.values, metadata={"xarray": da})
        assert panel._get_available_xaxis_dims() == ["z"]

    def test_combo_defaults_to_time_when_present(
        self, viewer, panel, sample_3dt_volume
    ):
        viewer.add_image(
            sample_3dt_volume.values, metadata={"xarray": sample_3dt_volume}
        )
        panel._refresh_xaxis_combo()
        items = [
            panel._xaxis_combo.itemText(i) for i in range(panel._xaxis_combo.count())
        ]
        assert items == ["time", "z"]
        assert panel._xaxis_combo.currentText() == "time"
