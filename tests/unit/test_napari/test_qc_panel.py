"""Unit tests for the QCPanel widget.

_time_val_from_da is the only non-trivial pure-logic method; it converts the
viewer's current step index to a physical time value using the DataArray's time
coordinate.  Layer combo refresh is tested to verify that the inserted/removed
event connections are wired correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def qc_panel(viewer):
    from confusius._napari._qc._panel import QCPanel

    return QCPanel(viewer)


# ---------------------------------------------------------------------------
# _time_val_from_da
# ---------------------------------------------------------------------------


class TestTimeValFromDa:
    def test_returns_none_without_time_dim(self, qc_panel):
        da = xr.DataArray(np.zeros((4, 6, 8)), dims=["z", "y", "x"])
        assert qc_panel._time_val_from_da(da) is None

    def test_returns_frame_index_without_time_coordinate(self, viewer, qc_panel):
        da = xr.DataArray(np.zeros((10, 4, 6, 8)), dims=["time", "z", "y", "x"])
        viewer.add_image(da.values)
        viewer.dims.set_current_step(0, 5)
        assert qc_panel._time_val_from_da(da) == pytest.approx(5.0)

    def test_returns_coordinate_value(self, viewer, qc_panel, sample_4d_volume):
        viewer.add_image(sample_4d_volume.values)
        viewer.dims.set_current_step(0, 3)
        result = qc_panel._time_val_from_da(sample_4d_volume)
        assert result == pytest.approx(float(sample_4d_volume.coords["time"][3]))


# ---------------------------------------------------------------------------
# Layer combo refresh
# ---------------------------------------------------------------------------


class TestRefreshLayers:
    def test_combo_populated_on_layer_add(self, viewer, qc_panel):
        assert qc_panel._layer_combo.count() == 0
        viewer.add_image(np.zeros((10, 4, 6, 8)), name="my_layer")
        assert qc_panel._layer_combo.count() == 1
        assert qc_panel._layer_combo.itemText(0) == "my_layer"

    def test_combo_cleared_on_layer_remove(self, viewer, qc_panel):
        layer = viewer.add_image(np.zeros((10, 4, 6, 8)), name="my_layer")
        viewer.layers.remove(layer)
        assert qc_panel._layer_combo.count() == 0
