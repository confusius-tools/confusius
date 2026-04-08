"""Unit tests for the QCPanel widget.

_time_val_from_layer is the only non-trivial pure-logic method; it converts
the viewer's world coordinate to a physical time value using the layer's
``world_to_data`` transform and xarray coordinate.  Layer combo refresh is
tested to verify that the inserted/removed event connections are wired
correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from confusius.plotting import plot_napari


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
# _time_val_from_layer
# ---------------------------------------------------------------------------


class TestTimeValFromLayer:
    def test_returns_none_for_layer_without_time(self, viewer, qc_panel):
        layer = viewer.add_image(
            np.zeros((4, 6, 8)), metadata={"xarray": None}
        )
        assert qc_panel._time_val_from_layer(layer) is None

    def test_returns_coordinate_value(self, viewer, qc_panel, sample_4d_volume):
        _, layer = plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        viewer.dims.set_current_step(0, 3)
        result = qc_panel._time_val_from_layer(layer)
        assert result == pytest.approx(
            float(sample_4d_volume.coords["time"][3])
        )

    def test_nonuniform_time_resolves_correctly(self, rng, viewer, qc_panel):
        """Non-uniform time spacing is resolved via world_to_data."""
        time_coords = np.array([0.0, 0.5, 2.0, 2.1, 5.0])
        da = xr.DataArray(
            rng.random((5, 4, 6, 8)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(
                    time_coords, dims=["time"], attrs={"units": "s"}
                ),
                "z": xr.DataArray(np.arange(4) * 0.2, dims=["z"]),
                "y": xr.DataArray(np.arange(6) * 0.1, dims=["y"]),
                "x": xr.DataArray(np.arange(8) * 0.05, dims=["x"]),
            },
        )
        with pytest.warns(UserWarning, match="non-uniform spacing"):
            _, layer = plot_napari(
                da, viewer=viewer, show_colorbar=False, show_scale_bar=False
            )

        for step, expected in enumerate(time_coords):
            viewer.dims.set_current_step(0, step)
            result = qc_panel._time_val_from_layer(layer)
            assert result == pytest.approx(expected), (
                f"step {step}: got {result}, expected {expected}"
            )


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
