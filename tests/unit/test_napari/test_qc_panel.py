"""Unit tests for the QCPanel widget.

Tests cover the time dimension helpers (`_time_dim_index`, `_current_time_world`)
that power the QC cursor and click-to-navigate, as well as the layer combo
refresh to verify that inserted/removed event connections are wired correctly.
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
# _time_dim_index / _current_time_world
# ---------------------------------------------------------------------------


class TestTimeDimIndex:
    def test_defaults_to_zero_without_xarray_layers(self, viewer, qc_panel):
        viewer.add_image(np.zeros((4, 6, 8)), metadata={"xarray": None})
        assert qc_panel._time_dim_index() == 0

    def test_finds_time_dim_from_xarray_layer(self, viewer, qc_panel, sample_4d_volume):
        plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        assert qc_panel._time_dim_index() == list(sample_4d_volume.dims).index("time")


class TestCurrentTimeWorld:
    def test_returns_world_coordinate(self, viewer, qc_panel, sample_4d_volume):
        plot_napari(
            sample_4d_volume,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        viewer.dims.set_current_step(0, 3)
        result = qc_panel._current_time_world()
        assert result == pytest.approx(float(viewer.dims.point[0]))

    def test_consistent_with_video_layer(self, rng, viewer, qc_panel):
        """World coordinate is correct even when a video layer is also loaded."""
        time_coords = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        da = xr.DataArray(
            rng.random((5, 4, 6, 8)).astype(np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(time_coords, dims=["time"], attrs={"units": "s"}),
                "z": xr.DataArray(np.arange(4) * 0.2, dims=["z"]),
                "y": xr.DataArray(np.arange(6) * 0.1, dims=["y"]),
                "x": xr.DataArray(np.arange(8) * 0.05, dims=["x"]),
            },
        )
        plot_napari(da, viewer=viewer, show_colorbar=False, show_scale_bar=False)
        # Add a plain image layer without xarray metadata (simulates a video).
        viewer.add_image(rng.random((20, 4, 6, 8)).astype(np.float32), name="video")
        # Select the video layer so it becomes active.
        viewer.layers.selection.active = viewer.layers["video"]

        for step in range(5):
            viewer.dims.set_current_step(0, step)
            result = qc_panel._current_time_world()
            assert result == pytest.approx(float(viewer.dims.point[0]))


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
