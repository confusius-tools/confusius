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
from confusius.xarray import create_voxeldata

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

    def test_finds_time_dim_from_xarray_layer(self, viewer, qc_panel, sample_voxeldata_3dt):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        assert qc_panel._time_dim_index() == list(sample_voxeldata_3dt.dims).index("time")


class TestCurrentTimeWorld:
    def test_returns_world_coordinate(self, viewer, qc_panel, sample_voxeldata_3dt):
        plot_napari(
            sample_voxeldata_3dt,
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
        da = create_voxeldata(
            rng.random((5, 4, 6, 8)).astype(np.float32),
            dims=("time", "k", "j", "i"),
            time=xr.DataArray(time_coords, dims=["time"], attrs={"units": "s"}),
            spacing=(0.2, 0.1, 0.05),
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


# ---------------------------------------------------------------------------
# _store_dvars_signal
# ---------------------------------------------------------------------------


class TestStoreDvarsSignal:
    def test_noop_without_a_signal_store(self, viewer):
        from confusius._napari._qc._panel import QCPanel

        panel = QCPanel(viewer)
        dvars = xr.DataArray([0.1, 0.2, 0.3], dims=["time"], coords={"time": [0, 1, 2]})
        panel._store_dvars_signal(dvars, "my_layer")  # Must not raise.

    def test_adds_dvars_to_the_signal_store(self, viewer):
        from confusius._napari._qc._panel import QCPanel
        from confusius._napari._signals._store import SignalStore

        store = SignalStore()
        panel = QCPanel(viewer, signal_store=store)
        dvars = xr.DataArray([0.1, 0.2, 0.3], dims=["time"], coords={"time": [0, 1, 2]})

        panel._store_dvars_signal(dvars, "my_layer")

        stored = store.stored_signals()
        assert len(stored) == 1
        assert stored[0].name == "DVARS (my_layer)"
        np.testing.assert_array_equal(stored[0].y, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(stored[0].x, [0, 1, 2])

    def test_recomputing_updates_in_place_instead_of_duplicating(self, viewer):
        from confusius._napari._qc._panel import QCPanel
        from confusius._napari._signals._store import SignalStore

        store = SignalStore()
        panel = QCPanel(viewer, signal_store=store)
        first = xr.DataArray([0.1, 0.2], dims=["time"], coords={"time": [0, 1]})
        second = xr.DataArray([0.9, 0.8], dims=["time"], coords={"time": [0, 1]})

        panel._store_dvars_signal(first, "my_layer")
        panel._store_dvars_signal(second, "my_layer")

        stored = store.stored_signals()
        assert len(stored) == 1
        np.testing.assert_array_equal(stored[0].y, [0.9, 0.8])
