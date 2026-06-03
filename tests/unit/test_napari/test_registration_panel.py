"""Unit tests for the napari registration panel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def registration_panel(viewer):
    from confusius._napari._registration._panel import RegistrationPanel

    return RegistrationPanel(viewer)


@dataclass(frozen=True)
class _FakeDiagnostics:
    metric: str = "correlation"
    metric_values: np.ndarray = field(default_factory=lambda: np.array([-1.0]))
    final_metric_value: float = -1.0
    n_iterations: int = 1
    stop_condition: str = "done"


class TestRefreshLayers:
    def test_combo_populated_on_layer_add(self, viewer, registration_panel):
        assert registration_panel._moving_combo.count() == 0
        viewer.add_image(np.zeros((4, 6, 8)), name="vol")
        assert registration_panel._moving_combo.count() == 1
        assert registration_panel._moving_combo.itemText(0) == "vol"


class TestOperationMode:
    def test_volumewise_hides_fixed_selector(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        assert registration_panel._fixed_combo.isHidden()
        assert not registration_panel._reference_time_spin.isHidden()
        assert not registration_panel._n_jobs_spin.isHidden()

    def test_volume_shows_fixed_selector(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._single_volume_radio.setChecked(True)
        assert not registration_panel._fixed_combo.isHidden()
        assert registration_panel._reference_time_spin.isHidden()
        assert registration_panel._n_jobs_spin.isHidden()

    def test_defaults_transform_to_rigid(self, registration_panel):
        assert registration_panel._transform_combo.currentText() == "rigid"
        registration_panel._time_series_radio.setChecked(True)
        assert registration_panel._transform_combo.currentText() == "rigid"

    def test_learning_rate_auto_disables_spinbox(self, registration_panel):
        assert registration_panel._learning_rate_auto_check.isChecked()
        assert not registration_panel._learning_rate_spin.isEnabled()
        registration_panel._learning_rate_auto_check.setChecked(False)
        assert registration_panel._learning_rate_spin.isEnabled()


class TestLayerToDataArray:
    def test_reconstructs_dataarray_from_generic_layer(self, viewer):
        from confusius._napari._registration._panel import _layer_to_dataarray

        layer = viewer.add_image(
            np.zeros((3, 5, 7), dtype=np.float32),
            name="plain",
            scale=(0.3, 0.2, 0.1),
            translate=(1.0, 2.0, 3.0),
        )
        layer.axis_labels = ("z", "y", "x")
        layer.units = ("mm", "mm", "mm")

        da = _layer_to_dataarray(layer)

        assert da.dims == ("z", "y", "x")
        assert da.coords["z"][0] == pytest.approx(1.0)
        assert da.coords["y"][1] == pytest.approx(2.2)
        assert da.coords["x"][2] == pytest.approx(3.2)
        assert da.coords["x"].attrs["units"] in {"mm", "millimeter"}


class TestPluginWidget:
    def test_registration_panel_is_present_in_main_widget(self, viewer):
        from confusius._napari._widget import ConfUSIusWidget

        widget = ConfUSIusWidget(viewer)

        assert "Registration" in widget._accordion_panels


class TestFinishedCallbacks:
    def test_volume_result_adds_new_layer_with_transform_metadata(
        self, viewer, registration_panel
    ):
        fixed = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        registered = fixed.copy()
        transform = np.eye(3)
        diagnostics = _FakeDiagnostics()

        payload = {
            "operation": "register_volume",
            "moving_layer_name": "moving",
            "fixed_layer_name": "fixed",
            "transform": "rigid",
            "metric": "correlation",
            "learning_rate": "auto",
            "number_of_iterations": 100,
            "use_multi_resolution": False,
            "resample_interpolation": "linear",
        }

        registration_panel._on_registration_finished(
            payload,
            (registered, transform, diagnostics),
        )

        layer = viewer.layers["moving → fixed"]
        assert layer.metadata["registration_transform"] is transform
        assert layer.metadata["registration_diagnostics"] is diagnostics
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volume"
        )

    def test_volumewise_result_adds_registered_layer(self, viewer, registration_panel):
        registered = xr.DataArray(
            np.ones((3, 4, 6), dtype=np.float32),
            dims=["time", "y", "x"],
            coords={
                "time": xr.DataArray(np.arange(3), dims=["time"]),
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
            attrs={"motion_params": object()},
        )

        payload = {
            "operation": "register_volumewise",
            "moving_layer_name": "series",
            "transform": "rigid",
            "metric": "correlation",
            "learning_rate": "auto",
            "number_of_iterations": 100,
            "use_multi_resolution": False,
            "resample_interpolation": "linear",
            "reference_time": 1,
        }

        registration_panel._on_registration_finished(payload, registered)

        layer = viewer.layers["series registered"]
        assert layer.metadata["reference_time"] == 1
        assert layer.metadata["registration_operation"] == "register_volumewise"
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volumewise"
        )
