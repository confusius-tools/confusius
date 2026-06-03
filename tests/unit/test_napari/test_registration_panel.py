"""Unit tests for the napari registration panel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import xarray as xr

from confusius._napari._registration._transforms import (
    affine_transform_from_payload,
    load_affine_transform_payload,
    make_affine_transform_payload,
    output_grid_from_payload,
    save_affine_transform_payload,
)


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
    def test_panel_switch_shows_one_subpanel(self, registration_panel):
        assert registration_panel._register_panel_radio.isCheckable()
        assert registration_panel._transforms_panel_radio.isCheckable()
        assert not registration_panel._register_panel.isHidden()
        assert registration_panel._transforms_panel.isHidden()
        registration_panel._transforms_panel_radio.setChecked(True)
        registration_panel._on_panel_changed()
        assert registration_panel._register_panel.isHidden()
        assert not registration_panel._transforms_panel.isHidden()

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


class TestValidation:
    def test_same_moving_and_fixed_is_flagged(self, viewer, registration_panel):
        viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="same")
        registration_panel._refresh_layers()
        registration_panel._moving_combo.setCurrentText("same")
        registration_panel._fixed_combo.setCurrentText("same")

        assert not registration_panel._validate_registration_selection()
        assert not registration_panel._layer_validation.isHidden()
        assert "must be different" in registration_panel._layer_validation.text()

    def test_between_scans_with_single_layer_flags_fixed(self, viewer, registration_panel):
        viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="only")
        registration_panel._refresh_layers()
        registration_panel._moving_combo.setCurrentText("only")

        assert not registration_panel._validate_registration_selection()
        assert "must be different" in registration_panel._layer_validation.text()

    def test_within_scan_requires_time_dimension(self, viewer, registration_panel):
        viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="vol")
        registration_panel._refresh_layers()
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._moving_combo.setCurrentText("vol")

        assert not registration_panel._validate_registration_selection()
        assert "Within-scan registration requires" in registration_panel._layer_validation.text()


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


class TestTransforms:
    def test_affine_payload_roundtrip(self, tmp_path):
        reference = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        payload = make_affine_transform_payload(
            np.eye(3),
            reference=reference,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="rigid",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )

        path = tmp_path / "transform.json"
        save_affine_transform_payload(path, payload)
        loaded = load_affine_transform_payload(path)

        assert loaded["source_layer_name"] == "moving"
        assert loaded["name"] == "moving → fixed (rigid)"
        assert output_grid_from_payload(loaded)["shape"] == [4, 6]
        np.testing.assert_array_equal(affine_transform_from_payload(loaded), np.eye(3))


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
        np.testing.assert_array_equal(
            affine_transform_from_payload(layer.metadata["confusius_transform"]),
            transform,
        )
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
