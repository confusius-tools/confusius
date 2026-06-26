"""Unit tests for the napari registration panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

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
    status: str = "completed"


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
        assert not registration_panel._n_jobs_row.isHidden()

    def test_parallel_jobs_is_in_advanced_parameters(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._advanced_toggle.setChecked(True)

        assert not registration_panel._n_jobs_row.isHidden()
        assert registration_panel._n_jobs_spin.parent() is not None

    def test_volume_shows_fixed_selector(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._single_volume_radio.setChecked(True)
        assert not registration_panel._fixed_combo.isHidden()
        assert registration_panel._reference_time_spin.isHidden()
        assert registration_panel._n_jobs_row.isHidden()

    def test_defaults_transform_to_rigid(self, registration_panel):
        assert registration_panel._transform_combo.currentText() == "rigid"
        registration_panel._time_series_radio.setChecked(True)
        assert registration_panel._transform_combo.currentText() == "rigid"

    def test_learning_rate_auto_disables_edit(self, registration_panel):
        assert registration_panel._learning_rate_auto_check.isChecked()
        assert not registration_panel._learning_rate_edit.isEnabled()
        registration_panel._learning_rate_auto_check.setChecked(False)
        assert registration_panel._learning_rate_edit.isEnabled()

    def test_volumewise_learning_rate_defaults_to_fixed_0_01(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)

        assert not registration_panel._learning_rate_auto_check.isChecked()
        assert registration_panel._learning_rate_edit.isEnabled()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.01)

    def test_mode_switch_preserves_session_parameters(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._learning_rate_auto_check.setChecked(True)
        registration_panel._learning_rate_edit.setValue(0.23)
        registration_panel._n_jobs_spin.setValue(3)

        registration_panel._single_volume_radio.setChecked(True)
        registration_panel._learning_rate_edit.setValue(0.42)
        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._learning_rate_auto_check.isChecked()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.23)
        assert registration_panel._n_jobs_spin.value() == 3

    def test_advanced_group_is_collapsed_by_default(self, registration_panel):
        assert not registration_panel._advanced_toggle.isChecked()
        assert registration_panel._advanced_content.isHidden()
        assert registration_panel._advanced_toggle.text() == "Advanced"
        registration_panel._advanced_toggle.click()
        assert not registration_panel._advanced_content.isHidden()

    def test_scientific_notation_spinboxes_parse_values(self, registration_panel):
        registration_panel._learning_rate_auto_check.setChecked(False)
        registration_panel._learning_rate_edit.lineEdit().setText("1e-5")
        registration_panel._learning_rate_edit.interpretText()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(1e-5)

        registration_panel._convergence_min_edit.lineEdit().setText("2.5e-7")
        registration_panel._convergence_min_edit.interpretText()
        assert registration_panel._convergence_min_edit.value() == pytest.approx(2.5e-7)

    def test_spinbox_defaults_and_minima(self, registration_panel):
        assert registration_panel._learning_rate_edit.minimum() == pytest.approx(1e-10)
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.1)
        assert registration_panel._convergence_min_edit.minimum() == pytest.approx(
            1e-10
        )
        assert registration_panel._convergence_min_edit.value() == pytest.approx(1e-6)

    def test_metric_specific_rows_follow_metric(self, registration_panel):
        registration_panel._advanced_toggle.setChecked(True)
        assert registration_panel._metric_combo.currentText() == "correlation"
        assert registration_panel._histogram_bins_row.isHidden()

        registration_panel._metric_combo.setCurrentText("mattes_mi")
        assert not registration_panel._histogram_bins_row.isHidden()

    def test_multi_resolution_toggle_hides_dependent_inputs(self, registration_panel):
        registration_panel._advanced_toggle.setChecked(True)
        assert not registration_panel._multi_resolution_check.isChecked()
        assert registration_panel._shrink_factors_row.isHidden()
        assert registration_panel._smoothing_sigmas_row.isHidden()

        registration_panel._multi_resolution_check.setChecked(True)

        assert not registration_panel._shrink_factors_row.isHidden()
        assert not registration_panel._smoothing_sigmas_row.isHidden()

    def test_initialization_is_in_basic_parameters(self, registration_panel):
        assert registration_panel._initialization_combo.parent() is not None


class TestAbort:
    def test_abort_sets_cancellation_event(self, registration_panel):
        registration_panel._worker = object()
        registration_panel._abort_event = Event()
        registration_panel._begin_work()

        registration_panel._abort_registration()

        assert registration_panel._abort_event.is_set()
        assert not registration_panel._abort_btn.isEnabled()
        assert registration_panel._abort_btn.text() == "Aborting…"


class TestValidation:
    def test_same_moving_and_fixed_is_flagged(self, viewer, registration_panel):
        viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="same")
        registration_panel._refresh_layers()
        registration_panel._moving_combo.setCurrentText("same")
        registration_panel._fixed_combo.setCurrentText("same")

        assert not registration_panel._validate_registration_selection()
        assert not registration_panel._layer_validation.isHidden()
        assert "must be different" in registration_panel._layer_validation.text()

    def test_between_scans_with_single_layer_flags_fixed(
        self, viewer, registration_panel
    ):
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
        assert (
            "Within-scan registration requires"
            in registration_panel._layer_validation.text()
        )


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


class TestVolumewiseProgress:
    def test_setup_updates_progress_bar_and_output_layer(
        self, viewer, registration_panel
    ):
        from confusius._napari._registration._panel import _layer_to_dataarray

        moving = xr.DataArray(
            np.linspace(-2.0, 3.0, 3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6),
            dims=["time", "y", "x"],
            coords={
                "time": xr.DataArray(np.arange(3), dims=["time"]),
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        moving_layer = viewer.add_image(
            moving.values,
            name="series",
            metadata={"xarray": moving},
        )
        moving = _layer_to_dataarray(moving_layer)

        progress = registration_panel._setup_volumewise_progress(
            moving_layer=moving_layer,
            moving=moving,
            layer_name="series registered",
            total_iterations_per_frame=5,
        )

        assert registration_panel._volumewise_progress_layer is not None
        assert registration_panel._progress.maximum() == 15
        assert registration_panel._progress.isTextVisible()
        assert registration_panel._progress.minimumHeight() >= 18
        assert moving_layer.colormap.name == "red"

        preview_layer = viewer.layers["series registered"]
        assert preview_layer.colormap.name == "cyan"
        assert preview_layer.blending == "additive"
        np.testing.assert_array_equal(
            np.asarray(preview_layer.data),
            np.full(moving.shape, float(moving.min()), dtype=np.float32),
        )

        progress.iteration(1, 2, 5)
        assert registration_panel._progress.value() == 2

        frame = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        progress.frame_completed(1, frame, _FakeDiagnostics(n_iterations=2))

        np.testing.assert_array_equal(
            np.asarray(viewer.layers["series registered"].data)[1],
            np.asarray(frame.values),
        )


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
        assert layer.metadata["registration_status"] == "completed"
        np.testing.assert_array_equal(
            affine_transform_from_payload(layer.metadata["confusius_transform"]),
            transform,
        )
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volume"
        )

    def test_volume_result_replaces_preview_layer(
        self, viewer, registration_panel, qtbot
    ):
        """A preview layer created by `_setup_volume_progress` is removed
        after `_on_registration_finished` so the final result is the only
        layer with that name."""
        moving = viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="moving")
        fixed = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        fixed_layer = viewer.add_image(np.ones((4, 6), dtype=np.float32), name="fixed")

        factory = registration_panel._setup_volume_progress(
            moving_layer=moving,
            fixed_layer=fixed_layer,
            fixed=fixed,
            layer_name="moving → fixed",
        )
        assert factory is not None
        assert "moving → fixed" in {layer.name for layer in viewer.layers}
        assert registration_panel._progress_layer is not None
        assert registration_panel._progress_bridge is not None
        # The fixed layer is tinted red so the cyan overlay reads as the
        # classic red/cyan alignment view.
        assert fixed_layer.colormap.name == "red"
        # The moving layer is re-tinted cyan + additive, then hidden before
        # the worker starts so the resampled preview never overlaps it.
        assert moving.colormap.name == "cyan"
        assert moving.blending == "additive"
        assert registration_panel._progress_hidden_layer is moving
        assert not moving.visible
        # The preview is rendered in cyan with additive blending and seeded
        # with the moving image resampled onto the fixed grid, so the first
        # frame is a meaningful "unaligned moving on fixed" view rather than
        # a zero-valued blank.
        preview_layer = viewer.layers["moving → fixed"]
        assert preview_layer.colormap.name == "cyan"
        assert preview_layer.blending == "additive"
        assert preview_layer.visible
        np.testing.assert_array_equal(
            np.asarray(preview_layer.data),
            np.asarray(moving.data),
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

        # Preview has been torn down; the result layer is the only match.
        assert registration_panel._progress_layer is None
        assert registration_panel._progress_bridge is None
        # The result layer picks up the same cyan + additive styling so the
        # red/cyan overlay survives past teardown.
        result_layer = viewer.layers["moving → fixed"]
        assert result_layer.colormap.name == "cyan"
        assert result_layer.blending == "additive"
        # The moving layer stays hidden, with its cyan + additive tint, so
        # the registered output remains the visible stand-in.
        assert not moving.visible
        assert moving.colormap.name == "cyan"
        assert moving.blending == "additive"
        assert np.array_equal(
            np.asarray(result_layer.data),
            np.asarray(registered.values),
        )

    def test_progress_layer_data_updates_on_iteration(
        self, viewer, registration_panel, qtbot
    ):
        """`_update_progress_layer` writes the iterated array into the preview
        layer's data, refreshing the canvas."""
        moving = viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="moving")
        fixed = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        fixed_layer = viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="fixed")

        registration_panel._setup_volume_progress(
            moving_layer=moving,
            fixed_layer=fixed_layer,
            fixed=fixed,
            layer_name="moving → fixed",
        )
        # The preview is seeded with the moving image resampled onto the
        # fixed grid, so it's visible and meaningful from the start.
        preview_layer = viewer.layers["moving → fixed"]
        assert preview_layer.visible

        next_arr = np.full((4, 6), 0.5, dtype=np.float32)
        registration_panel._update_progress_layer(next_arr)

        np.testing.assert_array_equal(
            np.asarray(viewer.layers["moving → fixed"].data), next_arr
        )

        # Shape mismatch is silently ignored.
        registration_panel._update_progress_layer(np.zeros((3, 6), dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(viewer.layers["moving → fixed"].data), next_arr
        )

        # Teardown removes the preview; the moving layer stays hidden with
        # its cyan + additive styling.
        registration_panel._teardown_volume_progress()
        assert registration_panel._progress_layer is None
        assert registration_panel._progress_bridge is None
        assert "moving → fixed" not in {layer.name for layer in viewer.layers}
        assert not moving.visible
        assert moving.colormap.name == "cyan"
        assert moving.blending == "additive"

    def test_setup_creates_metric_plotter_dock(self, viewer, registration_panel):
        """`_setup_volume_progress` lazily creates and docks the metric plotter."""
        moving = viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="moving")
        fixed = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        fixed_layer = viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="fixed")

        assert registration_panel._metric_plotter is None
        assert registration_panel._metric_dock is None

        registration_panel._setup_volume_progress(
            moving_layer=moving,
            fixed_layer=fixed_layer,
            fixed=fixed,
            layer_name="moving → fixed",
        )

        assert registration_panel._metric_plotter is not None
        assert registration_panel._metric_dock is not None
        # The plotter is parented to a dock (i.e. it has been re-parented
        # away from its original parent).
        assert registration_panel._metric_plotter.parent() is not None

        # Feeding a value through the bridge populates the plotter's buffer.
        bridge = registration_panel._progress_bridge
        assert bridge is not None
        bridge.metric_updated.emit(0.5)
        # Force a render so the throttled redraw is observed synchronously.
        registration_panel._metric_plotter._render()  # type: ignore[attr-defined]
        assert registration_panel._metric_plotter.metric_values == [0.5]  # type: ignore[attr-defined]

        # Tearing down keeps the plotter (so the user can inspect the trace).
        registration_panel._teardown_volume_progress()
        assert registration_panel._metric_plotter is not None
        assert registration_panel._metric_plotter.metric_values == [0.5]  # type: ignore[attr-defined]

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
        assert "registration_status" not in layer.metadata
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volumewise"
        )
