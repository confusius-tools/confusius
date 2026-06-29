"""Unit tests for the napari registration panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

import numpy as np
import pytest
import xarray as xr
from qtpy.QtWidgets import QApplication

from confusius._napari._registration._transforms import (
    affine_transform_from_payload,
    bspline_transform_from_payload,
    load_affine_transform_payload,
    load_transform_payload,
    make_affine_transform_payload,
    make_bspline_transform_payload,
    output_grid_from_payload,
    save_affine_transform_payload,
    save_transform_payload,
)
from confusius.registration import resample_like


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


def _make_bspline_transform() -> xr.DataArray:
    return xr.DataArray(
        np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
        dims=["component", "y", "x"],
        coords={
            "component": xr.DataArray([0, 1], dims=["component"]),
            "y": xr.DataArray(np.arange(3) * 0.2, dims=["y"]),
            "x": xr.DataArray(np.arange(4) * 0.1, dims=["x"]),
        },
        attrs={
            "transform_type": "bspline_transform",
            "order": 3,
            "direction": [[1.0, 0.0], [0.0, 1.0]],
            "affines": {"bspline_initialization": np.eye(3).tolist()},
        },
    )


class TestRefreshLayers:
    def test_combo_populated_on_layer_add(self, viewer, registration_panel):
        assert registration_panel._moving_combo.count() == 0
        viewer.add_image(np.zeros((4, 6, 8)), name="vol")
        assert registration_panel._moving_combo.count() == 1
        assert registration_panel._moving_combo.itemText(0) == "vol"

    def test_ignores_lazy_non_numpy_layers(self, viewer, registration_panel):
        import dask.array as da

        viewer.add_image(np.zeros((4, 6, 8)), name="vol")
        viewer.add_image(da.zeros((5, 4, 6), chunks=(1, 4, 6)), name="video")

        registration_panel._refresh_layers()

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

    def test_transform_target_label_is_apply_to(self, registration_panel):
        label = registration_panel._transforms_panel.layout().labelForField(
            registration_panel._transform_target_combo
        )
        assert label is not None
        assert label.text() == "Apply to"

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
        registration_panel._scale_combo.setCurrentText("square root")

        registration_panel._single_volume_radio.setChecked(True)
        registration_panel._learning_rate_edit.setValue(0.42)
        registration_panel._scale_combo.setCurrentText("none")
        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._learning_rate_auto_check.isChecked()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.23)
        assert registration_panel._n_jobs_spin.value() == 3
        assert registration_panel._scale_combo.currentText() == "square root"

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
        assert registration_panel._iterations_spin.singleStep() == 100

    def test_scale_defaults_to_db(self, registration_panel):
        assert registration_panel._scale_combo.currentText() == "decibel"

    def test_scale_preprocessing_resets_gamma_for_previews(self, viewer, registration_panel):
        moving_data = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        fixed = xr.DataArray(
            2 * np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords=moving_data.coords,
        )
        moving = viewer.add_image(moving_data.values, name="moving")
        fixed_layer = viewer.add_image(fixed.values, name="fixed")
        moving.gamma = 0.4
        fixed_layer.gamma = 0.6

        registration_panel._setup_volume_progress(
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
            scale_mode="sqrt",
        )
        assert viewer.layers["Fixed"].gamma == pytest.approx(1.0)
        assert viewer.layers["Moving"].gamma == pytest.approx(1.0)
        assert viewer.layers["Registered (rigid)"].gamma == pytest.approx(1.0)

        registration_panel._teardown_volume_progress()
        registration_panel._setup_volume_progress(
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
            scale_mode="off",
        )
        assert viewer.layers["Fixed"].gamma == pytest.approx(0.6)
        assert viewer.layers["Moving"].gamma == pytest.approx(0.4)
        assert viewer.layers["Registered (rigid)"].gamma == pytest.approx(0.4)

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

    def test_mesh_size_is_basic_and_only_visible_for_bspline(self, registration_panel):
        assert registration_panel._mesh_size_row.parent() is not None
        assert registration_panel._mesh_size_row.isHidden()
        assert registration_panel._mesh_size_z_spin.value() == 10
        assert registration_panel._mesh_size_y_spin.value() == 10
        assert registration_panel._mesh_size_x_spin.value() == 10

        registration_panel._transform_combo.setCurrentText("bspline")
        assert not registration_panel._mesh_size_row.isHidden()

        registration_panel._mesh_size_z_spin.setValue(5)
        registration_panel._mesh_size_y_spin.setValue(7)
        registration_panel._mesh_size_x_spin.setValue(9)
        assert registration_panel._mesh_size_z_spin.value() == 5
        assert registration_panel._mesh_size_y_spin.value() == 7
        assert registration_panel._mesh_size_x_spin.value() == 9

        registration_panel._transform_combo.setCurrentText("rigid")
        assert registration_panel._mesh_size_row.isHidden()

        registration_panel._time_series_radio.setChecked(True)
        assert registration_panel._mesh_size_row.isHidden()


class TestRunRegistration:
    def test_between_scan_run_uses_selected_initial_transform(
        self, viewer, registration_panel, monkeypatch
    ):
        moving = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        fixed = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        affine = np.array(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        transform_payload = make_affine_transform_payload(
            affine,
            reference=fixed,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="rigid",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )

        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})
        viewer.add_image(
            fixed.values,
            name="Previous registered",
            metadata={"confusius_transform": transform_payload},
        )
        registration_panel._refresh_layers()
        registration_panel._refresh_transform_controls()
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")
        registration_panel._scale_combo.setCurrentText("square root")
        for i in range(registration_panel._initialization_combo.count()):
            if registration_panel._initialization_combo.itemData(i) == (
                "layer",
                "Previous registered",
            ):
                registration_panel._initialization_combo.setCurrentIndex(i)
                break

        captured: dict[str, object] = {}

        class _FakeSignal:
            def connect(self, _slot):
                return None

        class _FakeWorker:
            def __init__(self) -> None:
                self.returned = _FakeSignal()
                self.errored = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                return None

        def _fake_thread_worker(func):
            def _runner(*args, **kwargs):
                captured["func"] = func
                captured["args"] = args
                captured["kwargs"] = kwargs
                return _FakeWorker()

            return _runner

        monkeypatch.setattr(
            "confusius._napari._registration._panel.thread_worker",
            _fake_thread_worker,
        )
        monkeypatch.setattr(
            registration_panel,
            "_setup_volume_progress",
            lambda **_kwargs: None,
        )

        registration_panel._run_registration()

        np.testing.assert_array_equal(captured["kwargs"]["initial_transform"], affine)
        assert captured["kwargs"]["center_initialization"] is None
        np.testing.assert_allclose(captured["args"][0].values, np.sqrt(moving.values))
        np.testing.assert_allclose(captured["args"][1].values, np.sqrt(fixed.values))
        assert registration_panel._worker is not None

    def test_between_scan_run_uses_selected_manual_napari_transform(
        self, viewer, registration_panel, monkeypatch
    ):
        moving = xr.DataArray(
            np.zeros((2, 4, 6, 8), dtype=np.float32),
            dims=["time", "z", "y", "x"],
            coords={
                "time": xr.DataArray(np.arange(2), dims=["time"]),
                "z": xr.DataArray(np.arange(4) * 0.3, dims=["z"]),
                "y": xr.DataArray(np.arange(6) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(8) * 0.1, dims=["x"]),
            },
        )
        fixed = xr.DataArray(
            np.ones((2, 4, 6, 8), dtype=np.float32),
            dims=["time", "z", "y", "x"],
            coords=moving.coords,
        )

        moving_layer = viewer.add_image(
            moving.values,
            name="moving",
            metadata={"xarray": moving},
        )
        viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})

        manual_affine = np.eye(5)
        manual_affine[0, 4] = 9.0
        manual_affine[1, 4] = 0.5
        manual_affine[2, 4] = -0.25
        manual_affine[3, 3] = 1.1
        manual_affine[3, 4] = 0.75
        moving_layer.affine = manual_affine

        registration_panel._refresh_layers()
        registration_panel._refresh_transform_controls()
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")
        for i in range(registration_panel._initialization_combo.count()):
            if registration_panel._initialization_combo.itemData(i) == (
                "manual",
                "moving",
            ):
                registration_panel._initialization_combo.setCurrentIndex(i)
                break

        captured: dict[str, object] = {}

        class _FakeSignal:
            def connect(self, _slot):
                return None

        class _FakeWorker:
            def __init__(self) -> None:
                self.returned = _FakeSignal()
                self.errored = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                return None

        def _fake_thread_worker(func):
            def _runner(*args, **kwargs):
                captured["func"] = func
                captured["args"] = args
                captured["kwargs"] = kwargs
                return _FakeWorker()

            return _runner

        monkeypatch.setattr(
            "confusius._napari._registration._panel.thread_worker",
            _fake_thread_worker,
        )
        monkeypatch.setattr(
            registration_panel,
            "_setup_volume_progress",
            lambda **_kwargs: None,
        )

        registration_panel._run_registration()

        expected = np.array(
            [
                [1.0, 0.0, 0.0, -0.5],
                [0.0, 1.0, 0.0, 0.25],
                [0.0, 0.0, 1.0 / 1.1, -0.75 / 1.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(captured["kwargs"]["initial_transform"], expected)
        assert captured["kwargs"]["center_initialization"] is None
        assert captured["args"][0].dims == ("z", "y", "x")
        assert registration_panel._worker is not None


class TestAbort:
    def test_abort_sets_cancellation_event(self, registration_panel):
        registration_panel._worker = object()
        registration_panel._abort_event = Event()
        registration_panel._begin_work()

        assert registration_panel._run_btn.isHidden()
        assert not registration_panel._abort_btn.isHidden()

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

    def test_between_scans_accepts_time_series_by_averaging(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((3, 4, 6), dtype=np.float32),
            dims=["time", "y", "x"],
            coords={
                "time": xr.DataArray(np.arange(3), dims=["time"]),
                "y": xr.DataArray(np.arange(4), dims=["y"]),
                "x": xr.DataArray(np.arange(6), dims=["x"]),
            },
        )
        fixed = xr.DataArray(
            np.ones((3, 4, 6), dtype=np.float32),
            dims=["time", "y", "x"],
            coords={
                "time": xr.DataArray(np.arange(3), dims=["time"]),
                "y": xr.DataArray(np.arange(4), dims=["y"]),
                "x": xr.DataArray(np.arange(6), dims=["x"]),
            },
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})
        registration_panel._refresh_layers()
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")

        assert registration_panel._validate_registration_selection()

    def test_initial_transform_dropdown_lists_available_transforms(
        self, viewer, registration_panel
    ):
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

        viewer.add_image(reference.values, name="Registered", metadata={"confusius_transform": payload})
        registration_panel._refresh_transform_controls()

        assert registration_panel._initialization_combo.itemText(0) == "center_geometry"
        assert registration_panel._initialization_combo.count() >= 4
        assert any(
            registration_panel._initialization_combo.itemData(i)
            == ("layer", "Registered")
            for i in range(registration_panel._initialization_combo.count())
        )

    def test_initial_transform_dropdown_lists_manual_napari_transforms(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((2, 4, 6), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": xr.DataArray(np.arange(2), dims=["z"]),
                "y": xr.DataArray(np.arange(4), dims=["y"]),
                "x": xr.DataArray(np.arange(6), dims=["x"]),
            },
        )
        layer = viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        manual_affine = np.eye(4)
        manual_affine[0, 3] = 1.0
        layer.affine = manual_affine

        registration_panel._refresh_transform_controls()

        assert any(
            registration_panel._initialization_combo.itemData(i)
            == ("manual", "moving")
            for i in range(registration_panel._initialization_combo.count())
        )

    def test_transform_source_dropdown_lists_manual_napari_transforms(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((2, 4, 6), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": xr.DataArray(np.arange(2), dims=["z"]),
                "y": xr.DataArray(np.arange(4), dims=["y"]),
                "x": xr.DataArray(np.arange(6), dims=["x"]),
            },
        )
        layer = viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        manual_affine = np.eye(4)
        manual_affine[0, 3] = 1.0
        layer.affine = manual_affine

        registration_panel._refresh_transform_controls()

        assert any(
            registration_panel._transform_source_combo.itemData(i)
            == ("manual", "moving")
            for i in range(registration_panel._transform_source_combo.count())
        )

    def test_initial_transform_dropdown_updates_when_manual_transform_changes(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((2, 4, 6), dtype=np.float32),
            dims=["z", "y", "x"],
            coords={
                "z": xr.DataArray(np.arange(2), dims=["z"]),
                "y": xr.DataArray(np.arange(4), dims=["y"]),
                "x": xr.DataArray(np.arange(6), dims=["x"]),
            },
        )
        layer = viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        registration_panel._refresh_layers()

        assert not any(
            registration_panel._initialization_combo.itemData(i)
            == ("manual", "moving")
            for i in range(registration_panel._initialization_combo.count())
        )

        manual_affine = np.eye(4)
        manual_affine[0, 3] = 1.0
        layer.affine = manual_affine
        QApplication.processEvents()

        assert any(
            registration_panel._initialization_combo.itemData(i)
            == ("manual", "moving")
            for i in range(registration_panel._initialization_combo.count())
        )


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

    def test_bspline_payload_roundtrip(self, tmp_path):
        reference = xr.DataArray(
            np.ones((3, 4), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(3) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(4) * 0.1, dims=["x"]),
            },
        )
        transform = _make_bspline_transform()
        payload = make_bspline_transform_payload(
            transform,
            reference=reference,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="bspline",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )

        path = tmp_path / "bspline.zarr"
        save_transform_payload(path, payload)
        loaded = load_transform_payload(path)

        assert loaded["name"] == "moving → fixed (bspline)"
        assert loaded["kind"] == "bspline"
        assert output_grid_from_payload(loaded)["shape"] == [3, 4]
        xr.testing.assert_identical(
            bspline_transform_from_payload(loaded),
            transform.astype(float),
        )

    def test_bspline_transform_is_not_offered_for_initialization(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((3, 4), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(3) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(4) * 0.1, dims=["x"]),
            },
        )
        payload = make_bspline_transform_payload(
            _make_bspline_transform(),
            reference=moving,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="bspline",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )
        viewer.add_image(
            moving.values,
            name="Registered (bspline)",
            metadata={"xarray": moving, "confusius_transform": payload},
        )

        registration_panel._refresh_transform_controls()

        transform_items = [
            registration_panel._transform_source_combo.itemText(i)
            for i in range(registration_panel._transform_source_combo.count())
        ]
        initialization_items = [
            registration_panel._initialization_combo.itemText(i)
            for i in range(registration_panel._initialization_combo.count())
        ]

        assert "moving → fixed (bspline)" in transform_items
        assert "moving → fixed (bspline)" not in initialization_items

    def test_apply_transform_uses_bspline_payload(self, viewer, registration_panel, monkeypatch):
        moving = xr.DataArray(
            np.zeros((3, 4), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(3) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(4) * 0.1, dims=["x"]),
            },
        )
        payload = make_bspline_transform_payload(
            _make_bspline_transform(),
            reference=moving,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="bspline",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )
        viewer.add_image(
            moving.values,
            name="moving",
            metadata={"xarray": moving},
        )
        viewer.add_image(
            moving.values,
            name="Registered (bspline)",
            metadata={"xarray": moving, "confusius_transform": payload},
        )
        registration_panel._refresh_transform_controls()
        registration_panel._transform_source_combo.setCurrentText("moving → fixed (bspline)")
        registration_panel._transform_target_combo.setCurrentText("moving")

        captured: dict[str, object] = {}

        class _FakeSignal:
            def connect(self, _slot):
                return None

        class _FakeWorker:
            def __init__(self) -> None:
                self.returned = _FakeSignal()
                self.errored = _FakeSignal()
                self.finished = _FakeSignal()

            def start(self) -> None:
                return None

        def _fake_thread_worker(func):
            def _runner(*args, **kwargs):
                captured["func"] = func
                captured["args"] = args
                captured["kwargs"] = kwargs
                return _FakeWorker()

            return _runner

        monkeypatch.setattr(
            "confusius._napari._registration._panel.thread_worker",
            _fake_thread_worker,
        )

        registration_panel._apply_transform()

        assert captured["func"].__name__ == "resample_volume"
        xr.testing.assert_identical(
            captured["args"][1],
            _make_bspline_transform().astype(float),
        )
        assert registration_panel._worker is not None


class TestPluginWidget:
    def test_registration_panel_is_present_in_main_widget(self, viewer):
        from confusius._napari._widget import ConfUSIusWidget

        widget = ConfUSIusWidget(viewer)

        assert "Registration" in widget._accordion_panels


class TestVolumewiseProgress:
    def test_setup_updates_progress_bar_and_output_layer(
        self, viewer, registration_panel
    ):
        from confusius._napari._registration._panel import _get_source_dataarray

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
        moving = _get_source_dataarray(moving_layer)

        progress = registration_panel._setup_volumewise_progress(
            moving_layer=moving_layer,
            moving=moving,
            layer_name="Motion corrected",
            scale_mode="off",
        )

        assert registration_panel._volumewise_progress_layer is not None
        assert registration_panel._volumewise_moving_preview_layer is not None
        assert registration_panel._progress.maximum() == 3
        assert registration_panel._progress.isTextVisible()
        assert registration_panel._progress.minimumHeight() >= 16
        assert moving_layer.colormap.name != "red"

        moving_preview_layer = viewer.layers["Moving"]
        assert moving_preview_layer.colormap.name == "red"
        preview_layer = viewer.layers["Motion corrected"]
        assert preview_layer.colormap.name == "cyan"
        assert preview_layer.blending == "additive"
        np.testing.assert_array_equal(
            np.asarray(preview_layer.data),
            np.full(moving.shape, float(moving.min()), dtype=np.float32),
        )

        assert registration_panel._progress.value() == 0

        frame = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        progress.frame_completed(1, frame, _FakeDiagnostics(n_iterations=2))

        assert registration_panel._progress.value() == 1
        np.testing.assert_array_equal(
            np.asarray(viewer.layers["Motion corrected"].data)[1],
            np.asarray(frame.values),
        )


    def test_frame_completion_updates_frame_progress(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((3, 4, 6), dtype=np.float32),
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
        progress = registration_panel._setup_volumewise_progress(
            moving_layer=moving_layer,
            moving=moving,
            layer_name="Motion corrected",
            scale_mode="off",
        )

        frame = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        progress.frame_completed(0, frame, _FakeDiagnostics(n_iterations=2))
        progress.frame_completed(1, frame, _FakeDiagnostics(n_iterations=1))
        progress.frame_completed(2, frame, _FakeDiagnostics(n_iterations=3))

        assert registration_panel._progress.maximum() == 3
        assert registration_panel._progress.value() == 3


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

        layer = viewer.layers["Registered (rigid)"]
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

    def test_volume_result_adds_bspline_transform_metadata(
        self, viewer, registration_panel
    ):
        fixed = xr.DataArray(
            np.ones((3, 4), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(3) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(4) * 0.1, dims=["x"]),
            },
        )
        registered = fixed.copy()
        transform = _make_bspline_transform()
        diagnostics = _FakeDiagnostics()

        payload = {
            "operation": "register_volume",
            "moving_layer_name": "moving",
            "fixed_layer_name": "fixed",
            "transform": "bspline",
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

        layer = viewer.layers["Registered (bspline)"]
        assert layer.metadata["registration_status"] == "completed"
        assert layer.metadata["confusius_transform"]["kind"] == "bspline"
        xr.testing.assert_identical(
            bspline_transform_from_payload(layer.metadata["confusius_transform"]),
            transform.astype(float),
        )

    def test_volume_result_replaces_preview_layer(
        self, viewer, registration_panel, qtbot
    ):
        """A preview layer created by `_setup_volume_progress` is removed
        after `_on_registration_finished` so the final result is the only
        layer with that name."""
        moving_data = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
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
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
        )
        assert factory is not None
        assert {"Fixed", "Moving", "Registered (rigid)"}.issubset(
            {layer.name for layer in viewer.layers}
        )
        assert registration_panel._progress_layer is not None
        assert registration_panel._progress_bridge is not None
        assert registration_panel._progress_fixed_layer is not None
        assert registration_panel._progress_moving_layer is not None
        # Original layers are left untouched.
        assert fixed_layer.colormap.name != "red"
        assert moving.colormap.name != "cyan"
        assert moving.blending != "additive"
        assert moving.visible
        # Dedicated preview layers carry the registration styling.
        fixed_preview = viewer.layers["Fixed"]
        moving_preview = viewer.layers["Moving"]
        preview_layer = viewer.layers["Registered (rigid)"]
        assert fixed_preview.colormap.name == "red"
        assert moving_preview.colormap.name == "cyan"
        assert moving_preview.blending == "additive"
        assert not moving_preview.visible
        assert preview_layer.colormap.name == "cyan"
        assert preview_layer.blending == "additive"
        assert preview_layer.visible
        np.testing.assert_array_equal(
            np.asarray(preview_layer.data),
            np.asarray(moving.data),
        )
        np.testing.assert_array_equal(
            np.asarray(moving_preview.data),
            np.asarray(preview_layer.data),
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

        # The resampled preview is kept and promoted to the final registered
        # layer so the user can keep reviewing the fixed / moving / result
        # stack after the run.
        assert registration_panel._progress_layer is None
        assert registration_panel._progress_bridge is None
        assert {"Fixed", "Moving", "Registered (rigid)"}.issubset(
            {layer.name for layer in viewer.layers}
        )
        assert not viewer.layers["Moving"].visible
        result_layer = viewer.layers["Registered (rigid)"]
        assert result_layer.colormap.name == "cyan"
        assert result_layer.blending == "additive"
        # Original source layers remain untouched.
        assert moving.visible
        assert moving.colormap.name != "cyan"
        assert moving.blending != "additive"
        assert fixed_layer.colormap.name != "red"
        assert np.array_equal(
            np.asarray(result_layer.data),
            np.asarray(registered.values),
        )

    def test_setup_volume_progress_applies_initial_transform_to_preview_layers(
        self, viewer, registration_panel
    ):
        moving_data = xr.DataArray(
            np.arange(24, dtype=np.float32).reshape(4, 6),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
        moving = viewer.add_image(moving_data.values, name="moving")
        fixed = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords=moving_data.coords,
        )
        fixed_layer = viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="fixed")
        initial_transform = np.array(
            [[1.0, 0.0, 0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

        registration_panel._setup_volume_progress(
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
            initial_transform=initial_transform,
        )

        expected = resample_like(moving_data, fixed, initial_transform)
        np.testing.assert_allclose(
            np.asarray(viewer.layers["Moving"].data),
            np.asarray(expected.data),
        )
        np.testing.assert_allclose(
            np.asarray(viewer.layers["Registered (rigid)"].data),
            np.asarray(expected.data),
        )

    def test_progress_layer_data_updates_on_iteration(
        self, viewer, registration_panel, qtbot
    ):
        """`_update_progress_layer` writes the iterated array into the preview
        layer's data, refreshing the canvas."""
        moving_data = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
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
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
        )
        # The preview is seeded with the moving image resampled onto the
        # fixed grid, so it's visible and meaningful from the start.
        preview_layer = viewer.layers["Registered (rigid)"]
        assert preview_layer.visible

        next_arr = np.full((4, 6), 0.5, dtype=np.float32)
        registration_panel._update_progress_layer(next_arr)

        np.testing.assert_array_equal(
            np.asarray(viewer.layers["Registered (rigid)"].data), next_arr
        )

        # Shape mismatch is silently ignored.
        registration_panel._update_progress_layer(np.zeros((3, 6), dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(viewer.layers["Registered (rigid)"].data), next_arr
        )

        # Teardown removes only the in-flight registered layer while leaving
        # the reusable fixed / moving previews and originals untouched.
        registration_panel._teardown_volume_progress()
        assert registration_panel._progress_layer is None
        assert registration_panel._progress_bridge is None
        assert "Registered (rigid)" not in {layer.name for layer in viewer.layers}
        assert "Fixed" in {layer.name for layer in viewer.layers}
        assert "Moving" in {layer.name for layer in viewer.layers}
        assert moving.visible
        assert moving.colormap.name != "cyan"
        assert moving.blending != "additive"

    def test_setup_creates_metric_plotter_dock(self, viewer, registration_panel):
        """`_setup_volume_progress` lazily creates and docks the metric plotter."""
        moving_data = xr.DataArray(
            np.zeros((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
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
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
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

        layer = viewer.layers["Motion corrected"]
        assert layer.metadata["reference_time"] == 1
        assert layer.metadata["registration_operation"] == "register_volumewise"
        assert "registration_status" not in layer.metadata
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volumewise"
        )

    def test_volumewise_finished_keeps_preview_layers(
        self, viewer, registration_panel
    ):
        moving = xr.DataArray(
            np.zeros((3, 4, 6), dtype=np.float32),
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
        registration_panel._setup_volumewise_progress(
            moving_layer=moving_layer,
            moving=moving,
            layer_name="Motion corrected",
            scale_mode="off",
        )

        registered = xr.DataArray(
            np.ones((3, 4, 6), dtype=np.float32),
            dims=["time", "y", "x"],
            coords=moving.coords,
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

        assert {"Moving", "Motion corrected"}.issubset(
            {layer.name for layer in viewer.layers}
        )
        assert viewer.layers["series"].colormap.name != "red"
        assert viewer.layers["Moving"].colormap.name == "red"

    def test_unique_transform_and_result_names(self, viewer, registration_panel):
        fixed = xr.DataArray(
            np.ones((4, 6), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": xr.DataArray(np.arange(4) * 0.2, dims=["y"]),
                "x": xr.DataArray(np.arange(6) * 0.1, dims=["x"]),
            },
        )
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
        transform = np.eye(3)
        diagnostics = _FakeDiagnostics()

        registration_panel._on_registration_finished(
            payload,
            (fixed.copy(), transform, diagnostics),
        )
        registration_panel._on_registration_finished(
            payload,
            (fixed.copy(), transform, diagnostics),
        )

        assert "Registered (rigid)" in {layer.name for layer in viewer.layers}
        assert "Registered (rigid) [1]" in {layer.name for layer in viewer.layers}
        names = [
            viewer.layers[name].metadata["confusius_transform"]["name"]
            for name in ("Registered (rigid)", "Registered (rigid) [1]")
        ]
        assert names == [
            "moving → fixed (rigid)",
            "moving → fixed (rigid) [1]",
        ]
