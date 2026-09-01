"""Unit tests for the napari registration panel."""

from __future__ import annotations

from threading import Event
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from qtpy.QtWidgets import QApplication, QFormLayout

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius._napari._registration._panel_progress import (
    create_volume_progress_plotter,
    setup_volumewise_progress,
    teardown_volume_progress,
    update_progress_layer,
)
from confusius._napari._registration._panel_results import on_registration_finished
from confusius._napari._registration._panel_transforms import (
    apply_selected_inverse_transform,
    apply_selected_transform,
    refresh_transform_controls,
)
from confusius._napari._registration._transform_payloads import (
    _deserialize_bspline_dataarray,
    _serialize_bspline_dataarray,
    get_affine_transform_from_payload,
    get_bspline_transform_from_payload,
    get_input_grid_from_payload,
    get_output_grid_from_payload,
    load_transform_payload,
    make_affine_transform_payload,
    make_bspline_transform_payload,
    save_transform_payload,
)
from confusius.registration import (
    RegistrationDiagnostics,
    resample_like,
    resample_volume,
)
from confusius.xarray import create_voxeldata


@pytest.fixture
def viewer(make_napari_viewer_proxy):
    return make_napari_viewer_proxy()


@pytest.fixture
def real_viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def registration_panel(viewer):
    from confusius._napari._registration._panel import RegistrationPanel

    return RegistrationPanel(viewer)


@pytest.fixture
def real_registration_panel(real_viewer):
    from confusius._napari._registration._panel import RegistrationPanel

    return RegistrationPanel(real_viewer)


def _FakeDiagnostics(
    *,
    metric: str = "correlation",
    metric_values: np.ndarray | None = None,
    final_metric_value: float = -1.0,
    n_iterations: int = 1,
    stop_condition: str = "done",
    status: str = "completed",
) -> RegistrationDiagnostics:
    return RegistrationDiagnostics(
        metric=cast("Any", metric),
        metric_values=np.array([-1.0]) if metric_values is None else metric_values,
        final_metric_value=final_metric_value,
        n_iterations=n_iterations,
        stop_condition=stop_condition,
        status=cast("Any", status),
    )


def _make_bspline_transform() -> xr.DataArray:
    # fUSI DataArrays always carry all three spatial dims (k singleton for 2D data),
    # matching what sitk_bspline_to_dataarray produces in practice.
    return create_voxeldata(
        np.arange(3 * 1 * 3 * 4, dtype=float).reshape(3, 1, 3, 4),
        dims=("component", "k", "j", "i"),
        extra_coords={"component": np.array(["k", "j", "i"], dtype=np.str_)},
        spacing=(1.0, 0.2, 0.1),
        origin=(0.0, 0.0, 0.0),
        attrs={
            "transform_type": "bspline_transform",
            "order": 3,
            "affines": {"bspline_initialization": np.eye(4).tolist()},
        },
    )


def _install_immediate_thread_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Signal:
        def __init__(self) -> None:
            self._slots: list[Any] = []

        def connect(self, slot: Any) -> None:
            self._slots.append(slot)

        def emit(self, *args: Any) -> None:
            for slot in list(self._slots):
                slot(*args)

    class _Worker:
        def __init__(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
            self._func = func
            self._args = args
            self._kwargs = kwargs
            self.returned = _Signal()
            self.errored = _Signal()
            self.finished = _Signal()

        def start(self) -> None:
            try:
                self.returned.emit(self._func(*self._args, **self._kwargs))
            except Exception as exc:  # noqa: BLE001
                self.errored.emit(exc)
            finally:
                self.finished.emit()

    def _thread_worker(func: Any) -> Any:
        def _runner(*args: Any, **kwargs: Any) -> _Worker:
            return _Worker(func, args, kwargs)

        return _runner

    monkeypatch.setattr(
        "confusius._napari._registration._panel_transforms.thread_worker",
        _thread_worker,
    )


class TestRefreshLayers:
    def test_combo_populated_on_layer_add(self, viewer, registration_panel):
        assert registration_panel._moving_combo.count() == 0
        viewer.add_image(np.zeros((4, 6, 8)), name="vol")
        assert registration_panel._moving_combo.count() == 1
        assert registration_panel._moving_combo.itemText(0) == "vol"

    def test_mask_combos_only_list_labels_layers(self, viewer, registration_panel):
        viewer.add_image(np.zeros((4, 6, 8)), name="vol")
        viewer.add_labels(np.zeros((4, 6, 8), dtype=np.int32), name="mask")

        registration_panel._refresh_layers()

        assert registration_panel._fixed_mask_combo.count() == 2
        assert registration_panel._fixed_mask_combo.itemText(0) == ""
        assert registration_panel._fixed_mask_combo.itemText(1) == "mask"
        assert registration_panel._moving_mask_combo.count() == 2
        assert registration_panel._moving_mask_combo.itemText(1) == "mask"

    def test_ignores_lazy_non_numpy_layers(self, viewer, registration_panel):
        import dask.array as da

        viewer.add_image(np.zeros((4, 6, 8)), name="vol")
        viewer.add_image(da.zeros((5, 4, 6), chunks=(1, 4, 6)), name="video")

        registration_panel._refresh_layers()

        assert registration_panel._moving_combo.count() == 1
        assert registration_panel._moving_combo.itemText(0) == "vol"


class TestTransformPayloads:
    def test_bspline_payload_round_trip_preserves_string_component_coords(self):
        transform = _make_bspline_transform()

        payload = _serialize_bspline_dataarray(transform)
        roundtripped = _deserialize_bspline_dataarray(payload)

        np.testing.assert_array_equal(
            roundtripped.coords["component"].values, ["k", "j", "i"]
        )
        np.testing.assert_allclose(
            roundtripped.coords["y"].isel(k=0, i=0).values, [0.0, 0.2, 0.4]
        )
        np.testing.assert_allclose(
            roundtripped.coords["x"].isel(k=0, j=0).values, [0.0, 0.1, 0.2, 0.3]
        )
        np.testing.assert_allclose(roundtripped.values, transform.values)


class TestOperationMode:
    def test_panel_switch_shows_one_subpanel(self, registration_panel):
        assert not registration_panel._register_panel.isHidden()
        assert registration_panel._transforms_panel.isHidden()

        registration_panel._transforms_panel_radio.setChecked(True)
        registration_panel._on_panel_changed()

        assert registration_panel._register_panel.isHidden()
        assert not registration_panel._transforms_panel.isHidden()

    def test_volumewise_disables_fixed_selector(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        assert not registration_panel._fixed_combo.isHidden()
        assert not registration_panel._fixed_combo.isEnabled()
        assert not registration_panel._reference_time_spin.isHidden()
        assert registration_panel._reference_time_spin.isEnabled()
        assert not registration_panel._n_jobs_row.isHidden()

    def test_volumewise_fixed_toggle_swaps_active_target(self, registration_panel):
        assert registration_panel._volumewise_use_fixed_check.isHidden()

        registration_panel._time_series_radio.setChecked(True)

        assert not registration_panel._volumewise_use_fixed_check.isHidden()
        assert not registration_panel._fixed_combo.isEnabled()
        assert not registration_panel._fixed_scale_combo.isEnabled()
        assert registration_panel._reference_time_spin.isEnabled()

        registration_panel._volumewise_use_fixed_check.setChecked(True)

        assert registration_panel._fixed_combo.isEnabled()
        assert registration_panel._fixed_scale_combo.isEnabled()
        assert not registration_panel._reference_time_spin.isEnabled()
        assert not registration_panel._reference_time_label.isEnabled()
        # Both targets stay on screen; only one of them is active.
        assert not registration_panel._fixed_combo.isHidden()
        assert not registration_panel._reference_time_spin.isHidden()

        registration_panel._volumewise_use_fixed_check.setChecked(False)

        assert not registration_panel._fixed_combo.isEnabled()
        assert not registration_panel._fixed_scale_combo.isEnabled()
        assert registration_panel._reference_time_spin.isEnabled()

    def test_volumewise_target_rows_are_ordered_under_moving_layer(
        self, registration_panel
    ):
        """Within-scan rows read moving layer, reference time, then fixed layer."""
        registration_panel._time_series_radio.setChecked(True)
        operation_group = registration_panel._reference_time_label.parentWidget()
        operation_layout = operation_group.layout()
        assert isinstance(operation_layout, QFormLayout)

        def _row(widget):
            row, _role = operation_layout.getWidgetPosition(widget)
            return row

        assert (
            _row(registration_panel._moving_label)
            < _row(registration_panel._reference_time_label)
            < _row(registration_panel._fixed_label_row)
        )
        # The fixed toggle labels its own combo box instead of taking a row.
        assert _row(registration_panel._fixed_label_row) == _row(
            registration_panel._fixed_combo
        )
        assert (
            registration_panel._volumewise_use_fixed_check.parentWidget()
            is registration_panel._fixed_label_row
        )

    def test_switching_to_within_scan_does_not_widen_panel(self, registration_panel):
        # Regression test: a label wider than the between-scan ones grows
        # QFormLayout's label column, which wraps the Mode row's radio pair.
        registration_panel.show()
        QApplication.processEvents()
        between_scan_width = registration_panel.sizeHint().width()

        registration_panel._time_series_radio.setChecked(True)
        QApplication.processEvents()

        # A long within-scan label widens QFormLayout's shared label column: at
        # ~130px extra it pushed the Mode row's radio buttons onto their own line.
        assert registration_panel.sizeHint().width() <= between_scan_width + 16

    def test_volumewise_fixed_choice_survives_mode_switch(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._volumewise_use_fixed_check.setChecked(True)

        registration_panel._single_volume_radio.setChecked(True)

        assert registration_panel._volumewise_use_fixed_check.isHidden()
        assert registration_panel._reference_time_spin.isHidden()
        assert registration_panel._fixed_combo.isEnabled()

        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._volumewise_use_fixed_check.isChecked()
        assert registration_panel._fixed_combo.isEnabled()
        assert not registration_panel._reference_time_spin.isEnabled()

    def test_between_scan_shows_masks_and_sitk_threads(self, registration_panel):
        registration_panel._advanced_toggle.setChecked(True)

        assert not registration_panel._fixed_mask_row.isHidden()
        assert not registration_panel._moving_mask_row.isHidden()
        assert not registration_panel._sitk_threads_row.isHidden()

        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._fixed_mask_row.isHidden()
        assert registration_panel._moving_mask_row.isHidden()
        assert registration_panel._sitk_threads_row.isHidden()

    def test_volume_shows_fixed_selector(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._single_volume_radio.setChecked(True)
        assert not registration_panel._fixed_combo.isHidden()
        assert registration_panel._reference_time_spin.isHidden()
        assert registration_panel._n_jobs_row.isHidden()

    def test_learning_rate_auto_disables_edit(self, registration_panel):
        assert registration_panel._learning_rate_auto_check.isChecked()
        assert not registration_panel._learning_rate_edit.isEnabled()
        registration_panel._learning_rate_auto_check.setChecked(False)
        assert registration_panel._learning_rate_edit.isEnabled()

    def test_volumewise_learning_rate_defaults_to_fixed_value(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._learning_rate_auto_check.isHidden()
        assert not registration_panel._learning_rate_auto_check.isChecked()
        assert registration_panel._learning_rate_edit.isEnabled()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.01)

    def test_mode_switch_preserves_session_parameters(self, registration_panel):
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._learning_rate_edit.setValue(0.23)
        registration_panel._n_jobs_spin.setValue(3)
        registration_panel._scale_combo.setCurrentText("square root")

        registration_panel._single_volume_radio.setChecked(True)
        registration_panel._learning_rate_auto_check.setChecked(True)
        registration_panel._learning_rate_edit.setValue(0.42)
        registration_panel._scale_combo.setCurrentText("none")
        registration_panel._fixed_scale_combo.setCurrentText("square root")
        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._learning_rate_auto_check.isHidden()
        assert not registration_panel._learning_rate_auto_check.isChecked()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.23)
        assert registration_panel._n_jobs_spin.value() == 3
        assert registration_panel._scale_combo.currentText() == "square root"

        registration_panel._single_volume_radio.setChecked(True)

        assert registration_panel._scale_combo.currentText() == "none"
        assert registration_panel._fixed_scale_combo.currentText() == "square root"

    def test_fixed_scale_combo_disabled_within_scan_by_default(
        self, registration_panel
    ):
        assert registration_panel._fixed_scale_combo.isEnabled()
        assert registration_panel._scale_label.text() == "Moving intensity scaling"

        registration_panel._time_series_radio.setChecked(True)

        assert not registration_panel._fixed_scale_combo.isEnabled()
        assert registration_panel._scale_label.text() == "Moving intensity scaling"

    def test_advanced_group_is_collapsed_by_default(self, registration_panel):
        assert not registration_panel._advanced_toggle.isChecked()
        assert registration_panel._advanced_content.isHidden()

        registration_panel._advanced_toggle.click()

        assert not registration_panel._advanced_content.isHidden()

    def test_opening_advanced_group_does_not_widen_panel_minimum(
        self, registration_panel
    ):
        # Regression test for the issue #183 overflow pattern: advanced rows
        # must wrap on narrow docks instead of raising the panel's minimum
        # width, which forced horizontal overflow in the sidebar scroll area.
        registration_panel.show()
        QApplication.processEvents()
        closed_min_width = registration_panel.minimumSizeHint().width()
        registration_panel._advanced_toggle.setChecked(True)
        QApplication.processEvents()
        assert registration_panel.minimumSizeHint().width() <= closed_min_width

    def test_scientific_notation_spinboxes_parse_values(self, registration_panel):
        registration_panel._learning_rate_auto_check.setChecked(False)
        registration_panel._learning_rate_edit.lineEdit().setText("1e-5")
        registration_panel._learning_rate_edit.interpretText()
        assert registration_panel._learning_rate_edit.value() == pytest.approx(1e-5)

        registration_panel._convergence_min_edit.lineEdit().setText("2.5e-7")
        registration_panel._convergence_min_edit.interpretText()
        assert registration_panel._convergence_min_edit.value() == pytest.approx(2.5e-7)

    def test_default_parameter_values(self, registration_panel):
        assert registration_panel._transform_combo.currentText() == "rigid"
        assert registration_panel._scale_combo.currentText() == "decibel"
        assert registration_panel._fixed_scale_combo.currentText() == "decibel"
        assert registration_panel._learning_rate_edit.minimum() == pytest.approx(1e-10)
        assert registration_panel._learning_rate_edit.value() == pytest.approx(0.01)
        assert registration_panel._convergence_min_edit.minimum() == pytest.approx(
            1e-10
        )
        assert registration_panel._convergence_min_edit.value() == pytest.approx(1e-6)
        assert registration_panel._iterations_spin.singleStep() == 100

        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._transform_combo.currentText() == "rigid"

    def test_preview_layers_preserve_source_gamma(
        self, real_viewer, real_registration_panel
    ):
        """Preview layers keep the source layers' gamma.

        Intensity scaling is applied only to the registration optimizer's inputs
        (inside `register_volume`/`register_volumewise`); preview layers always show
        the original, unscaled data, so their gamma is never reset.
        """
        moving_data = create_voxeldata(
            np.ones((4, 6), dtype=np.float32), dims=("j", "i"), spacing=(1.0, 0.2, 0.1)
        )
        fixed = create_voxeldata(
            2 * np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        moving = real_viewer.add_image(moving_data.values, name="moving")
        fixed_layer = real_viewer.add_image(fixed.values, name="fixed")
        moving.gamma = 0.4
        fixed_layer.gamma = 0.6

        create_volume_progress_plotter(
            real_registration_panel,
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
        )
        assert real_viewer.layers["Fixed"].gamma == pytest.approx(0.6)
        assert real_viewer.layers["Moving"].gamma == pytest.approx(0.4)
        assert real_viewer.layers["Registered (rigid)"].gamma == pytest.approx(0.4)

    def test_create_volume_progress_plotter_preserves_camera_view(
        self, real_viewer, real_registration_panel
    ):
        moving_data = create_voxeldata(
            np.ones((5, 4, 6), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.3, 0.2, 0.1),
        )
        fixed = create_voxeldata(
            2 * np.ones((5, 4, 6), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.3, 0.2, 0.1),
        )
        moving = real_viewer.add_image(moving_data.values, name="moving")
        fixed_layer = real_viewer.add_image(fixed.values, name="fixed")

        # User navigates to a custom 3D view before launching the run.
        real_viewer.dims.ndisplay = 3
        real_viewer.scene.camera.center = (1.0, 2.0, 3.0)
        real_viewer.scene.camera.zoom = 7.0
        before = (
            tuple(real_viewer.scene.camera.center),
            real_viewer.scene.camera.zoom,
            real_viewer.dims.ndisplay,
        )

        create_volume_progress_plotter(
            real_registration_panel,
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
        )

        after = (
            tuple(real_viewer.scene.camera.center),
            real_viewer.scene.camera.zoom,
            real_viewer.dims.ndisplay,
        )
        assert after == before

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

    def test_mesh_size_is_basic_and_only_visible_for_bspline(self, registration_panel):
        assert registration_panel._mesh_size_row.isHidden()
        assert registration_panel._optimizer_weights_check.isEnabled()

        registration_panel._transform_combo.setCurrentText("bspline")

        assert not registration_panel._mesh_size_row.isHidden()
        assert not registration_panel._optimizer_weights_check.isEnabled()

        registration_panel._transform_combo.setCurrentText("rigid")

        assert registration_panel._mesh_size_row.isHidden()
        assert registration_panel._optimizer_weights_check.isEnabled()

        registration_panel._time_series_radio.setChecked(True)

        assert registration_panel._mesh_size_row.isHidden()


class TestRunRegistration:
    def test_mask_buttons_create_layers_matching_reference_grid(
        self, viewer, registration_panel
    ):
        moving = create_voxeldata(
            np.zeros((2, 4, 6, 8), dtype=np.float32),
            dims=("time", "k", "j", "i"),
            time=np.arange(2),
            spacing=(0.3, 0.2, 0.1),
            origin=(1.0, 2.0, 3.0),
        )
        fixed = create_voxeldata(
            np.zeros((3, 5, 7), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.5, 0.4, 0.25),
            origin=(10.0, 20.0, 30.0),
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")

        registration_panel._new_moving_mask_btn.click()
        registration_panel._new_fixed_mask_btn.click()

        moving_mask = viewer.layers["Moving mask"]
        fixed_mask = viewer.layers["Fixed mask"]
        assert np.asarray(moving_mask.data).shape == (4, 6, 8)
        assert tuple(moving_mask.axis_labels) == ("z", "y", "x")
        np.testing.assert_allclose(moving_mask.scale, (0.3, 0.2, 0.1))
        np.testing.assert_allclose(moving_mask.translate, (1.0, 2.0, 3.0))
        assert tuple(str(u) for u in moving_mask.units) == ("millimeter",) * 3
        assert np.asarray(fixed_mask.data).shape == (3, 5, 7)
        assert tuple(fixed_mask.axis_labels) == ("z", "y", "x")
        np.testing.assert_allclose(fixed_mask.scale, (0.5, 0.4, 0.25))
        np.testing.assert_allclose(fixed_mask.translate, (10.0, 20.0, 30.0))
        # create_voxeldata always sets "mm" world-coordinate units by default.
        assert tuple(str(u) for u in fixed_mask.units) == ("millimeter",) * 3

    def test_mask_buttons_use_affine_spacing_and_origin_for_oblique_reference(
        self, viewer, registration_panel
    ):
        """Mask scale/translate come from the affine, not raw coordinate diffs.

        Regression test: with an oblique (non-axis-aligned) voxel-to-world index,
        per-dimension coordinate differences give the wrong spacing/origin -- only
        the affine itself (via `.fusi.spacing`/`.fusi.origin`) is correct.
        """
        # Swap k/j in the direction matrix: still a valid affine, but off-diagonal,
        # so `_is_axis_aligned_affine` treats it as oblique.
        direction = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        moving = create_voxeldata(
            np.zeros((4, 6, 8), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.3, 0.2, 0.1),
            origin=(1.0, 2.0, 3.0),
            direction=direction,
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        registration_panel._moving_combo.setCurrentText("moving")

        registration_panel._new_moving_mask_btn.click()

        moving_mask = viewer.layers["Moving mask"]
        expected_spacing = moving.fusi.spacing
        expected_origin = moving.fusi.origin
        world_dim = {"k": "z", "j": "y", "i": "x"}
        np.testing.assert_allclose(
            moving_mask.scale, [expected_spacing[d] for d in ("k", "j", "i")]
        )
        np.testing.assert_allclose(
            moving_mask.translate,
            [expected_origin[world_dim[d]] for d in ("k", "j", "i")],
        )

    def test_mask_buttons_require_selected_reference_layer(
        self, viewer, registration_panel
    ):
        assert not registration_panel._new_moving_mask_btn.isEnabled()
        assert not registration_panel._new_fixed_mask_btn.isEnabled()

        registration_panel._new_moving_mask_btn.click()
        registration_panel._new_fixed_mask_btn.click()

        assert len(viewer.layers) == 0

    def test_between_scan_run_uses_selected_initial_transform(
        self, viewer, registration_panel, monkeypatch
    ):
        moving = create_voxeldata(
            np.zeros((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        fixed = create_voxeldata(
            np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        affine = np.eye(4, dtype=float)
        affine[1, 3] = 0.5
        affine[2, 3] = -0.25
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
        refresh_transform_controls(registration_panel)
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")
        registration_panel._scale_combo.setCurrentText("square root")
        registration_panel._fixed_scale_combo.setCurrentText("none")
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
            "confusius._napari._registration._panel.create_volume_progress_plotter",
            lambda *_args, **_kwargs: None,
        )

        registration_panel._run_registration()

        kwargs = cast("dict[str, Any]", captured["kwargs"])
        args = cast("tuple[Any, ...]", captured["args"])
        np.testing.assert_array_equal(kwargs["initialization"], affine)
        assert kwargs["moving_intensity_scaling"] == "sqrt"
        assert kwargs["fixed_intensity_scaling"] == "none"
        # Scaling happens inside register_volume; the panel passes raw data.
        np.testing.assert_array_equal(args[0].values, moving.values)
        np.testing.assert_array_equal(args[1].values, fixed.values)
        assert registration_panel._worker is not None

    def test_between_scan_run_uses_selected_manual_napari_transform(
        self, viewer, registration_panel, monkeypatch
    ):
        moving = create_voxeldata(
            np.zeros((2, 4, 6, 8), dtype=np.float32),
            dims=("time", "k", "j", "i"),
            time=np.arange(2),
            spacing=(0.3, 0.2, 0.1),
        )
        fixed = create_voxeldata(
            np.ones((2, 4, 6, 8), dtype=np.float32),
            dims=("time", "k", "j", "i"),
            time=np.arange(2),
            spacing=(0.3, 0.2, 0.1),
        )

        moving_layer = viewer.add_image(
            moving.values,
            name="moving",
            metadata={"xarray": moving},
        )
        viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})

        registration_panel._refresh_layers()
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")

        manual_affine = np.eye(5)
        manual_affine[0, 4] = 9.0
        manual_affine[1, 4] = 0.5
        manual_affine[2, 4] = -0.25
        manual_affine[3, 3] = 1.1
        manual_affine[3, 4] = 0.75
        moving_layer.affine = manual_affine
        QApplication.processEvents()

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
            "confusius._napari._registration._panel.create_volume_progress_plotter",
            lambda *_args, **_kwargs: None,
        )

        registration_panel._run_registration()

        kwargs = cast("dict[str, Any]", captured["kwargs"])
        args = cast("tuple[Any, ...]", captured["args"])
        expected = np.array(
            [
                [1.0, 0.0, 0.0, -0.5],
                [0.0, 1.0, 0.0, 0.25],
                [0.0, 0.0, 1.0 / 1.1, -0.75 / 1.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(kwargs["initialization"], expected)
        assert args[0].dims == ("k", "j", "i")
        assert registration_panel._worker is not None

    @pytest.mark.parametrize(
        ("within_scan", "transform", "weights"),
        [
            (False, "rigid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            (True, "translation", [0.7, 0.8, 0.9]),
        ],
    )
    def test_run_passes_mode_specific_worker_kwargs(
        self, viewer, registration_panel, monkeypatch, within_scan, transform, weights
    ):
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

        if within_scan:
            moving = create_voxeldata(
                np.zeros((2, 4, 6, 8), dtype=np.float32),
                dims=("time", "k", "j", "i"),
                time=np.arange(2),
                spacing=(0.3, 0.2, 0.1),
            )
            viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
            registration_panel._time_series_radio.setChecked(True)
            registration_panel._refresh_layers()
            registration_panel._moving_combo.setCurrentText("moving")
            monkeypatch.setattr(
                "confusius._napari._registration._panel.setup_volumewise_progress",
                lambda *_args, **_kwargs: None,
            )
        else:
            moving = create_voxeldata(
                np.zeros((4, 6, 8), dtype=np.float32),
                dims=("k", "j", "i"),
                spacing=(0.3, 0.2, 0.1),
            )
            fixed = moving.copy(data=np.ones((4, 6, 8), dtype=np.float32))
            viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
            viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})
            viewer.add_labels(np.ones((4, 6, 8), dtype=np.int32), name="fixed mask")
            viewer.add_labels(np.ones((4, 6, 8), dtype=np.int32), name="moving mask")
            registration_panel._refresh_layers()
            registration_panel._moving_combo.setCurrentText("moving")
            registration_panel._fixed_combo.setCurrentText("fixed")
            registration_panel._fixed_mask_combo.setCurrentText("fixed mask")
            registration_panel._moving_mask_combo.setCurrentText("moving mask")
            registration_panel._sitk_threads_spin.setValue(3)
            monkeypatch.setattr(
                "confusius._napari._registration._panel.create_volume_progress_plotter",
                lambda *_args, **_kwargs: None,
            )

        registration_panel._transform_combo.setCurrentText(transform)
        registration_panel._optimizer_weights_check.setChecked(True)
        for spin, value in zip(
            registration_panel._optimizer_weight_spins,
            weights,
            strict=False,
        ):
            spin.setValue(value)

        registration_panel._run_registration()

        kwargs = cast("dict[str, Any]", captured["kwargs"])
        assert kwargs["optimizer_weights"] == weights
        if within_scan:
            assert kwargs["reference_time"] == 0
            assert kwargs["fixed"] is None
            assert kwargs["fixed_intensity_scaling"] is None
        else:
            assert kwargs["sitk_threads"] == 3
            assert kwargs["fixed_mask"].dtype == bool
            assert kwargs["moving_mask"].dtype == bool
        assert registration_panel._worker is not None

    def test_within_scan_run_passes_fixed_layer(
        self, viewer, registration_panel, monkeypatch
    ):
        moving = create_voxeldata(
            np.zeros((2, 4, 6, 8), dtype=np.float32),
            dims=("time", "k", "j", "i"),
            time=np.arange(2),
            spacing=(0.3, 0.2, 0.1),
        )
        fixed = create_voxeldata(
            np.ones((4, 6, 8), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.3, 0.2, 0.1),
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(fixed.values, name="target", metadata={"xarray": fixed})
        registration_panel._refresh_layers()
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._volumewise_use_fixed_check.setChecked(True)
        registration_panel._fixed_combo.setCurrentText("target")
        registration_panel._scale_combo.setCurrentText("square root")
        registration_panel._fixed_scale_combo.setCurrentText("none")

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
                captured["args"] = args
                captured["kwargs"] = kwargs
                return _FakeWorker()

            return _runner

        monkeypatch.setattr(
            "confusius._napari._registration._panel.thread_worker",
            _fake_thread_worker,
        )
        monkeypatch.setattr(
            "confusius._napari._registration._panel.setup_volumewise_progress",
            lambda *_args, **_kwargs: None,
        )

        registration_panel._run_registration()

        kwargs = cast("dict[str, Any]", captured["kwargs"])
        assert kwargs["reference_time"] is None
        np.testing.assert_array_equal(kwargs["fixed"].values, fixed.values)
        assert kwargs["fixed_intensity_scaling"] == "none"
        assert kwargs["moving_intensity_scaling"] == "sqrt"
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
    def test_same_moving_and_fixed_is_flagged(
        self, real_viewer, real_registration_panel
    ):
        real_viewer.add_image(np.zeros((4, 6), dtype=np.float32), name="same")
        real_registration_panel._refresh_layers()
        real_registration_panel._moving_combo.setCurrentText("same")
        real_registration_panel._fixed_combo.setCurrentText("same")

        assert not real_registration_panel._validate_registration_selection()
        assert not real_registration_panel._layer_validation.isHidden()
        assert "must be different" in real_registration_panel._layer_validation.text()

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

    def test_within_scan_fixed_toggle_requires_fixed_layer(
        self, viewer, registration_panel, sample_voxeldata_3dt
    ):
        viewer.add_image(
            sample_voxeldata_3dt.values,
            name="moving",
            metadata={"xarray": sample_voxeldata_3dt},
        )
        registration_panel._refresh_layers()
        registration_panel._time_series_radio.setChecked(True)
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._volumewise_use_fixed_check.setChecked(True)
        registration_panel._fixed_combo.setCurrentIndex(-1)

        assert not registration_panel._validate_registration_selection()
        assert not registration_panel._run_btn.isEnabled()
        assert "Select a fixed layer" in registration_panel._layer_validation.text()

        registration_panel._fixed_combo.setCurrentText("moving")

        assert registration_panel._validate_registration_selection()
        assert registration_panel._run_btn.isEnabled()

    def test_between_scans_accepts_time_series_by_averaging(
        self, viewer, registration_panel, sample_voxeldata_3dt
    ):
        moving = sample_voxeldata_3dt.copy(
            data=np.zeros(sample_voxeldata_3dt.shape, dtype=np.float32)
        )
        fixed = sample_voxeldata_3dt.copy(
            data=np.ones(sample_voxeldata_3dt.shape, dtype=np.float32)
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(fixed.values, name="fixed", metadata={"xarray": fixed})
        registration_panel._refresh_layers()
        registration_panel._moving_combo.setCurrentText("moving")
        registration_panel._fixed_combo.setCurrentText("fixed")

        assert registration_panel._validate_registration_selection()


class TestTransforms:
    def test_affine_payload_roundtrip(self, tmp_path):
        reference = create_voxeldata(
            np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        payload = make_affine_transform_payload(
            np.eye(3),
            reference=reference,
            source=reference,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="rigid",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )

        path = tmp_path / "transform.json"
        save_transform_payload(path, payload)
        loaded = load_transform_payload(path)

        assert loaded["source_layer_name"] == "moving"
        assert loaded["name"] == "moving → fixed (rigid)"
        assert get_output_grid_from_payload(loaded)["shape"] == [1, 4, 6]
        input_grid = get_input_grid_from_payload(loaded)
        assert input_grid is not None
        assert input_grid["shape"] == [1, 4, 6]
        np.testing.assert_array_equal(
            get_affine_transform_from_payload(loaded), np.eye(3)
        )

    def test_bspline_payload_missing_voxel_to_world_index_is_rejected(self):
        reference = create_voxeldata(
            np.ones((3, 4), dtype=np.float32), dims=("j", "i"), spacing=(1.0, 0.2, 0.1)
        )
        # A B-spline control-point grid must be a canonical, voxel-to-world-index-backed
        # DataArray; a plain DataArray with world-named dims and no index is rejected.
        transform = xr.DataArray(
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            dims=["component", "y", "x"],
            coords={"component": xr.DataArray(["y", "x"], dims=["component"])},
            attrs={"transform_type": "bspline_transform", "order": 3},
        )

        with pytest.raises(ValueError, match="native voxel dimensions"):
            make_bspline_transform_payload(
                transform,
                reference=reference,
                source=reference,
                source_layer_name="moving",
                target_layer_name="fixed",
                operation="register_volume",
                transform_model="bspline",
                metric="correlation",
                diagnostics=_FakeDiagnostics(),
            )

    def test_bspline_transform_is_not_offered_for_initialization(
        self, viewer, registration_panel
    ):
        moving = create_voxeldata(
            np.zeros((3, 4), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
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

        refresh_transform_controls(registration_panel)

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

    def test_apply_transform_uses_bspline_payload(
        self, viewer, registration_panel, monkeypatch
    ):
        moving = create_voxeldata(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        transform = _make_bspline_transform().astype(float)
        payload = make_bspline_transform_payload(
            transform,
            reference=moving,
            source_layer_name="moving",
            target_layer_name="fixed",
            operation="register_volume",
            transform_model="bspline",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(
            moving.values,
            name="Registered (bspline)",
            metadata={"xarray": moving, "confusius_transform": payload},
        )
        refresh_transform_controls(registration_panel)
        registration_panel._transform_source_combo.setCurrentText(
            "moving → fixed (bspline)"
        )
        registration_panel._transform_target_combo.setCurrentText("moving")
        _install_immediate_thread_worker(monkeypatch)

        output_grid = get_output_grid_from_payload(payload)
        expected = create_voxeldata(
            np.full(output_grid["shape"], 7.0, dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=output_grid["spacing"],
            origin=output_grid["origin"],
        )

        def _fake_resample_volume(*args: Any, **kwargs: Any) -> xr.DataArray:
            return expected

        monkeypatch.setattr(
            "confusius._napari._registration._panel_transforms.resample_volume",
            _fake_resample_volume,
        )

        apply_selected_transform(registration_panel)

        layer = viewer.layers["moving → fixed"]
        result = layer.metadata["xarray"]
        np.testing.assert_array_equal(result.values, expected.values)
        np.testing.assert_allclose(result.coords["y"], expected.coords["y"])
        np.testing.assert_allclose(result.coords["x"], expected.coords["x"])
        assert layer.metadata["transform_source"] == "moving → fixed (bspline)"
        assert layer.metadata["registration_operation"] == "apply_transform"

    def test_apply_transform_layers_keep_grid_units_and_singleton_spacing(
        self, viewer, registration_panel, monkeypatch
    ):
        target = create_voxeldata(
            np.zeros((1, 5, 7), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.4, 0.3, 0.3),
        )
        moving = target.copy()
        payload = make_affine_transform_payload(
            np.eye(4),
            reference=target,
            source=moving,
            source_layer_name="moving",
            target_layer_name="target",
            operation="register_volume",
            transform_model="affine",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )
        viewer.add_image(moving.values, name="moving", metadata={"xarray": moving})
        viewer.add_image(
            target.values,
            name="Registered",
            metadata={"xarray": target, "confusius_transform": payload},
        )
        refresh_transform_controls(registration_panel)
        registration_panel._transform_source_combo.setCurrentText(
            "moving → target (affine)"
        )
        registration_panel._transform_target_combo.setCurrentText("moving")
        _install_immediate_thread_worker(monkeypatch)

        apply_selected_transform(registration_panel)
        apply_selected_inverse_transform(registration_panel)

        for layer_name in ("moving → target", "moving → moving"):
            layer = viewer.layers[layer_name]
            assert layer.blending == "additive"
            np.testing.assert_allclose(layer.scale, (0.4, 0.3, 0.3))
            assert tuple(str(u) for u in layer.units) == ("millimeter",) * 3
            result = layer.metadata["xarray"]
            assert result.coords["y"].attrs["units"] == "mm"

    def test_apply_inverse_transform_uses_inverse_affine_and_input_grid(
        self, viewer, registration_panel, monkeypatch
    ):
        source = create_voxeldata(
            np.zeros((2, 20, 20), dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=(0.3, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
        )
        source.values[:, 5:10, 6:11] = 1.0
        affine = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -0.4],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        target = resample_volume(
            source,
            affine,
            output_sizes=source.sizes,
            output_spacing={"k": 0.3, "j": 0.2, "i": 0.1},
            output_origin={"z": 0.0, "y": 0.0, "x": 0.0},
            output_direction=np.eye(3),
            interpolation="nearest",
        )
        payload = make_affine_transform_payload(
            affine,
            reference=target,
            source=source,
            source_layer_name="source",
            target_layer_name="target",
            operation="register_volume",
            transform_model="affine",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )
        viewer.add_image(source.values, name="source", metadata={"xarray": source})
        viewer.add_image(target.values, name="target", metadata={"xarray": target})
        viewer.add_image(
            target.values,
            name="Registered",
            metadata={"xarray": target, "confusius_transform": payload},
        )
        refresh_transform_controls(registration_panel)
        registration_panel._transform_source_combo.setCurrentText(
            "source → target (affine)"
        )
        registration_panel._transform_target_combo.setCurrentText("target")
        _install_immediate_thread_worker(monkeypatch)

        apply_selected_inverse_transform(registration_panel)

        layer = viewer.layers["target → source"]
        result = layer.metadata["xarray"]
        np.testing.assert_allclose(result.values, source.values, atol=1e-6)
        assert tuple(result.dims) == tuple(source.dims)
        np.testing.assert_allclose(result.coords["y"], source.coords["y"])
        np.testing.assert_allclose(result.coords["x"], source.coords["x"])
        assert layer.metadata["transform_source"] == "source → target (affine)"
        assert layer.metadata["registration_operation"] == "apply_inverse_transform"

    def test_apply_inverse_transform_uses_approximate_inverse_bspline_and_input_grid(
        self, viewer, registration_panel, monkeypatch
    ):
        source = create_voxeldata(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        target = create_voxeldata(
            np.arange(30, dtype=np.float32).reshape(5, 6),
            dims=("j", "i"),
            spacing=(1.0, 0.3, 0.15),
        )
        bspline = _make_bspline_transform().astype(float)
        payload = make_bspline_transform_payload(
            bspline,
            reference=target,
            source=source,
            source_layer_name="source",
            target_layer_name="target",
            operation="register_volume",
            transform_model="bspline",
            metric="correlation",
            diagnostics=_FakeDiagnostics(),
        )
        viewer.add_image(source.values, name="source", metadata={"xarray": source})
        viewer.add_image(target.values, name="target", metadata={"xarray": target})
        viewer.add_image(
            target.values,
            name="Registered",
            metadata={"xarray": target, "confusius_transform": payload},
        )
        refresh_transform_controls(registration_panel)
        registration_panel._transform_source_combo.setCurrentText(
            "source → target (bspline)"
        )
        registration_panel._transform_target_combo.setCurrentText("target")
        _install_immediate_thread_worker(monkeypatch)

        calls: dict[str, Any] = {}
        input_grid = get_input_grid_from_payload(payload)
        assert input_grid is not None
        inverse_field = xr.DataArray(
            np.zeros((2, *input_grid["shape"]), dtype=np.float32),
            dims=["component", *input_grid["dims"]],
            coords={
                "component": xr.DataArray([0, 1], dims=["component"]),
                **{
                    dim: xr.DataArray(
                        input_grid["origin"][i]
                        + np.arange(input_grid["shape"][i]) * input_grid["spacing"][i],
                        dims=[dim],
                    )
                    for i, dim in enumerate(input_grid["dims"])
                },
            },
            attrs={"type": "displacement_field_transform"},
        )
        expected = create_voxeldata(
            np.full(input_grid["shape"], 9.0, dtype=np.float32),
            dims=("k", "j", "i"),
            spacing=input_grid["spacing"],
            origin=input_grid["origin"],
        )

        def _fake_sample_displacement_field(
            transform: Any, **grid: Any
        ) -> xr.DataArray:
            calls["sample_transform"] = transform
            calls["sample_grid"] = grid
            return inverse_field

        def _fake_invert_displacement_field(field: xr.DataArray) -> xr.DataArray:
            calls["invert_field"] = field
            return inverse_field

        def _fake_resample_volume(*args: Any, **kwargs: Any) -> xr.DataArray:
            calls["resample_args"] = args
            calls["resample_kwargs"] = kwargs
            return expected

        monkeypatch.setattr(
            "confusius._napari._registration._panel_transforms.sample_displacement_field",
            _fake_sample_displacement_field,
        )
        monkeypatch.setattr(
            "confusius._napari._registration._panel_transforms.invert_displacement_field",
            _fake_invert_displacement_field,
        )
        monkeypatch.setattr(
            "confusius._napari._registration._panel_transforms.resample_volume",
            _fake_resample_volume,
        )

        apply_selected_inverse_transform(registration_panel)

        assert calls["sample_transform"].identical(bspline)
        assert calls["sample_grid"] == {
            "shape": input_grid["shape"],
            "spacing": input_grid["spacing"],
            "origin": input_grid["origin"],
            "dims": input_grid["dims"],
        }
        assert calls["invert_field"] is inverse_field
        assert calls["resample_args"][0] is target
        assert calls["resample_args"][1] is inverse_field
        resample_kwargs = calls["resample_kwargs"]
        np.testing.assert_allclose(
            resample_kwargs.pop("output_direction"), input_grid["direction"]
        )
        assert resample_kwargs == {
            "output_sizes": dict(zip(VOXEL_DIMS, input_grid["shape"], strict=True)),
            "output_spacing": dict(zip(VOXEL_DIMS, input_grid["spacing"], strict=True)),
            "output_origin": dict(zip(WORLD_DIMS, input_grid["origin"], strict=True)),
            "interpolation": "linear",
        }

        layer = viewer.layers["target → source"]
        result = layer.metadata["xarray"]
        np.testing.assert_array_equal(result.values, expected.values)
        assert layer.metadata["transform_source"] == "source → target (bspline)"
        assert layer.metadata["registration_operation"] == "apply_inverse_transform"
        assert (
            "approximate" in registration_panel._apply_inverse_transform_btn.toolTip()
        )


class TestVolumewiseProgress:
    def test_setup_updates_progress_bar_and_output_layer(
        self, viewer, registration_panel
    ):
        moving = create_voxeldata(
            np.linspace(-2.0, 3.0, 3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6),
            dims=("time", "j", "i"),
            time=np.arange(3, dtype=float),
            spacing=(1.0, 0.2, 0.1),
        )
        moving_layer = viewer.add_image(
            moving.values,
            name="series",
            metadata={"xarray": moving},
        )
        progress = setup_volumewise_progress(
            registration_panel,
            moving_layer=moving_layer,
            moving=moving,
            layer_name="Motion corrected",
        )

        assert registration_panel._volumewise_progress_layer is not None
        assert registration_panel._volumewise_moving_preview_layer is not None
        assert registration_panel._progress.maximum() == 3
        assert registration_panel._progress.value() == 0
        np.testing.assert_array_equal(
            np.asarray(viewer.layers["Motion corrected"].data),
            np.full(moving.shape, float(moving.min()), dtype=np.float32),
        )

        frame = create_voxeldata(
            np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
        )
        progress.frame_completed(1, frame, _FakeDiagnostics(n_iterations=2))

        assert registration_panel._progress.value() == 1
        np.testing.assert_array_equal(
            np.asarray(viewer.layers["Motion corrected"].data)[1],
            np.asarray(frame.values),
        )


class TestFinishedCallbacks:
    def test_volume_result_adds_new_layer_with_transform_metadata(
        self, viewer, registration_panel
    ):
        fixed = create_voxeldata(
            np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
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

        on_registration_finished(
            registration_panel,
            payload,
            (registered, transform, diagnostics),
        )

        layer = viewer.layers["Registered (rigid)"]
        assert layer.metadata["registration_transform"] is transform
        assert layer.metadata["registration_diagnostics"] is diagnostics
        assert layer.metadata["registration_status"] == "completed"
        np.testing.assert_array_equal(
            get_affine_transform_from_payload(layer.metadata["confusius_transform"]),
            transform,
        )
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volume"
        )

    def test_volume_result_adds_bspline_transform_metadata(
        self, viewer, registration_panel
    ):
        fixed = create_voxeldata(
            np.ones((3, 4), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
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

        on_registration_finished(
            registration_panel,
            payload,
            (registered, transform, diagnostics),
        )

        layer = viewer.layers["Registered (bspline)"]
        assert layer.metadata["registration_status"] == "completed"
        assert layer.metadata["confusius_transform"]["kind"] == "bspline"
        xr.testing.assert_identical(
            get_bspline_transform_from_payload(layer.metadata["confusius_transform"]),
            transform.astype(float),
        )

    def test_volume_result_replaces_preview_layer(
        self, real_viewer, real_registration_panel
    ):
        moving_data = create_voxeldata(
            np.zeros((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
        )
        moving = real_viewer.add_image(
            np.zeros((4, 6), dtype=np.float32), name="moving"
        )
        fixed = create_voxeldata(
            np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
        )
        fixed_layer = real_viewer.add_image(
            np.ones((4, 6), dtype=np.float32), name="fixed"
        )

        create_volume_progress_plotter(
            real_registration_panel,
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
        )
        assert {"Fixed", "Moving", "Registered (rigid)"}.issubset(
            {layer.name for layer in real_viewer.layers}
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
        on_registration_finished(
            real_registration_panel,
            payload,
            (registered, transform, diagnostics),
        )

        assert real_registration_panel._progress_layer is None
        assert real_registration_panel._progress_bridge is None
        result_layer = real_viewer.layers["Registered (rigid)"]
        np.testing.assert_array_equal(
            np.asarray(result_layer.data),
            np.asarray(registered.values),
        )
        assert {"Fixed", "Moving", "Registered (rigid)"}.issubset(
            {layer.name for layer in real_viewer.layers}
        )

    def test_create_volume_progress_plotter_applies_initial_transform_to_preview_layers(
        self, real_viewer, real_registration_panel
    ):
        moving_data = create_voxeldata(
            np.arange(24, dtype=np.float32).reshape(1, 4, 6),
            dims=("k", "j", "i"),
            spacing=(0.2, 0.2, 0.1),
        )
        moving = real_viewer.add_image(moving_data.values, name="moving")
        fixed = moving_data.copy(data=np.zeros((1, 4, 6), dtype=np.float32))
        fixed_layer = real_viewer.add_image(
            np.zeros((1, 4, 6), dtype=np.float32), name="fixed"
        )
        initial_transform = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        create_volume_progress_plotter(
            real_registration_panel,
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
            initial_transform=initial_transform,
        )

        expected = resample_like(moving_data, fixed, initial_transform)
        np.testing.assert_allclose(
            np.asarray(real_viewer.layers["Moving"].data),
            np.asarray(expected.data),
        )
        np.testing.assert_allclose(
            np.asarray(real_viewer.layers["Registered (rigid)"].data),
            np.asarray(expected.data),
        )

    def test_progress_layer_data_updates_on_iteration(
        self, real_viewer, real_registration_panel
    ):
        moving_data = create_voxeldata(
            np.zeros((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
        )
        moving = real_viewer.add_image(
            np.zeros((4, 6), dtype=np.float32), name="moving"
        )
        fixed = create_voxeldata(
            np.zeros((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
        )
        fixed_layer = real_viewer.add_image(
            np.zeros((4, 6), dtype=np.float32), name="fixed"
        )

        create_volume_progress_plotter(
            real_registration_panel,
            moving_layer=moving,
            fixed_layer=fixed_layer,
            moving=moving_data,
            fixed=fixed,
            layer_name="Registered (rigid)",
        )

        next_arr = np.full((1, 4, 6), 0.5, dtype=np.float32)
        update_progress_layer(real_registration_panel, next_arr)
        np.testing.assert_array_equal(
            np.asarray(real_viewer.layers["Registered (rigid)"].data), next_arr
        )

        update_progress_layer(
            real_registration_panel, np.zeros((3, 6), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(real_viewer.layers["Registered (rigid)"].data), next_arr
        )

        teardown_volume_progress(real_registration_panel)
        assert real_registration_panel._progress_layer is None
        assert "Registered (rigid)" not in {layer.name for layer in real_viewer.layers}

    def test_volumewise_result_adds_registered_layer(self, viewer, registration_panel):
        moving = create_voxeldata(
            np.zeros((3, 4, 6), dtype=np.float32),
            dims=("time", "j", "i"),
            time=np.arange(3, dtype=float),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
        )
        moving_layer = viewer.add_image(
            moving.values,
            name="series",
            metadata={"xarray": moving},
        )
        setup_volumewise_progress(
            registration_panel,
            moving_layer=moving_layer,
            moving=moving,
            layer_name="Motion corrected",
        )

        registered = moving.copy(data=np.ones((3, 1, 4, 6), dtype=np.float32))
        registered.attrs["motion_params"] = object()
        payload = {
            "operation": "register_volumewise",
            "moving_layer_name": "series",
            "transform": "rigid",
            "metric": "correlation",
            "learning_rate": "auto",
            "number_of_iterations": 100,
            "use_multi_resolution": False,
            "resample_interpolation": "linear",
            "reference_time": None,
            "fixed_layer_name": "target",
        }

        on_registration_finished(registration_panel, payload, registered)

        layer = viewer.layers["Motion corrected"]
        np.testing.assert_array_equal(np.asarray(layer.data), registered.values)
        assert layer.metadata["reference_time"] is None
        assert layer.metadata["fixed_layer_name"] == "target"
        assert layer.metadata["registration_operation"] == "register_volumewise"
        assert {"Moving", "Motion corrected"}.issubset(
            {existing.name for existing in viewer.layers}
        )
        assert "registration_status" not in layer.metadata
        assert (
            layer.metadata["xarray"].attrs["registration_operation"]
            == "register_volumewise"
        )

    def test_unique_transform_and_result_names(self, viewer, registration_panel):
        fixed = create_voxeldata(
            np.ones((4, 6), dtype=np.float32),
            dims=("j", "i"),
            spacing=(1.0, 0.2, 0.1),
            origin=(0.0, 0.0, 0.0),
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

        on_registration_finished(
            registration_panel,
            payload,
            (fixed.copy(), transform, diagnostics),
        )
        on_registration_finished(
            registration_panel,
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
