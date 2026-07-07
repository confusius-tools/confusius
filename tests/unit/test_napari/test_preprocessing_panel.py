"""Unit tests for the PreprocessingPanel widget.

Covers source-layer/signal combo population (including live point/label signals from
the signal store), pipeline keyword-argument building from UI state, raw signal
alignment (confounds/sample mask), live-signal re-extraction, and end-to-end Apply
runs (via `qtbot.waitUntil`) that compare the resulting layer against directly-chained
calls to `confusius.timing`/`confusius.spatial`/`confusius.signal`.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from confusius._napari._preprocessing._panel import _align_series
from confusius._napari._signals._store import ImportedSignal, LiveSignal
from confusius.plotting import plot_napari
from confusius.signal import clean
from confusius.spatial import smooth_volume
from confusius.timing import resample_time, resample_to_uniform_time

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def panel(viewer, signals_store):
    from confusius._napari._preprocessing._panel import PreprocessingPanel

    return PreprocessingPanel(viewer, signal_store=signals_store)


def _imported_signal(name: str, x: np.ndarray, y: np.ndarray) -> ImportedSignal:
    return ImportedSignal(
        id=f"imported-{name}",
        name=name,
        x=x,
        y=y,
        visible=True,
        color="#000000",
        source_label="test.csv",
        file_path="test.csv",  # type: ignore[arg-type]
        original_column_name=name,
    )


# ---------------------------------------------------------------------------
# Source / reference combos
# ---------------------------------------------------------------------------


class TestSourceCombo:
    def test_lists_image_layers_with_time_dim(self, viewer, panel, sample_3dt_volume):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        items = [
            panel._source_combo.itemText(i) for i in range(panel._source_combo.count())
        ]
        assert items == ["power_doppler"]

    def test_excludes_layers_without_xarray_metadata(self, viewer, panel):
        viewer.add_image(np.zeros((10, 4, 6, 8)))
        panel._refresh_layer_combos()
        assert panel._source_combo.count() == 0

    def test_excludes_layers_with_singleton_time_dim(self, viewer, panel):
        da = xr.DataArray(np.zeros((1, 4, 6, 8)), dims=["time", "z", "y", "x"])
        viewer.add_image(da.values, name="static", metadata={"xarray": da})
        panel._refresh_layer_combos()
        assert panel._source_combo.count() == 0

    def test_excludes_labels_layers(self, viewer, panel, sample_3dt_volume):
        labels = (sample_3dt_volume.values > 0.5).astype(np.int32)
        viewer.add_labels(labels, name="mask", metadata={"xarray": sample_3dt_volume})
        panel._refresh_layer_combos()
        assert panel._source_combo.count() == 0


class TestReferenceCombo:
    def test_shares_eligibility_with_source_combo(
        self, viewer, panel, sample_3dt_volume
    ):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        items = [
            panel._resample_reference_combo.itemText(i)
            for i in range(panel._resample_reference_combo.count())
        ]
        assert items == ["power_doppler"]


class TestPrefillResampleDefaults:
    def test_prefills_step_from_source_time_coordinate(
        self, viewer, panel, sample_3dt_volume
    ):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")

        # sample_3dt_volume's time coordinate is 10.0 + arange(10) * 0.5.
        assert panel._resample_step_spin.value() == pytest.approx(0.5)

    def test_noop_without_time_coordinate(self, viewer, panel):
        da = xr.DataArray(np.zeros((5, 4, 6, 8)), dims=["time", "z", "y", "x"])
        viewer.add_image(da.values, name="raw", metadata={"xarray": da})
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("raw")
        # No time coordinate to prefill from: spin keeps its construction default.
        assert panel._resample_step_spin.value() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Resample mode / advanced-filter fold toggles
# ---------------------------------------------------------------------------


class TestResampleModeToggle:
    def test_no_resampling_hides_everything(self, panel):
        assert panel._resample_mode_combo.currentText() == "No resampling"
        assert not panel._uniform_widget.isVisibleTo(panel)
        assert not panel._reference_widget.isVisibleTo(panel)
        assert not panel._resample_method_widget.isVisibleTo(panel)

    def test_uniform_grid_shows_step_and_method(self, panel):
        panel._resample_mode_combo.setCurrentText("Uniform grid")
        assert panel._uniform_widget.isVisibleTo(panel)
        assert panel._resample_method_widget.isVisibleTo(panel)
        assert not panel._reference_widget.isVisibleTo(panel)

    def test_match_reference_shows_reference_and_method(self, panel):
        panel._resample_mode_combo.setCurrentText("Match reference layer")
        assert panel._reference_widget.isVisibleTo(panel)
        assert panel._resample_method_widget.isVisibleTo(panel)
        assert not panel._uniform_widget.isVisibleTo(panel)


class TestAdvancedFilterToggle:
    def test_hidden_by_default(self, panel):
        assert not panel._advanced_widget.isVisibleTo(panel)

    def test_shown_when_checked(self, panel):
        panel._advanced_toggle.setChecked(True)
        assert panel._advanced_widget.isVisibleTo(panel)


# ---------------------------------------------------------------------------
# Signal (confounds/sample mask) combos
# ---------------------------------------------------------------------------


class TestSignalCombos:
    def test_lists_none_and_imported_signals(self, panel, signals_store, signals_csv):
        signals_store.import_file(signals_csv)
        panel._refresh_signal_combos()
        items = [
            panel._confounds_combo.itemText(i)
            for i in range(panel._confounds_combo.count())
        ]
        assert items == ["None", "a", "b"]

    def test_refreshes_on_store_change(self, panel, signals_store, signals_csv):
        assert panel._mask_combo.count() == 1  # Just "None".
        signals_store.import_file(signals_csv)
        items = [
            panel._mask_combo.itemText(i) for i in range(panel._mask_combo.count())
        ]
        assert items == ["None", "a", "b"]

    def test_includes_point_and_label_live_signals(self, panel, signals_store):
        signals_store.register_live_signals(
            [
                LiveSignal(
                    id="point-0",
                    name="Point 0",
                    color="#000000",
                    visible=True,
                    source_type="point",
                    source_id=0,
                    layer_name="Points (3D)",
                ),
                LiveSignal(
                    id="label-1",
                    name="Label 1",
                    color="#000000",
                    visible=True,
                    source_type="label",
                    source_id=1,
                    layer_name="Labels (3D)",
                ),
            ]
        )
        panel._refresh_signal_combos()
        items = [
            panel._confounds_combo.itemText(i)
            for i in range(panel._confounds_combo.count())
        ]
        assert items == ["None", "Point 0", "Label 1"]

    def test_excludes_mouse_live_signal(self, panel, signals_store):
        signals_store.register_live_signals(
            [
                LiveSignal(
                    id="mouse-0",
                    name="Cursor",
                    color="#000000",
                    visible=True,
                    source_type="mouse",
                    source_id=None,
                    layer_name=None,
                ),
            ]
        )
        panel._refresh_signal_combos()
        items = [
            panel._confounds_combo.itemText(i)
            for i in range(panel._confounds_combo.count())
        ]
        assert items == ["None"]


# ---------------------------------------------------------------------------
# _align_series
# ---------------------------------------------------------------------------


class TestAlignSeries:
    def test_interpolates_onto_matching_time_grid(self):
        reference = xr.DataArray(
            np.zeros(4), dims="time", coords={"time": [0.0, 1.0, 2.0, 3.0]}
        )
        result = _align_series(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
            reference,
            as_mask=False,
        )
        np.testing.assert_allclose(result.values, [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(result.coords["time"].values, reference["time"])

    def test_thresholds_into_boolean_mask(self):
        reference = xr.DataArray(
            np.zeros(4), dims="time", coords={"time": [0.0, 1.0, 2.0, 3.0]}
        )
        result = _align_series(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            reference,
            as_mask=True,
        )
        assert result.dtype == bool
        np.testing.assert_array_equal(result.values, [False, True, False, True])

    def test_uses_raw_values_without_x(self):
        reference = xr.DataArray(np.zeros(3), dims="time")
        result = _align_series(None, np.array([1.0, 2.0, 3.0]), reference, as_mask=False)
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])

    def test_raises_on_length_mismatch_without_x(self):
        reference = xr.DataArray(np.zeros(3), dims="time")
        with pytest.raises(ValueError, match="no shared 'time' coordinate to align by"):
            _align_series(None, np.array([1.0, 2.0]), reference, as_mask=False)


# ---------------------------------------------------------------------------
# _resolve_raw_signal / _extract_live_series
# ---------------------------------------------------------------------------


class TestResolveRawSignal:
    def test_none_selection_returns_none(self, viewer, panel, sample_3dt_volume):
        _, layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        assert panel._resolve_raw_signal("None", layer) is None
        assert panel._resolve_raw_signal("", layer) is None

    def test_unknown_signal_raises(self, viewer, panel, sample_3dt_volume):
        _, layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        with pytest.raises(ValueError, match="not found"):
            panel._resolve_raw_signal("nope", layer)

    def test_resolves_imported_signal_raw_xy(
        self, viewer, panel, signals_store, signals_csv, sample_3dt_volume
    ):
        signals_store.import_file(signals_csv)  # time=[0,1,2], a=[1,2,3], b=[4,5,6].
        _, layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        x, y = panel._resolve_raw_signal("a", layer)
        np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(y, [1.0, 2.0, 3.0])

    def test_resolves_point_live_signal_by_voxel_position(
        self, viewer, panel, signals_store, sample_3dt_volume
    ):
        _, image_layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        spatial_scale = np.asarray(image_layer.scale)[1:]
        spatial_translate = np.asarray(image_layer.translate)[1:]
        points_layer = viewer.add_points(
            np.array([[1.0, 2.0, 3.0]]),
            name="Points (3D)",
            ndim=3,
            scale=spatial_scale,
            translate=spatial_translate,
        )
        signals_store.register_live_signals(
            [
                LiveSignal(
                    id="point-0",
                    name="Point 0",
                    color="#000000",
                    visible=True,
                    source_type="point",
                    source_id=0,
                    layer_name=points_layer.name,
                ),
            ]
        )

        x, y = panel._resolve_raw_signal("Point 0", image_layer)
        np.testing.assert_allclose(x, sample_3dt_volume["time"].values)
        np.testing.assert_allclose(y, sample_3dt_volume.values[:, 1, 2, 3])

    def test_resolves_label_live_signal_as_mean_trace(
        self, viewer, panel, signals_store, sample_3dt_volume
    ):
        _, image_layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        labels_data = np.zeros(sample_3dt_volume.shape[1:], dtype=np.int32)
        labels_data[0, 0, :2] = 1
        labels_layer = viewer.add_labels(labels_data, name="Labels (3D)")
        signals_store.register_live_signals(
            [
                LiveSignal(
                    id="label-1",
                    name="Label 1",
                    color="#000000",
                    visible=True,
                    source_type="label",
                    source_id=1,
                    layer_name=labels_layer.name,
                ),
            ]
        )

        x, y = panel._resolve_raw_signal("Label 1", image_layer)
        expected = sample_3dt_volume.values[:, 0, 0, :2].mean(axis=-1)
        np.testing.assert_allclose(x, sample_3dt_volume["time"].values)
        np.testing.assert_allclose(y, expected)

    def test_stale_live_signal_layer_raises(
        self, viewer, panel, signals_store, sample_3dt_volume
    ):
        _, image_layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        signals_store.register_live_signals(
            [
                LiveSignal(
                    id="point-0",
                    name="Point 0",
                    color="#000000",
                    visible=True,
                    source_type="point",
                    source_id=0,
                    layer_name="Nonexistent Points",
                ),
            ]
        )
        with pytest.raises(ValueError, match="Could not extract signal"):
            panel._resolve_raw_signal("Point 0", image_layer)


# ---------------------------------------------------------------------------
# _build_clean_kwargs / _build_resample_spec / _build_smooth_kwargs
# ---------------------------------------------------------------------------


class TestBuildCleanKwargs:
    def test_defaults(self, panel):
        kwargs = panel._build_clean_kwargs()
        assert kwargs["detrend_order"] is None  # Detrending is off by default.
        assert kwargs["standardize_method"] is None
        assert kwargs["low_cutoff"] is None
        assert kwargs["high_cutoff"] is None
        assert kwargs["filter_butterworth_kwargs"] == {
            "order": 5,
            "padtype": "odd",
            "padlen": None,
            "uniformity_tolerance": 1e-2,
        }
        assert kwargs["standardize_confounds"] is True
        assert kwargs["ensure_finite"] is False
        assert kwargs["interpolate_method"] == "linear"

    def test_checked_detrend_uses_spin_value(self, panel):
        panel._detrend_check.setChecked(True)
        panel._detrend_order_spin.setValue(2)
        assert panel._build_clean_kwargs()["detrend_order"] == 2

    def test_standardize_combo_maps_to_clean_literal(self, panel):
        panel._standardize_combo.setCurrentText("Z-score")
        assert panel._build_clean_kwargs()["standardize_method"] == "zscore"
        panel._standardize_combo.setCurrentText("Percent signal change")
        assert panel._build_clean_kwargs()["standardize_method"] == "psc"

    def test_cutoffs_only_applied_when_checked(self, panel):
        panel._low_cutoff_check.setChecked(True)
        panel._low_cutoff_spin.setValue(0.02)
        panel._high_cutoff_check.setChecked(True)
        panel._high_cutoff_spin.setValue(0.5)
        kwargs = panel._build_clean_kwargs()
        assert kwargs["low_cutoff"] == pytest.approx(0.02)
        assert kwargs["high_cutoff"] == pytest.approx(0.5)

    def test_padtype_none_option_maps_to_python_none(self, panel):
        panel._padtype_combo.setCurrentText("None")
        assert (
            panel._build_clean_kwargs()["filter_butterworth_kwargs"]["padtype"] is None
        )

    def test_nonzero_padlen_is_forwarded(self, panel):
        panel._padlen_spin.setValue(64)
        assert panel._build_clean_kwargs()["filter_butterworth_kwargs"]["padlen"] == 64


class TestBuildResampleSpec:
    def test_no_resampling_returns_none(self, panel):
        assert panel._build_resample_spec() is None

    def test_uniform_grid_spec_always_uses_recording_bounds(self, panel):
        panel._resample_mode_combo.setCurrentText("Uniform grid")
        panel._resample_step_spin.setValue(0.25)
        mode, kwargs = panel._build_resample_spec()
        assert mode == "Uniform grid"
        assert kwargs == {
            "start": None,
            "stop": None,
            "step": 0.25,
            "method": "linear",
        }

    def test_match_reference_without_reference_raises(self, panel):
        panel._resample_mode_combo.setCurrentText("Match reference layer")
        panel._resample_reference_combo.clear()
        with pytest.raises(ValueError, match="No reference layer selected"):
            panel._build_resample_spec()

    def test_match_reference_spec(self, viewer, panel, sample_3dt_volume):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        panel._resample_mode_combo.setCurrentText("Match reference layer")
        panel._resample_reference_combo.setCurrentText("power_doppler")
        mode, kwargs = panel._build_resample_spec()
        assert mode == "Match reference layer"
        np.testing.assert_allclose(kwargs["new_time"], sample_3dt_volume["time"].values)
        assert kwargs["method"] == "linear"


class TestBuildSmoothKwargs:
    def test_disabled_returns_none(self, panel):
        assert panel._build_smooth_kwargs() is None

    def test_enabled_returns_kwargs(self, panel):
        panel._smooth_enable_check.setChecked(True)
        panel._smooth_fwhm_spin.setValue(0.4)
        panel._smooth_ensure_finite_check.setChecked(True)
        assert panel._build_smooth_kwargs() == {"fwhm": 0.4, "ensure_finite": True}


# ---------------------------------------------------------------------------
# Apply — validation
# ---------------------------------------------------------------------------


class TestApplyValidation:
    def test_no_source_selected_shows_error(self, panel, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "confusius._napari._preprocessing._panel.show_error", calls.append
        )
        panel._apply()
        assert calls == ["No source layer selected."]

    def test_match_reference_without_reference_shows_error(
        self, viewer, panel, monkeypatch, sample_3dt_volume
    ):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._resample_mode_combo.setCurrentText("Match reference layer")
        panel._resample_reference_combo.clear()

        calls = []
        monkeypatch.setattr(
            "confusius._napari._preprocessing._panel.show_error", calls.append
        )
        panel._apply()
        assert calls == ["No reference layer selected."]


# ---------------------------------------------------------------------------
# Apply — end to end
# ---------------------------------------------------------------------------


class TestApplyEndToEnd:
    def test_clean_only_matches_direct_call(
        self, qtbot, viewer, panel, sample_3dt_volume
    ):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._standardize_combo.setCurrentText("Z-score")

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        expected = clean(
            sample_3dt_volume,
            detrend_order=None,
            standardize_method="zscore",
            low_cutoff=None,
            high_cutoff=None,
            filter_butterworth_kwargs={
                "order": 5,
                "padtype": "odd",
                "padlen": None,
                "uniformity_tolerance": 1e-2,
            },
            confounds=None,
            standardize_confounds=True,
            ensure_finite=False,
            sample_mask=None,
            interpolate_method="linear",
        )
        np.testing.assert_allclose(new_layer.data, expected.values, rtol=1e-5, atol=1e-8)

    def test_full_pipeline_order_matches_chained_direct_calls(
        self, qtbot, viewer, panel, sample_3dt_volume
    ):
        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")

        panel._resample_mode_combo.setCurrentText("Uniform grid")
        panel._resample_step_spin.setValue(0.25)

        panel._smooth_enable_check.setChecked(True)
        panel._smooth_fwhm_spin.setValue(0.4)

        panel._detrend_check.setChecked(True)
        panel._detrend_order_spin.setValue(1)

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        resampled = resample_to_uniform_time(
            sample_3dt_volume, start=None, stop=None, step=0.25, method="linear"
        )
        smoothed = smooth_volume(resampled, fwhm=0.4, ensure_finite=False)
        expected = clean(
            smoothed,
            detrend_order=1,
            standardize_method=None,
            low_cutoff=None,
            high_cutoff=None,
            filter_butterworth_kwargs={
                "order": 5,
                "padtype": "odd",
                "padlen": None,
                "uniformity_tolerance": 1e-2,
            },
            confounds=None,
            standardize_confounds=True,
            ensure_finite=False,
            sample_mask=None,
            interpolate_method="linear",
        )
        np.testing.assert_allclose(new_layer.data, expected.values, rtol=1e-5, atol=1e-8)

    def test_imported_confound_matches_direct_clean_call(
        self, qtbot, viewer, panel, signals_store, sample_3dt_volume
    ):
        # Aligned 1:1 with sample_3dt_volume's time coordinate (10.0 + arange(10) * 0.5).
        confound_signal = _imported_signal(
            "motion",
            10.0 + np.arange(10) * 0.5,
            np.sin(np.arange(10) * 0.3),
        )
        signals_store._imported_signals.append(confound_signal)

        plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        panel._refresh_layer_combos()
        panel._refresh_signal_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._confounds_combo.setCurrentText("motion")

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        expected_confounds = xr.DataArray(
            confound_signal.y, dims="time", coords={"time": sample_3dt_volume["time"]}
        )
        expected = clean(
            sample_3dt_volume,
            detrend_order=None,
            standardize_method=None,
            low_cutoff=None,
            high_cutoff=None,
            filter_butterworth_kwargs={
                "order": 5,
                "padtype": "odd",
                "padlen": None,
                "uniformity_tolerance": 1e-2,
            },
            confounds=expected_confounds,
            standardize_confounds=True,
            ensure_finite=False,
            sample_mask=None,
            interpolate_method="linear",
        )
        np.testing.assert_allclose(new_layer.data, expected.values, rtol=1e-5, atol=1e-8)

    def test_point_live_signal_confound_matches_direct_clean_call(
        self, qtbot, viewer, panel, signals_store, sample_3dt_volume
    ):
        _, image_layer = plot_napari(
            sample_3dt_volume, viewer=viewer, show_colorbar=False, show_scale_bar=False
        )
        spatial_scale = np.asarray(image_layer.scale)[1:]
        spatial_translate = np.asarray(image_layer.translate)[1:]
        points_layer = viewer.add_points(
            np.array([[1.0, 2.0, 3.0]]),
            name="Points (3D)",
            ndim=3,
            scale=spatial_scale,
            translate=spatial_translate,
        )
        signals_store.register_live_signals(
            [
                LiveSignal(
                    id="point-0",
                    name="Point 0",
                    color="#000000",
                    visible=True,
                    source_type="point",
                    source_id=0,
                    layer_name=points_layer.name,
                ),
            ]
        )

        panel._refresh_layer_combos()
        panel._refresh_signal_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._confounds_combo.setCurrentText("Point 0")

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        expected_confounds = xr.DataArray(
            sample_3dt_volume.values[:, 1, 2, 3],
            dims="time",
            coords={"time": sample_3dt_volume["time"]},
        )
        expected = clean(
            sample_3dt_volume,
            detrend_order=None,
            standardize_method=None,
            low_cutoff=None,
            high_cutoff=None,
            filter_butterworth_kwargs={
                "order": 5,
                "padtype": "odd",
                "padlen": None,
                "uniformity_tolerance": 1e-2,
            },
            confounds=expected_confounds,
            standardize_confounds=True,
            ensure_finite=False,
            sample_mask=None,
            interpolate_method="linear",
        )
        np.testing.assert_allclose(new_layer.data, expected.values, rtol=1e-5, atol=1e-8)
