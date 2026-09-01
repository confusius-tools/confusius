"""Unit tests for the PreprocessingPanel widget.

Covers source-layer/signal combo population (including live point/label signals from
the signal store), pipeline keyword-argument building from UI state, raw signal
alignment (confounds/sample mask), live-signal re-extraction, and end-to-end Apply
runs (via `qtbot.waitUntil`) that compare the resulting layer against directly-chained
calls to `confusius.timing`/`confusius.spatial`/`confusius.signal`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from confusius._napari._preprocessing._panel import (
    _align_series,
    _MaskSpec,
    _threshold_mask,
)
from confusius._napari._signals._store import LiveSignal, StoredSignal
from confusius.plotting import plot_napari
from confusius.signal import clean
from confusius.spatial import smooth_volume
from confusius.timing import resample_to_uniform_time

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


def _imported_signal(name: str, x: np.ndarray, y: np.ndarray) -> StoredSignal:
    return StoredSignal(
        id=f"imported-{name}",
        name=name,
        x=x,
        y=y,
        visible=True,
        color="#000000",
        source_label="test.csv",
        file_path=Path("test.csv"),
        original_column_name=name,
    )


# ---------------------------------------------------------------------------
# Source / reference combos
# ---------------------------------------------------------------------------


class TestSourceCombo:
    def test_lists_image_layers_with_time_dim(
        self, viewer, panel, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
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

    def test_excludes_labels_layers(self, viewer, panel, sample_voxeldata_3dt):
        labels = (sample_voxeldata_3dt.values > 0.5).astype(np.int32)
        viewer.add_labels(
            labels, name="mask", metadata={"xarray": sample_voxeldata_3dt}
        )
        panel._refresh_layer_combos()
        assert panel._source_combo.count() == 0


class TestReferenceCombo:
    def test_shares_eligibility_with_source_combo(
        self, viewer, panel, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel._refresh_layer_combos()
        items = [
            panel._resample_reference_combo.itemText(i)
            for i in range(panel._resample_reference_combo.count())
        ]
        assert items == ["power_doppler"]


class TestPrefillResampleDefaults:
    def test_prefills_step_from_source_time_coordinate(
        self, viewer, panel, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")

        # sample_voxeldata_3dt's time coordinate is 10.0 + arange(10) * 0.5.
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


def _confounds_list_items(panel) -> list[str]:
    return [
        panel._confounds_list.item(i).text()
        for i in range(panel._confounds_list.count())
    ]


def _check_confound(panel, name: str) -> None:
    """Select the confounds-list item with the given name."""
    for i in range(panel._confounds_list.count()):
        item = panel._confounds_list.item(i)
        if item.text() == name:
            item.setSelected(True)
            return
    raise AssertionError(f"No confound item named {name!r}.")


class TestSignalCombos:
    def test_confounds_list_has_no_none_sentinel(
        self, panel, signals_store, signals_csv
    ):
        signals_store.import_file(signals_csv)
        panel._refresh_signal_combos()
        assert _confounds_list_items(panel) == ["a", "b"]

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
        assert _confounds_list_items(panel) == ["Point 0", "Label 1"]

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
        assert _confounds_list_items(panel) == []

    def test_selected_confounds_survive_a_refresh(
        self, panel, signals_store, signals_csv
    ):
        signals_store.import_file(signals_csv)
        panel._refresh_signal_combos()
        panel._confounds_list.item(0).setSelected(True)

        # Trigger another refresh (e.g. a new signal arriving).
        signals_store.pin_signal(
            origin="test",
            name="c",
            x=np.array([0.0]),
            y=np.array([1.0]),
            color="#000000",
            source_label="test",
        )

        assert panel._selected_confound_names() == ["a"]

    def test_selecting_a_range_yields_every_item_in_it(
        self, panel, signals_store, tmp_path
    ):
        # Simulates a shift-click range select spanning several stored signals
        # (e.g. a batch of CompCor components), not just single-item toggling.
        for name in ["a", "b", "c", "d"]:
            path = tmp_path / f"{name}.csv"
            path.write_text(f"time,{name}\n0,1\n1,2\n")
            signals_store.import_file(path)
        panel._refresh_signal_combos()

        selection_model = panel._confounds_list.selectionModel()
        top = panel._confounds_list.model().index(1, 0)
        bottom = panel._confounds_list.model().index(2, 0)
        selection_model.select(
            top,
            selection_model.SelectionFlag.Select | selection_model.SelectionFlag.Rows,
        )
        selection_model.select(
            bottom,
            selection_model.SelectionFlag.Select | selection_model.SelectionFlag.Rows,
        )

        assert set(panel._selected_confound_names()) == {"b", "c"}


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
        )
        np.testing.assert_allclose(result.values, [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(result.coords["time"].values, reference["time"])

    def test_uses_raw_values_without_x(self):
        reference = xr.DataArray(np.zeros(3), dims="time")
        result = _align_series(None, np.array([1.0, 2.0, 3.0]), reference)
        np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0])

    def test_raises_on_length_mismatch_without_x(self):
        reference = xr.DataArray(np.zeros(3), dims="time")
        with pytest.raises(ValueError, match="no shared 'time' coordinate to align by"):
            _align_series(None, np.array([1.0, 2.0]), reference)


# ---------------------------------------------------------------------------
# _threshold_mask
# ---------------------------------------------------------------------------


class TestThresholdMask:
    def test_already_binary_uses_nonzero_as_keep(self):
        aligned = xr.DataArray(np.array([0.0, 1.0, 0.0, 1.0]), dims="time")
        result = _threshold_mask(
            aligned,
            _MaskSpec(x=None, y=aligned.values, threshold=None, keep_above=True),
        )
        assert result.dtype == bool
        np.testing.assert_array_equal(result.values, [False, True, False, True])

    def test_keep_above_threshold(self):
        aligned = xr.DataArray(np.array([0.1, 0.4, 0.2, 0.6]), dims="time")
        result = _threshold_mask(
            aligned,
            _MaskSpec(x=None, y=aligned.values, threshold=0.3, keep_above=True),
        )
        np.testing.assert_array_equal(result.values, [False, True, False, True])

    def test_keep_below_threshold(self):
        aligned = xr.DataArray(np.array([0.1, 0.4, 0.2, 0.6]), dims="time")
        result = _threshold_mask(
            aligned,
            _MaskSpec(x=None, y=aligned.values, threshold=0.3, keep_above=False),
        )
        np.testing.assert_array_equal(result.values, [True, False, True, False])


# ---------------------------------------------------------------------------
# _resolve_raw_signal / _extract_live_series
# ---------------------------------------------------------------------------


class TestResolveRawSignal:
    def test_none_selection_returns_none(self, viewer, panel, sample_voxeldata_3dt):
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        assert panel._resolve_raw_signal("None", layer) is None
        assert panel._resolve_raw_signal("", layer) is None

    def test_unknown_signal_raises(self, viewer, panel, sample_voxeldata_3dt):
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        with pytest.raises(ValueError, match="not found"):
            panel._resolve_raw_signal("nope", layer)

    def test_resolves_imported_signal_raw_xy(
        self, viewer, panel, signals_store, signals_csv, sample_voxeldata_3dt
    ):
        signals_store.import_file(signals_csv)  # time=[0,1,2], a=[1,2,3], b=[4,5,6].
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        x, y = panel._resolve_raw_signal("a", layer)
        np.testing.assert_allclose(x, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(y, [1.0, 2.0, 3.0])

    def test_resolves_point_live_signal_by_voxel_position(
        self, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        _, image_layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
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
        np.testing.assert_allclose(x, sample_voxeldata_3dt["time"].values)
        np.testing.assert_allclose(y, sample_voxeldata_3dt.values[:, 1, 2, 3])

    def test_resolves_label_live_signal_as_mean_trace(
        self, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        _, image_layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        labels_data = np.zeros(sample_voxeldata_3dt.shape[1:], dtype=np.int32)
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
        expected = sample_voxeldata_3dt.values[:, 0, 0, :2].mean(axis=-1)
        np.testing.assert_allclose(x, sample_voxeldata_3dt["time"].values)
        np.testing.assert_allclose(y, expected)

    def test_stale_live_signal_layer_raises(
        self, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        _, image_layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
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
        assert kwargs["filter_kwargs"] == {
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
        assert panel._build_clean_kwargs()["filter_kwargs"]["padtype"] is None

    def test_nonzero_padlen_is_forwarded(self, panel):
        panel._padlen_spin.setValue(64)
        assert panel._build_clean_kwargs()["filter_kwargs"]["padlen"] == 64


class TestBuildMaskSpec:
    def test_no_signal_selected_returns_none(self, panel, viewer, sample_voxeldata_3dt):
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        assert panel._build_mask_spec(layer) is None

    def test_already_binary_mode_ignores_threshold_spin(
        self, panel, viewer, signals_store, sample_voxeldata_3dt
    ):
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        signals_store.pin_signal(
            origin="test-censor",
            name="censor",
            x=np.arange(10.0),
            y=np.array([0, 1] * 5, dtype=float),
            color="#000000",
            source_label="test",
        )
        panel._refresh_signal_combos()
        panel._mask_combo.setCurrentText("censor")
        panel._mask_threshold_spin.setValue(999.0)  # must be ignored in this mode

        spec = panel._build_mask_spec(layer)
        assert spec is not None
        assert spec.threshold is None

    def test_threshold_mode_reads_spin_and_direction(
        self, panel, viewer, signals_store, sample_voxeldata_3dt
    ):
        _, layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        signals_store.pin_signal(
            origin="test-fd",
            name="fd",
            x=np.arange(10.0),
            y=np.linspace(0, 1, 10),
            color="#000000",
            source_label="test",
        )
        panel._refresh_signal_combos()
        panel._mask_combo.setCurrentText("fd")
        panel._mask_mode_combo.setCurrentText("Keep where value <")
        panel._mask_threshold_spin.setValue(0.3)

        spec = panel._build_mask_spec(layer)
        assert spec is not None
        assert spec.threshold == pytest.approx(0.3)
        assert spec.keep_above is False


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

    def test_match_reference_spec(self, viewer, panel, sample_voxeldata_3dt):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel._refresh_layer_combos()
        panel._resample_mode_combo.setCurrentText("Match reference layer")
        panel._resample_reference_combo.setCurrentText("power_doppler")
        mode, kwargs = panel._build_resample_spec()
        assert mode == "Match reference layer"
        np.testing.assert_allclose(
            kwargs["new_time"], sample_voxeldata_3dt["time"].values
        )
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
        self, viewer, panel, monkeypatch, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
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
        self, qtbot, viewer, panel, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._standardize_combo.setCurrentText("Z-score")

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        expected = clean(
            sample_voxeldata_3dt,
            detrend_order=None,
            standardize_method="zscore",
            low_cutoff=None,
            high_cutoff=None,
            filter_method="butterworth",
            filter_kwargs={
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
        np.testing.assert_allclose(
            new_layer.data, expected.values, rtol=1e-5, atol=1e-8
        )
        # Z-scored output is centered on zero: expect a diverging colormap and a
        # symmetric contrast range rather than the default sequential one.
        assert new_layer.colormap.name == "twilight"
        peak = float(np.nanmax(np.abs(expected.values)))
        assert new_layer.contrast_limits == pytest.approx([-peak, peak], rel=1e-5)

    def test_full_pipeline_order_matches_chained_direct_calls(
        self, qtbot, viewer, panel, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
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
            sample_voxeldata_3dt, start=None, stop=None, step=0.25, method="linear"
        )
        smoothed = smooth_volume(resampled, fwhm=0.4, ensure_finite=False)
        expected = clean(
            smoothed,
            detrend_order=1,
            standardize_method=None,
            low_cutoff=None,
            high_cutoff=None,
            filter_method="butterworth",
            filter_kwargs={
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
        np.testing.assert_allclose(
            new_layer.data, expected.values, rtol=1e-5, atol=1e-8
        )

    def test_imported_confound_matches_direct_clean_call(
        self, qtbot, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        # Aligned 1:1 with sample_voxeldata_3dt's time coordinate (10.0 + arange(10) * 0.5).
        confound_signal = _imported_signal(
            "motion",
            10.0 + np.arange(10) * 0.5,
            np.sin(np.arange(10) * 0.3),
        )
        signals_store._stored_signals.append(confound_signal)

        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel._refresh_layer_combos()
        panel._refresh_signal_combos()
        panel._source_combo.setCurrentText("power_doppler")
        _check_confound(panel, "motion")

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        # The panel always stacks selected confounds along a "confound" dim (even
        # for a single one), matching multi-confound selection.
        expected_confounds = xr.concat(
            [
                xr.DataArray(
                    confound_signal.y,
                    dims="time",
                    coords={"time": sample_voxeldata_3dt["time"]},
                )
            ],
            dim="confound",
        )
        expected = clean(
            sample_voxeldata_3dt,
            detrend_order=None,
            standardize_method=None,
            low_cutoff=None,
            high_cutoff=None,
            filter_method="butterworth",
            filter_kwargs={
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
        np.testing.assert_allclose(
            new_layer.data, expected.values, rtol=1e-5, atol=1e-8
        )

    def test_point_live_signal_confound_matches_direct_clean_call(
        self, qtbot, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        _, image_layer = plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
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
        _check_confound(panel, "Point 0")

        panel._apply()
        qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=5000)

        new_layer = viewer.layers["power_doppler — cleaned"]
        expected_confounds = xr.concat(
            [
                xr.DataArray(
                    sample_voxeldata_3dt.values[:, 1, 2, 3],
                    dims="time",
                    coords={"time": sample_voxeldata_3dt["time"]},
                )
            ],
            dim="confound",
        )
        expected = clean(
            sample_voxeldata_3dt,
            detrend_order=None,
            standardize_method=None,
            low_cutoff=None,
            high_cutoff=None,
            filter_method="butterworth",
            filter_kwargs={
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
        np.testing.assert_allclose(
            new_layer.data, expected.values, rtol=1e-5, atol=1e-8
        )


# ---------------------------------------------------------------------------
# _build_noise_mask
# ---------------------------------------------------------------------------


class TestBuildNoiseMask:
    def test_full_shape_labels_collapsed_over_time(self, sample_voxeldata_3dt):
        from confusius._napari._preprocessing._panel import _build_noise_mask

        labels = np.zeros(sample_voxeldata_3dt.shape, dtype=np.int32)
        labels[:, 0, 0, 0] = 1  # One voxel marked across every timepoint.
        mask = _build_noise_mask(sample_voxeldata_3dt, labels)
        assert mask is not None
        assert mask.shape == sample_voxeldata_3dt.shape[1:]
        assert mask.values[0, 0, 0]
        assert not mask.values[0, 0, 1]

    def test_spatial_shape_labels_used_directly(self, sample_voxeldata_3dt):
        from confusius._napari._preprocessing._panel import _build_noise_mask

        labels = np.zeros(sample_voxeldata_3dt.shape[1:], dtype=np.int32)
        labels[1, 2, 3] = 5
        mask = _build_noise_mask(sample_voxeldata_3dt, labels)
        assert mask is not None
        assert mask.values[1, 2, 3]
        assert mask.values.sum() == 1

    def test_mismatched_shape_returns_none(self, sample_voxeldata_3dt):
        from confusius._napari._preprocessing._panel import _build_noise_mask

        labels = np.zeros((2, 2, 2), dtype=np.int32)
        assert _build_noise_mask(sample_voxeldata_3dt, labels) is None


# ---------------------------------------------------------------------------
# CompCor
# ---------------------------------------------------------------------------


class TestCompcorSubprocess:
    def test_matches_direct_call_on_plain_arrays(self, rng):
        from confusius._napari._preprocessing._panel import _compcor_subprocess
        from confusius.signal import compute_compcor_confounds

        values = rng.random((20, 3, 4, 5))
        time_values = np.arange(20, dtype=float) * 0.5
        noise_mask_values = np.zeros((3, 4, 5), dtype=bool)
        noise_mask_values[:2] = True

        result_time, result_values = _compcor_subprocess(
            values, ("time", "k", "j", "i"), time_values, noise_mask_values, None, 2
        )

        signals = xr.DataArray(
            values, dims=("time", "k", "j", "i"), coords={"time": time_values}
        )
        noise_mask = xr.DataArray(noise_mask_values, dims=("k", "j", "i"))
        expected = compute_compcor_confounds(
            signals, noise_mask=noise_mask, n_components=2
        )

        np.testing.assert_allclose(result_time, expected.coords["time"].values)
        np.testing.assert_allclose(result_values, expected.values, rtol=1e-5)

    def test_picklable_for_process_pool_dispatch(self, rng):
        # ProcessPoolExecutor needs to pickle both the callable and its
        # arguments/return value; a closure or an object carrying a Qt/napari
        # reference would fail here even though it works when called directly.
        from concurrent.futures import ProcessPoolExecutor

        from confusius._napari._preprocessing._panel import _compcor_subprocess

        values = rng.random((20, 3, 4, 5))
        time_values = np.arange(20, dtype=float) * 0.5

        with ProcessPoolExecutor(max_workers=1) as pool:
            result_time, result_values = pool.submit(
                _compcor_subprocess,
                values,
                ("time", "k", "j", "i"),
                time_values,
                None,
                0.5,
                2,
            ).result()

        assert result_values.shape == (20, 2)
        np.testing.assert_allclose(result_time, time_values)


class TestComputeCompcor:
    def test_matches_direct_compute_compcor_confounds_call(
        self, qtbot, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        from confusius.signal import compute_compcor_confounds

        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        labels = np.zeros(sample_voxeldata_3dt.shape[1:], dtype=np.int32)
        labels[:2] = 1
        viewer.add_labels(labels, name="wm")

        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._compcor_mask_combo.setCurrentText("wm")
        panel._compcor_components_spin.setValue(2)

        panel._compute_compcor()
        qtbot.waitUntil(lambda: len(signals_store.stored_signals()) == 2, timeout=5000)

        noise_mask = xr.zeros_like(
            sample_voxeldata_3dt.isel(time=0, drop=True), dtype=bool
        )
        noise_mask.values[:2] = True
        expected = compute_compcor_confounds(
            sample_voxeldata_3dt, noise_mask=noise_mask, n_components=2
        )

        stored = {s.name: s for s in signals_store.stored_signals()}
        assert set(stored) == {
            "CompCor 0 (power_doppler)",
            "CompCor 1 (power_doppler)",
        }
        np.testing.assert_allclose(
            stored["CompCor 0 (power_doppler)"].y,
            expected.isel(component=0).values,
            rtol=1e-5,
        )

    def test_shows_its_own_busy_indicator_not_the_shared_one(
        self, qtbot, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        labels = np.zeros(sample_voxeldata_3dt.shape[1:], dtype=np.int32)
        labels[:2] = 1
        viewer.add_labels(labels, name="wm")

        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._compcor_mask_combo.setCurrentText("wm")

        panel._compute_compcor()
        assert panel._compcor_progress.isVisibleTo(panel)
        assert not panel._progress.isVisibleTo(panel)

        qtbot.waitUntil(lambda: len(signals_store.stored_signals()) == 5, timeout=5000)
        assert not panel._compcor_progress.isVisibleTo(panel)

    def test_recomputing_updates_signals_in_place(
        self, qtbot, viewer, panel, signals_store, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        labels = np.zeros(sample_voxeldata_3dt.shape[1:], dtype=np.int32)
        labels[:2] = 1
        viewer.add_labels(labels, name="wm")

        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")
        panel._compcor_mask_combo.setCurrentText("wm")
        panel._compcor_components_spin.setValue(1)

        panel._compute_compcor()
        qtbot.waitUntil(lambda: len(signals_store.stored_signals()) == 1, timeout=5000)
        panel._compute_compcor()
        qtbot.wait(500)  # Give a second run a chance to (wrongly) duplicate.

        assert len(signals_store.stored_signals()) == 1

    def test_no_mask_or_variance_threshold_shows_error(
        self, viewer, panel, monkeypatch, sample_voxeldata_3dt
    ):
        plot_napari(
            sample_voxeldata_3dt,
            viewer=viewer,
            show_colorbar=False,
            show_scale_bar=False,
        )
        panel._refresh_layer_combos()
        panel._source_combo.setCurrentText("power_doppler")

        errors = []
        monkeypatch.setattr(
            "confusius._napari._preprocessing._panel.show_error", errors.append
        )
        panel._compute_compcor()

        assert len(errors) == 1
        assert "noise mask" in errors[0]
