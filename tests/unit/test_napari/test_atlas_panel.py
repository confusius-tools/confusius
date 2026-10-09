"""Unit tests for the AtlasPanel widget.

The `_on_*_returned` slots are exercised directly with the in-memory `atlas_ds`
fixture so tests never touch the network or the background `thread_worker` machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Image, Labels
from numpy.testing import assert_array_equal
from qtpy.QtWidgets import QApplication

from confusius._napari._atlas._panel import TEMPLATES, AtlasPanel


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def panel(viewer):
    return AtlasPanel(viewer)


@pytest.fixture
def errors(monkeypatch) -> list[str]:
    """Collect `show_error` messages emitted by the panel."""
    messages: list[str] = []
    monkeypatch.setattr("confusius._napari._atlas._panel.show_error", messages.append)
    return messages


def _load(panel, ds, name: str) -> None:
    panel._atlas_name = name
    panel._on_atlas_returned(ds)


@pytest.fixture
def loaded_panel(panel, atlas_ds):
    """Panel with `atlas_ds` loaded under the name `mock`."""
    _load(panel, atlas_ds, "mock")
    return panel


@pytest.fixture
def masks_panel(loaded_panel):
    """Loaded panel with the region-mask group opened for `mock`."""
    loaded_panel._toggle_masks("mock")
    return loaded_panel


def _select_rows(table, rows: list[int]) -> None:
    # `selectRow` replaces the selection on every call; go through the selection
    # model so several rows end up selected, as a ctrl-click would do.
    from qtpy.QtCore import QItemSelectionModel

    table.clearSelection()
    selection = table.selectionModel()
    flags = (
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows
    )
    for row in rows:
        selection.select(table.model().index(row, 0), flags)


def _rows(table) -> list[tuple[str, str, str]]:
    return [
        tuple(table.item(row, col).text() for col in range(3))
        for row in range(table.rowCount())
    ]


class TestAtlasLoading:
    def test_registers_row_without_adding_layers(
        self, loaded_panel, viewer, atlas_ds
    ) -> None:
        assert list(loaded_panel._atlases) == ["mock"]
        assert list(loaded_panel._atlas_rows) == ["mock"]
        assert loaded_panel._atlas_rows["mock"].name == "mock"
        assert len(viewer.layers) == 0
        assert loaded_panel._atlas_progress.isHidden()

    def test_two_atlases_give_two_rows(self, loaded_panel, atlas_ds) -> None:
        _load(loaded_panel, atlas_ds, "other")
        assert list(loaded_panel._atlas_rows) == ["mock", "other"]
        assert loaded_panel._rows_layout.count() == 2

    def test_duplicate_load_is_refused(self, loaded_panel, errors) -> None:
        loaded_panel._atlas_combo.clear()
        loaded_panel._atlas_combo.addItem("mock (downloaded)", "mock")
        loaded_panel._load_atlas()
        assert errors == ["mock is already loaded."]
        assert not loaded_panel._load_atlas_btn.isEnabled()

    def test_load_button_follows_combo_selection(self, loaded_panel) -> None:
        loaded_panel._on_atlases_listed((["mock"], ["mock", "other"]))
        combo = loaded_panel._atlas_combo
        combo.setCurrentIndex(0)
        assert not loaded_panel._load_atlas_btn.isEnabled()
        combo.setCurrentIndex(combo.count() - 1)
        assert combo.currentData() == "other"
        assert loaded_panel._load_atlas_btn.isEnabled()

    def test_layers_button_adds_three_layers(
        self, loaded_panel, viewer, atlas_ds
    ) -> None:
        loaded_panel._add_atlas_layers("mock")

        names = [layer.name for layer in viewer.layers]
        assert names == ["mock reference", "mock annotation", "mock hemispheres"]
        assert isinstance(viewer.layers["mock reference"], Image)
        assert isinstance(viewer.layers["mock annotation"], Labels)
        assert isinstance(viewer.layers["mock hemispheres"], Labels)
        assert_array_equal(
            viewer.layers["mock annotation"].data, atlas_ds["annotation"].values
        )

    def test_error_restores_idle_state(self, panel, errors) -> None:
        panel._begin_atlas_work()
        panel._on_atlas_error(RuntimeError("boom"))

        assert errors == ["boom"]
        assert panel._atlas_progress.isHidden()
        assert panel._load_atlas_btn.text() == "Load atlas"


class TestAtlasRemoval:
    def test_remove_drops_row_data_and_tree_dock(self, loaded_panel, viewer) -> None:
        loaded_panel._toggle_tree("mock")
        dock = loaded_panel._tree_docks["mock"]
        assert "Atlas structures: mock" in viewer.window.dock_widgets

        loaded_panel._remove_atlas("mock")

        assert loaded_panel._atlases == {}
        assert loaded_panel._atlas_rows == {}
        assert loaded_panel._trees == {}
        assert loaded_panel._tree_docks == {}
        assert dock.widget() is None
        assert "Atlas structures: mock" not in viewer.window.dock_widgets
        assert not loaded_panel._loaded_label.isVisibleTo(loaded_panel)

    def test_remove_hides_masks_group_showing_it(self, masks_panel) -> None:
        masks_panel._remove_atlas("mock")
        assert masks_panel._masks_atlas_name is None
        assert not masks_panel._masks_group.isVisibleTo(masks_panel)

    def test_remove_other_atlas_keeps_masks_group(self, masks_panel, atlas_ds) -> None:
        _load(masks_panel, atlas_ds, "other")
        masks_panel._remove_atlas("other")
        assert masks_panel._masks_atlas_name == "mock"
        assert masks_panel._masks_group.isVisibleTo(masks_panel)
        assert list(masks_panel._atlas_rows) == ["mock"]


class TestAtlasList:
    def test_downloaded_first_then_separator_then_others(self, panel) -> None:
        panel._on_atlases_listed((["b_atlas"], ["a_atlas", "b_atlas", "c_atlas"]))

        combo = panel._atlas_combo
        assert combo.itemText(0) == "b_atlas (downloaded)"
        assert combo.itemData(0) == "b_atlas"
        # Index 1 is the separator.
        assert [combo.itemData(i) for i in range(2, combo.count())] == [
            "a_atlas",
            "c_atlas",
        ]
        assert combo.isEnabled()
        assert panel._load_atlas_btn.isEnabled()

    def test_unavailable_list_keeps_downloaded_and_shows_status(self, panel) -> None:
        panel._on_atlases_listed((["b_atlas"], None))

        assert [panel._atlas_combo.itemData(i) for i in range(1)] == ["b_atlas"]
        assert panel._atlas_combo.count() == 1
        assert panel._atlas_status.isVisibleTo(panel)

    def test_nothing_available_disables_load(self, panel) -> None:
        panel._on_atlases_listed(([], None))

        assert panel._atlas_combo.count() == 0
        assert not panel._load_atlas_btn.isEnabled()


class TestMasksGroup:
    def test_masks_button_shows_group_titled_after_atlas(self, masks_panel) -> None:
        assert masks_panel._masks_group.isVisibleTo(masks_panel)
        assert masks_panel._masks_group.title() == "Region masks: mock"
        assert masks_panel._masks_atlas_name == "mock"

    def test_masks_button_toggles_group_for_same_atlas(self, masks_panel) -> None:
        masks_panel._toggle_masks("mock")
        assert not masks_panel._masks_group.isVisibleTo(masks_panel)
        masks_panel._toggle_masks("mock")
        assert masks_panel._masks_group.isVisibleTo(masks_panel)

    def test_switching_atlas_retitles_and_clears_selection(
        self, masks_panel, atlas_ds
    ) -> None:
        _select_rows(masks_panel._results_table, [0])
        masks_panel._add_regions()
        assert masks_panel._selected_table.rowCount() == 1

        _load(masks_panel, atlas_ds, "other")
        masks_panel._toggle_masks("other")

        assert masks_panel._masks_group.title() == "Region masks: other"
        assert masks_panel._selected_table.rowCount() == 0
        assert masks_panel._results_table.rowCount() == 3

    def test_group_hidden_before_masks_button(self, loaded_panel) -> None:
        assert not loaded_panel._masks_group.isVisibleTo(loaded_panel)


class TestRegionSearch:
    def test_empty_pattern_lists_every_structure(self, masks_panel) -> None:
        ids = [row[0] for row in _rows(masks_panel._results_table)]
        assert sorted(ids) == ["10", "20", "997"]

    def test_all_field_matches_name_substring(self, masks_panel) -> None:
        masks_panel._search_edit.setText("child")
        ids = [row[0] for row in _rows(masks_panel._results_table)]
        assert sorted(ids) == ["10", "20"]

    def test_acronym_field_requires_full_match(self, masks_panel) -> None:
        masks_panel._field_combo.setCurrentText("acronym")
        masks_panel._search_edit.setText("ch")
        assert _rows(masks_panel._results_table) == [("10", "ch", "child region")]

    def test_invalid_regex_shows_no_rows(self, masks_panel) -> None:
        masks_panel._search_edit.setText("ch(")
        assert masks_panel._results_table.rowCount() == 0

    def test_no_atlas_leaves_table_empty(self, panel) -> None:
        panel._search_edit.setText("ch")
        assert panel._results_table.rowCount() == 0


class TestRegionSelection:
    def test_add_moves_rows_and_dedups(self, masks_panel) -> None:
        masks_panel._field_combo.setCurrentText("acronym")
        masks_panel._search_edit.setText("ch")
        results = masks_panel._results_table
        assert _rows(results) == [("10", "ch", "child region")]

        _select_rows(results, [0])
        masks_panel._add_regions()
        masks_panel._add_regions()

        assert _rows(masks_panel._selected_table) == [("10", "ch", "child region")]
        assert masks_panel._load_masks_btn.isEnabled()

    def test_remove_drops_rows(self, masks_panel) -> None:
        results = masks_panel._results_table
        _select_rows(results, list(range(results.rowCount())))
        masks_panel._add_regions()
        assert masks_panel._selected_table.rowCount() == 3

        _select_rows(masks_panel._selected_table, [0, 2])
        masks_panel._remove_regions()

        assert masks_panel._selected_table.rowCount() == 1
        assert masks_panel._load_masks_btn.isEnabled()

        _select_rows(masks_panel._selected_table, [0])
        masks_panel._remove_regions()
        assert not masks_panel._load_masks_btn.isEnabled()

    def test_reloading_atlas_clears_selection(self, masks_panel, atlas_ds) -> None:
        _select_rows(masks_panel._results_table, [0])
        masks_panel._add_regions()
        _load(masks_panel, atlas_ds, "mock")
        assert masks_panel._selected_table.rowCount() == 0
        assert list(masks_panel._atlas_rows) == ["mock"]

    def test_tables_share_geometry(self, masks_panel) -> None:
        results, selected = masks_panel._results_table, masks_panel._selected_table
        _select_rows(results, [0])
        masks_panel._add_regions()
        for column in range(3):
            assert results.columnWidth(column) == selected.columnWidth(column)
        assert results.rowHeight(0) == selected.rowHeight(0)


class TestMaskLoading:
    def test_left_side_mask_layer(self, masks_panel, viewer, atlas_ds) -> None:
        masks_panel._side_combo.setCurrentText("left")
        masks = atlas_ds.atlas.get_masks([10], sides="left")
        masks_panel._on_masks_returned(masks)

        layer = viewer.layers["ch_L"]
        assert isinstance(layer, Labels)
        annotation = atlas_ds["annotation"].values
        hemispheres = atlas_ds["hemispheres"].values
        expected = np.where(np.isin(annotation, [10, 20]) & (hemispheres == 1), 10, 0)
        assert_array_equal(layer.data, expected)
        assert layer.scale.tolist() == pytest.approx([0.05, 0.05, 0.05])

    def test_one_layer_per_region(self, masks_panel, viewer, atlas_ds) -> None:
        n_before = len(viewer.layers)
        masks = atlas_ds.atlas.get_masks([10, 20], sides="both")
        masks_panel._on_masks_returned(masks)

        new_layers = list(viewer.layers)[n_before:]
        assert [layer.name for layer in new_layers] == ["ch", "gc"]
        assert_array_equal(np.unique(viewer.layers["gc"].data), [0, 20])
        assert masks_panel._masks_progress.isHidden()

    def test_load_masks_without_selection_reports_error(
        self, masks_panel, errors
    ) -> None:
        masks_panel._load_masks()
        assert errors == ["Add at least one region."]


class TestTemplateLoading:
    def test_dropdown_lists_every_template(self, panel) -> None:
        combo = panel._template_combo
        assert [combo.itemText(i) for i in range(combo.count())] == list(TEMPLATES)

    def test_reference_atlas_label_follows_combo(self, panel) -> None:
        combo = panel._template_combo
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            expected = TEMPLATES[combo.currentText()]["reference_atlas"]
            assert panel._template_reference.text() == f"Reference atlas: {expected}"

    def test_template_group_comes_first(self, panel) -> None:
        layout = panel.layout()
        assert layout.itemAt(0).widget() is panel._template_group
        assert layout.itemAt(1).widget() is panel._atlas_group
        assert layout.itemAt(2).widget() is panel._masks_group

    def test_adds_image_layer_named_after_label(
        self, panel, viewer, sample_voxeldata_3d
    ) -> None:
        panel._template_label = "Huang 2025 mouse vascular"
        panel._begin_template_work()
        panel._on_template_returned(sample_voxeldata_3d)

        assert len(viewer.layers) == 1
        layer = viewer.layers["Huang 2025 mouse vascular"]
        assert isinstance(layer, Image)
        assert panel._template_progress.isHidden()
        assert panel._load_template_btn.isEnabled()

    def test_error_restores_idle_state(self, panel, errors) -> None:
        panel._begin_template_work()
        panel._on_template_error(RuntimeError("offline"))

        assert errors == ["offline"]
        assert panel._template_combo.isEnabled()
        assert panel._load_template_btn.text() == "Load template"


class TestStructureTreeDock:
    def test_opens_right_dock_once_per_atlas(
        self, loaded_panel, viewer, atlas_ds
    ) -> None:
        loaded_panel._toggle_tree("mock")
        dock = loaded_panel._tree_docks["mock"]
        assert dock.windowTitle().startswith("Atlas structures: mock")
        assert loaded_panel._trees["mock"].topLevelItem(0).text(0) == "root"

        # Second click hides the dock, third shows it again; same dock throughout.
        loaded_panel._toggle_tree("mock")
        assert loaded_panel._tree_docks["mock"] is dock
        assert dock.isHidden()
        loaded_panel._toggle_tree("mock")
        assert loaded_panel._tree_docks["mock"] is dock
        assert not dock.isHidden()

        _load(loaded_panel, atlas_ds, "other")
        loaded_panel._toggle_tree("other")
        assert set(loaded_panel._tree_docks) == {"mock", "other"}
        assert loaded_panel._tree_docks["other"] is not dock

    def test_ignored_for_unknown_atlas(self, panel) -> None:
        panel._toggle_tree("nope")
        assert panel._trees == {}


class TestPanelWidth:
    """Regression tests for the issue #183 horizontal-overflow pattern."""

    SIDEBAR_MIN_WIDTH = 430

    def test_long_atlas_names_do_not_widen_panel(self, panel) -> None:
        # Showing the panel otherwise starts the BrainGlobe listing worker.
        panel._atlases_listed = True
        panel.show()
        QApplication.processEvents()
        initial = panel.minimumSizeHint().width()
        long_names = [f"demba_allen_seg_dev_mouse_p{i}_25um" for i in range(10, 60)]
        panel._on_atlases_listed((["allen_mouse_bluebrain_barrels_10um"], long_names))
        QApplication.processEvents()
        assert panel.minimumSizeHint().width() <= initial
        assert initial <= self.SIDEBAR_MIN_WIDTH

    def test_loaded_row_and_masks_group_fit_sidebar(self, panel, atlas_ds) -> None:
        # Showing the panel otherwise starts the BrainGlobe listing worker.
        panel._atlases_listed = True
        panel.show()
        panel._atlas_name = "allen_mouse_bluebrain_barrels_10um"
        panel._on_atlas_returned(atlas_ds)
        panel._toggle_masks("allen_mouse_bluebrain_barrels_10um")
        QApplication.processEvents()
        assert panel.minimumSizeHint().width() <= self.SIDEBAR_MIN_WIDTH
