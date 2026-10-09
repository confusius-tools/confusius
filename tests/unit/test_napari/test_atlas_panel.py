"""Unit tests for the AtlasPanel widget.

The `_on_*_returned` slots are exercised directly with the in-memory `atlas_ds`
fixture so tests never touch the network or the background `thread_worker` machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Image, Labels
from numpy.testing import assert_array_equal

from confusius._napari._atlas._panel import TEMPLATES, AtlasPanel


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def panel(viewer):
    return AtlasPanel(viewer)


@pytest.fixture
def loaded_panel(panel, atlas_ds):
    """Panel with `atlas_ds` loaded under the name `mock`."""
    panel._atlas_name = "mock"
    panel._on_atlas_returned(atlas_ds)
    return panel


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
    def test_adds_reference_annotation_and_hemispheres_layers(
        self, loaded_panel, viewer, atlas_ds
    ) -> None:
        names = [layer.name for layer in viewer.layers]
        assert names == ["mock reference", "mock annotation", "mock hemispheres"]
        assert isinstance(viewer.layers["mock reference"], Image)
        assert isinstance(viewer.layers["mock annotation"], Labels)
        assert isinstance(viewer.layers["mock hemispheres"], Labels)
        assert_array_equal(
            viewer.layers["mock annotation"].data, atlas_ds["annotation"].values
        )

    def test_enables_tree_and_mask_buttons(self, panel, loaded_panel) -> None:
        assert loaded_panel._tree_btn.isEnabled()
        assert loaded_panel._mask_btn.isEnabled()
        assert loaded_panel._atlas_progress.isHidden()

    def test_buttons_disabled_before_load(self, panel) -> None:
        assert not panel._tree_btn.isEnabled()
        assert not panel._mask_btn.isEnabled()

    def test_error_restores_idle_state(self, panel, monkeypatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.show_error", messages.append
        )
        panel._begin_atlas_work()
        panel._on_atlas_error(RuntimeError("boom"))

        assert messages == ["boom"]
        assert panel._atlas_progress.isHidden()
        assert panel._load_atlas_btn.text() == "Load atlas"
        assert not panel._tree_btn.isEnabled()


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


class TestRegionSearch:
    def test_empty_pattern_lists_every_structure(self, loaded_panel) -> None:
        ids = [row[0] for row in _rows(loaded_panel._results_table)]
        assert sorted(ids) == ["10", "20", "997"]

    def test_all_field_matches_name_substring(self, loaded_panel) -> None:
        loaded_panel._search_edit.setText("child")
        ids = [row[0] for row in _rows(loaded_panel._results_table)]
        assert sorted(ids) == ["10", "20"]

    def test_acronym_field_requires_full_match(self, loaded_panel) -> None:
        loaded_panel._field_combo.setCurrentText("acronym")
        loaded_panel._search_edit.setText("ch")
        assert _rows(loaded_panel._results_table) == [("10", "ch", "child region")]

    def test_invalid_regex_shows_no_rows(self, loaded_panel) -> None:
        loaded_panel._search_edit.setText("ch(")
        assert loaded_panel._results_table.rowCount() == 0

    def test_no_atlas_leaves_table_empty(self, panel) -> None:
        panel._search_edit.setText("ch")
        assert panel._results_table.rowCount() == 0


class TestRegionSelection:
    def test_add_moves_rows_and_dedups(self, loaded_panel) -> None:
        loaded_panel._field_combo.setCurrentText("acronym")
        loaded_panel._search_edit.setText("ch")
        results = loaded_panel._results_table
        assert _rows(results) == [("10", "ch", "child region")]

        _select_rows(results, [0])
        loaded_panel._add_regions()
        loaded_panel._add_regions()

        assert _rows(loaded_panel._selected_table) == [("10", "ch", "child region")]
        assert loaded_panel._load_masks_btn.isEnabled()

    def test_remove_drops_rows(self, loaded_panel) -> None:
        results = loaded_panel._results_table
        _select_rows(results, list(range(results.rowCount())))
        loaded_panel._add_regions()
        assert loaded_panel._selected_table.rowCount() == 3

        _select_rows(loaded_panel._selected_table, [0, 2])
        loaded_panel._remove_regions()

        assert loaded_panel._selected_table.rowCount() == 1
        assert loaded_panel._load_masks_btn.isEnabled()

        _select_rows(loaded_panel._selected_table, [0])
        loaded_panel._remove_regions()
        assert not loaded_panel._load_masks_btn.isEnabled()

    def test_reloading_atlas_clears_selection(self, loaded_panel, atlas_ds) -> None:
        _select_rows(loaded_panel._results_table, [0])
        loaded_panel._add_regions()
        loaded_panel._on_atlas_returned(atlas_ds)
        assert loaded_panel._selected_table.rowCount() == 0


class TestMaskLoading:
    def test_left_side_mask_layer(self, loaded_panel, viewer, atlas_ds) -> None:
        loaded_panel._side_combo.setCurrentText("left")
        masks = atlas_ds.atlas.get_masks([10], sides="left")
        loaded_panel._on_masks_returned(masks)

        layer = viewer.layers["ch_L"]
        assert isinstance(layer, Labels)
        annotation = atlas_ds["annotation"].values
        hemispheres = atlas_ds["hemispheres"].values
        expected = np.where(np.isin(annotation, [10, 20]) & (hemispheres == 1), 10, 0)
        assert_array_equal(layer.data, expected)
        assert layer.scale.tolist() == pytest.approx([0.05, 0.05, 0.05])

    def test_one_layer_per_region(self, loaded_panel, viewer, atlas_ds) -> None:
        n_before = len(viewer.layers)
        masks = atlas_ds.atlas.get_masks([10, 20], sides="both")
        loaded_panel._on_masks_returned(masks)

        new_layers = list(viewer.layers)[n_before:]
        assert [layer.name for layer in new_layers] == ["ch", "gc"]
        assert_array_equal(np.unique(viewer.layers["gc"].data), [0, 20])
        assert loaded_panel._masks_progress.isHidden()

    def test_load_masks_without_selection_reports_error(
        self, loaded_panel, monkeypatch
    ) -> None:
        messages: list[str] = []
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.show_error", messages.append
        )
        loaded_panel._load_masks()
        assert messages == ["Add at least one region."]


class TestTemplateLoading:
    def test_dropdown_lists_every_template(self, panel) -> None:
        combo = panel._template_combo
        assert [combo.itemText(i) for i in range(combo.count())] == list(TEMPLATES)

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

    def test_error_restores_idle_state(self, panel, monkeypatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.show_error", messages.append
        )
        panel._begin_template_work()
        panel._on_template_error(RuntimeError("offline"))

        assert messages == ["offline"]
        assert panel._template_combo.isEnabled()
        assert panel._load_template_btn.text() == "Load template"


class TestStructureTreeDock:
    def test_opens_right_dock_once(self, loaded_panel, viewer) -> None:
        loaded_panel._show_tree()
        dock = loaded_panel._tree_dock
        assert dock is not None
        assert dock.windowTitle().startswith("Atlas structures")
        assert loaded_panel._tree.topLevelItem(0).text(0) == "root"

        loaded_panel._show_tree()
        assert loaded_panel._tree_dock is dock

    def test_ignored_without_atlas(self, panel) -> None:
        panel._show_tree()
        assert panel._tree is None
