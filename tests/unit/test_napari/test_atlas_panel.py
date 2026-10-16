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

from confusius._napari._atlas._panel import (
    ATLAS_VOLUME_LAYER_TYPES,
    AtlasPanel,
    _compute_masks,
    _fetch_template,
)
from confusius._utils.templates import FUSI_TEMPLATES
from confusius.validation import validate_voxeldata


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
    panel._on_atlas_returned(ds, name)


def _list(panel, downloaded=("mock",), others=("other",), latest=None) -> None:
    """Feed the panel a listing where every atlas is at version 1.0 unless `latest`
    names a newer remote version for some of them."""
    local = {name: "1.0" for name in downloaded}
    remote = {name: "1.0" for name in [*downloaded, *others]}
    remote.update(latest or {})
    panel._on_atlases_listed((local, remote))


@pytest.fixture
def loaded_panel(panel, atlas_ds):
    """Panel with `atlas_ds` loaded under the name `mock`."""
    _load(panel, atlas_ds, "mock")
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


def _atlas_controls_enabled(panel) -> dict[str, bool]:
    return {
        "combo": panel._atlas_combo.isEnabled(),
        "load": panel._load_atlas_btn.isEnabled(),
        "remove": panel._remove_atlas_btn.isEnabled(),
        "volumes": all(btn.isEnabled() for btn in panel._volume_btns.values()),
        "tree": panel._tree_btn.isEnabled(),
        "masks": panel._masks_group.isEnabled(),
    }


class TestAtlasControls:
    def test_idle_after_listing_enables_only_dropdown_and_load(self, panel) -> None:
        _list(panel)
        assert _atlas_controls_enabled(panel) == {
            "combo": True,
            "load": True,
            "remove": False,
            "volumes": False,
            "tree": False,
            "masks": False,
        }

    def test_load_disabled_while_list_is_fetching(self, panel) -> None:
        assert panel._atlas_combo.currentData() is None
        assert not panel._load_atlas_btn.isEnabled()

    def test_loading_locks_dropdown_and_load(self, panel) -> None:
        _list(panel)
        panel._begin_atlas_work()
        assert panel._load_atlas_btn.text() == "Loading…"
        assert not panel._atlas_combo.isEnabled()
        assert not panel._load_atlas_btn.isEnabled()
        assert not panel._remove_atlas_btn.isEnabled()
        assert not panel._atlas_notice.isVisibleTo(panel)

    def test_download_notice_names_the_atlas_until_it_is_loaded(
        self, panel, atlas_ds
    ) -> None:
        panel._begin_atlas_work()
        panel._on_atlas_downloading("allen_mouse_10um")

        assert panel._load_atlas_btn.text() == "Downloading…"
        assert panel._atlas_notice.isVisibleTo(panel)
        assert panel._atlas_notice.text().startswith("Downloading allen_mouse_10um")

        _load(panel, atlas_ds, "allen_mouse_10um")

        assert not panel._atlas_notice.isVisibleTo(panel)
        assert panel._atlas_downloading is False

    def test_download_notice_hidden_after_error(self, panel, errors) -> None:
        panel._begin_atlas_work()
        panel._on_atlas_downloading("allen_mouse_10um")
        panel._on_atlas_error(RuntimeError("offline"))

        assert not panel._atlas_notice.isVisibleTo(panel)
        assert panel._load_atlas_btn.text() == "Load atlas"

    def test_search_placeholder_does_not_mention_regex(self, panel) -> None:
        assert panel._search_edit.placeholderText() == "Acronym or name"

    def test_loaded_unlocks_atlas_buttons_and_locks_load(self, loaded_panel) -> None:
        _list(loaded_panel)
        assert _atlas_controls_enabled(loaded_panel) == {
            "combo": False,
            "load": False,
            "remove": True,
            "volumes": True,
            "tree": True,
            "masks": True,
        }
        assert (
            loaded_panel._load_atlas_btn.text() == "Remove atlas to load another one →"
        )

    def test_volume_buttons_follow_volume_order(self, panel) -> None:
        assert list(panel._volume_btns) == list(ATLAS_VOLUME_LAYER_TYPES)
        assert [btn.text() for btn in panel._volume_btns.values()] == [
            "Template",
            "Annotation",
            "Hemispheres",
        ]


class TestAtlasLoading:
    def test_loads_without_adding_layers(self, loaded_panel, viewer, atlas_ds) -> None:
        assert loaded_panel._atlas is atlas_ds
        assert loaded_panel._atlas_name == "mock"
        assert len(viewer.layers) == 0
        assert loaded_panel._atlas_progress.isHidden()

    def test_each_volume_button_adds_its_layer(
        self, loaded_panel, viewer, atlas_ds
    ) -> None:
        for btn in loaded_panel._volume_btns.values():
            btn.click()

        names = [layer.name for layer in viewer.layers]
        assert names == ["mock template", "mock annotation", "mock hemispheres"]
        assert isinstance(viewer.layers["mock template"], Image)
        assert isinstance(viewer.layers["mock annotation"], Labels)
        assert isinstance(viewer.layers["mock hemispheres"], Labels)
        assert_array_equal(
            viewer.layers["mock annotation"].data, atlas_ds["annotation"].values
        )
        assert loaded_panel._layers_progress.isHidden()

    def test_second_click_adds_the_layer_again(self, loaded_panel, viewer) -> None:
        loaded_panel._volume_btns["template"].click()
        loaded_panel._volume_btns["template"].click()
        assert [layer.name for layer in viewer.layers] == [
            "mock template",
            "mock template [1]",
        ]

    def test_dropdown_points_at_loaded_atlas_once_listed(self, loaded_panel) -> None:
        _list(loaded_panel, downloaded=("other",), others=("mock",))
        assert loaded_panel._atlas_combo.currentData() == "mock"
        assert not loaded_panel._atlas_combo.isEnabled()

    def test_load_before_listing_leaves_the_placeholder(self, loaded_panel) -> None:
        combo = loaded_panel._atlas_combo
        assert combo.count() == 1
        assert combo.currentData() is None

    def test_fresh_download_moves_atlas_into_downloaded_group(
        self, panel, atlas_ds
    ) -> None:
        _list(panel, downloaded=("mock",), others=("other", "zzz"))
        combo = panel._atlas_combo
        assert combo.itemText(2) == "other"

        _load(panel, atlas_ds, "other")

        assert [combo.itemText(i) for i in range(combo.count())] == [
            "mock (downloaded)",
            "other (downloaded)",
            "",  # Separator.
            "zzz",
        ]
        assert combo.currentData() == "other"

    def test_reloading_an_outdated_atlas_keeps_its_update_flag(
        self, panel, atlas_ds
    ) -> None:
        _list(panel, downloaded=("mock",), others=(), latest={"mock": "2.0"})
        _load(panel, atlas_ds, "mock")
        assert panel._atlas_combo.itemText(0) == "mock (downloaded, update available)"

    def test_error_restores_idle_state(self, panel, errors) -> None:
        _list(panel)
        panel._pending_layers = True
        panel._begin_atlas_work()
        panel._on_atlas_error(RuntimeError("boom"))

        assert errors == ["boom"]
        assert panel._atlas_progress.isHidden()
        assert panel._load_atlas_btn.text() == "Load atlas"
        assert panel._load_atlas_btn.isEnabled()
        assert panel._pending_layers is False

    def test_other_atlas_replaces_loaded_one(
        self, loaded_panel, viewer, atlas_ds
    ) -> None:
        loaded_panel._toggle_tree()
        _select_rows(loaded_panel._results_table, [0])
        loaded_panel._add_regions()

        _load(loaded_panel, atlas_ds, "other")

        assert loaded_panel._atlas_name == "other"
        assert loaded_panel._tree is None
        assert "Atlas structures: mock" not in viewer.window.dock_widgets
        assert loaded_panel._selected_table.rowCount() == 0
        assert loaded_panel._results_table.rowCount() == 3

    def test_reloading_same_atlas_keeps_tree_and_clears_selection(
        self, loaded_panel, atlas_ds
    ) -> None:
        loaded_panel._toggle_tree()
        dock = loaded_panel._tree_dock
        _select_rows(loaded_panel._results_table, [0])
        loaded_panel._add_regions()

        _load(loaded_panel, atlas_ds, "mock")

        assert loaded_panel._tree_dock is dock
        assert loaded_panel._selected_table.rowCount() == 0
        assert not loaded_panel._load_masks_btn.isEnabled()


class TestAtlasRemoval:
    def test_remove_resets_panel(self, loaded_panel, viewer) -> None:
        _list(loaded_panel)
        loaded_panel._toggle_tree()
        dock = loaded_panel._tree_dock
        _select_rows(loaded_panel._results_table, [0])
        loaded_panel._add_regions()

        loaded_panel._remove_atlas_btn.click()

        assert loaded_panel._atlas is None
        assert loaded_panel._atlas_name is None
        assert loaded_panel._tree is None
        assert loaded_panel._tree_dock is None
        assert dock.widget() is None
        assert "Atlas structures: mock" not in viewer.window.dock_widgets
        assert loaded_panel._results_table.rowCount() == 0
        assert loaded_panel._selected_table.rowCount() == 0
        assert _atlas_controls_enabled(loaded_panel) == {
            "combo": True,
            "load": True,
            "remove": False,
            "volumes": False,
            "tree": False,
            "masks": False,
        }

    def test_remove_keeps_layers_in_viewer(self, loaded_panel, viewer) -> None:
        loaded_panel._volume_btns["annotation"].click()
        loaded_panel._remove_atlas()
        assert [layer.name for layer in viewer.layers] == ["mock annotation"]


class TestAtlasList:
    def test_downloaded_first_then_separator_then_others(self, panel) -> None:
        panel._on_atlases_listed(
            ({"b_atlas": "1.0"}, {"c_atlas": "1.0", "a_atlas": "1.0", "b_atlas": "1.0"})
        )

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

    def test_newer_remote_version_flags_downloaded_atlas(self, panel) -> None:
        panel._on_atlases_listed(
            ({"a_atlas": "1.0", "b_atlas": "1.0"}, {"a_atlas": "1.1", "b_atlas": "1.0"})
        )

        combo = panel._atlas_combo
        assert combo.itemText(0) == "a_atlas (downloaded, update available)"
        assert combo.itemText(1) == "b_atlas (downloaded)"
        assert combo.count() == 2

    def test_unknown_local_version_is_not_flagged(self, panel) -> None:
        panel._on_atlases_listed(({"a_atlas": None}, {"a_atlas": "1.1"}))
        assert panel._atlas_combo.itemText(0) == "a_atlas (downloaded)"

    def test_unavailable_list_keeps_downloaded_and_shows_status(self, panel) -> None:
        panel._on_atlases_listed(({"b_atlas": "1.0"}, None))

        assert panel._atlas_combo.itemText(0) == "b_atlas (downloaded)"
        assert panel._atlas_combo.itemData(0) == "b_atlas"
        assert panel._atlas_combo.count() == 1
        assert panel._atlas_status.isVisibleTo(panel)

    def test_relisting_keeps_the_selected_atlas(self, panel) -> None:
        _list(panel, downloaded=("mock",), others=("other",))
        combo = panel._atlas_combo
        combo.setCurrentIndex(combo.findData("other"))

        _list(panel, downloaded=("mock",), others=("other", "zzz"))

        assert combo.currentData() == "other"

    def test_nothing_available_disables_dropdown_and_load(self, panel) -> None:
        panel._on_atlases_listed(({}, None))

        assert panel._atlas_combo.count() == 0
        assert not panel._atlas_combo.isEnabled()
        assert not panel._load_atlas_btn.isEnabled()

    def test_listing_error_shows_status(self, panel) -> None:
        panel._on_atlases_error(RuntimeError("offline"))
        assert (
            panel._atlas_status.text() == "Could not list BrainGlobe atlases: offline"
        )
        assert not panel._load_atlas_btn.isEnabled()


class TestMasksGroup:
    def test_always_visible_but_disabled_without_atlas(self, panel) -> None:
        assert panel._masks_group.isVisibleTo(panel)
        assert not panel._masks_group.isEnabled()
        assert panel._masks_group.title() == "Region masks"
        assert panel._results_table.rowCount() == 0

    def test_filled_on_load_and_emptied_on_remove(self, loaded_panel) -> None:
        assert loaded_panel._masks_group.isEnabled()
        assert loaded_panel._results_table.rowCount() == 3

        loaded_panel._remove_atlas()

        assert not loaded_panel._masks_group.isEnabled()
        assert loaded_panel._results_table.rowCount() == 0


class TestRegionSearch:
    def test_empty_pattern_lists_every_structure(self, loaded_panel) -> None:
        ids = [row[0] for row in _rows(loaded_panel._results_table)]
        assert sorted(ids) == ["10", "20", "997"]

    def test_pattern_matches_acronym_or_name_substring(self, loaded_panel) -> None:
        loaded_panel._search_edit.setText("child")
        ids = [row[0] for row in _rows(loaded_panel._results_table)]
        assert sorted(ids) == ["10", "20"]

    def test_anchored_regex_matches_acronym_only(self, loaded_panel) -> None:
        loaded_panel._search_edit.setText("^ch$")
        assert _rows(loaded_panel._results_table) == [("10", "ch", "child region")]

    def test_selected_table_is_shorter_than_results(self, panel) -> None:
        assert (
            panel._selected_table.minimumHeight() < panel._results_table.minimumHeight()
        )

    def test_invalid_regex_shows_no_rows(self, loaded_panel) -> None:
        loaded_panel._search_edit.setText("ch(")
        assert loaded_panel._results_table.rowCount() == 0

    def test_no_atlas_leaves_table_empty(self, panel) -> None:
        panel._search_edit.setText("ch")
        assert panel._results_table.rowCount() == 0


class TestRegionSelection:
    def test_add_moves_rows_and_dedups(self, loaded_panel) -> None:
        loaded_panel._search_edit.setText("^ch$")
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

    def test_tables_share_geometry(self, loaded_panel) -> None:
        results, selected = loaded_panel._results_table, loaded_panel._selected_table
        _select_rows(results, [0])
        loaded_panel._add_regions()
        for column in range(3):
            assert results.columnWidth(column) == selected.columnWidth(column)
        assert results.rowHeight(0) == selected.rowHeight(0)


class TestLazyAtlas:
    """Atlases stay Dask-backed until masks need the full volumes."""

    @pytest.fixture
    def lazy_atlas(self, atlas_ds):
        return atlas_ds.chunk({"k": 2})

    def test_volume_layers_keep_dask_arrays(self, panel, viewer, lazy_atlas) -> None:
        import dask.array as da

        _load(panel, lazy_atlas, "lazy")
        for volume in ATLAS_VOLUME_LAYER_TYPES:
            panel._add_layer(volume)

        for suffix in ("template", "annotation", "hemispheres"):
            assert isinstance(viewer.layers[f"lazy {suffix}"].data, da.Array)

    def test_mask_request_loads_volumes_once(
        self, panel, viewer, lazy_atlas, atlas_ds
    ) -> None:
        _load(panel, lazy_atlas, "lazy")
        masks, in_memory = _compute_masks.__wrapped__(lazy_atlas, [10], "left")
        panel._on_masks_worker_returned(lazy_atlas, (masks, in_memory))

        expected = atlas_ds.atlas.get_masks([10], sides="left")
        assert_array_equal(viewer.layers["ch_L"].data, expected.values[0])
        stored = panel._atlas
        assert isinstance(stored["annotation"].data, np.ndarray)
        assert isinstance(stored["hemispheres"].data, np.ndarray)
        validate_voxeldata(stored["annotation"])

    def test_removed_atlas_is_not_restored(self, panel, lazy_atlas) -> None:
        _load(panel, lazy_atlas, "lazy")
        result = _compute_masks.__wrapped__(lazy_atlas, [10], "both")
        panel._remove_atlas()

        panel._on_masks_worker_returned(lazy_atlas, result)

        assert panel._atlas is None


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
        self, loaded_panel, errors
    ) -> None:
        loaded_panel._load_masks()
        assert errors == ["Add at least one region."]


class TestTemplateLoading:
    def test_dropdown_lists_every_template(self, panel) -> None:
        combo = panel._template_combo
        assert [combo.itemText(i) for i in range(combo.count())] == list(FUSI_TEMPLATES)
        assert "brunner_grillet_2021" in FUSI_TEMPLATES

    def test_reference_atlas_label_follows_combo(self, panel) -> None:
        combo = panel._template_combo
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            expected = FUSI_TEMPLATES[combo.currentText()]["reference_atlas"]
            assert panel._template_reference.text() == f"Reference atlas: {expected}"

    def test_adds_image_layer_named_after_template(
        self, panel, viewer, sample_voxeldata_3d
    ) -> None:
        panel._begin_template_work()
        panel._on_template_returned(sample_voxeldata_3d, "brunner_grillet_2021")

        assert len(viewer.layers) == 1
        layer = viewer.layers["brunner_grillet_2021"]
        assert isinstance(layer, Image)
        assert panel._template_progress.isHidden()
        assert panel._load_template_btn.isEnabled()

    def test_error_restores_idle_state(self, panel, errors) -> None:
        panel._begin_template_work()
        panel._on_template_error(RuntimeError("offline"))

        assert errors == ["offline"]
        assert panel._template_combo.isEnabled()
        assert panel._load_template_btn.text() == "Load template"

    def test_load_lazily_unchecked_by_default(self, panel) -> None:
        assert not panel._template_lazy_check.isChecked()

    @pytest.mark.parametrize("lazy", [False, True])
    def test_fetch_reads_voxels_once_unless_lazy(
        self, monkeypatch, sample_voxeldata_3d, lazy
    ) -> None:
        import dask.array as da

        monkeypatch.setattr(
            "confusius._napari._atlas._panel.fetch_template",
            lambda name, **kwargs: sample_voxeldata_3d.chunk({"k": 2}),
        )

        result = _fetch_template.__wrapped__("brunner_grillet_2021", lazy)

        assert isinstance(result.data, da.Array if lazy else np.ndarray)
        # The world coordinates stay index-derived either way.
        validate_voxeldata(result)
        assert_array_equal(np.asarray(result.data), sample_voxeldata_3d.values)


class TestStructureTreeDock:
    def test_button_toggles_one_right_dock(self, loaded_panel, viewer) -> None:
        loaded_panel._tree_btn.click()
        dock = loaded_panel._tree_dock
        assert dock.windowTitle().startswith("Atlas structures: mock")
        assert loaded_panel._tree.topLevelItem(0).text(0) == "root"

        # Second click hides the dock, third shows it again; same dock throughout.
        loaded_panel._tree_btn.click()
        assert loaded_panel._tree_dock is dock
        assert dock.isHidden()
        loaded_panel._tree_btn.click()
        assert loaded_panel._tree_dock is dock
        assert not dock.isHidden()

    def test_ignored_without_atlas(self, panel) -> None:
        panel._toggle_tree()
        assert panel._tree is None
        assert panel._tree_dock is None


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
        panel._on_atlases_listed(
            (
                {"allen_mouse_bluebrain_barrels_10um": "1.0"},
                {name: "1.0" for name in long_names},
            )
        )
        QApplication.processEvents()
        assert panel.minimumSizeHint().width() <= initial
        assert initial <= self.SIDEBAR_MIN_WIDTH

    def test_loaded_atlas_fits_sidebar(self, panel, atlas_ds) -> None:
        # Showing the panel otherwise starts the BrainGlobe listing worker.
        panel._atlases_listed = True
        panel.show()
        panel._on_atlas_returned(atlas_ds, "allen_mouse_bluebrain_barrels_10um")
        QApplication.processEvents()
        assert panel.minimumSizeHint().width() <= self.SIDEBAR_MIN_WIDTH


class TestPages:
    def test_starts_on_atlases_page(self, panel) -> None:
        assert panel._atlases_page_btn.isChecked()
        assert panel._atlases_page.isVisibleTo(panel)
        assert not panel._templates_page.isVisibleTo(panel)

    def test_templates_button_switches_page(self, panel) -> None:
        panel._templates_page_btn.click()
        assert panel._templates_page.isVisibleTo(panel)
        assert not panel._atlases_page.isVisibleTo(panel)
        assert not panel._atlases_page_btn.isChecked()

        panel._atlases_page_btn.click()
        assert panel._atlases_page.isVisibleTo(panel)
        assert not panel._templates_page.isVisibleTo(panel)

    def test_switching_pages_does_not_widen_panel(self, panel) -> None:
        panel._atlases_listed = True
        panel.show()
        QApplication.processEvents()
        on_atlases = panel.minimumSizeHint().width()
        panel._templates_page_btn.click()
        QApplication.processEvents()
        assert panel.minimumSizeHint().width() <= max(on_atlases, 430)


class TestProgressBarsKeepLayout:
    """Showing a progress bar must not change the panel height (sidebar scrollbar)."""

    def test_layers_masks_and_atlas_bars_reserve_their_space(
        self, loaded_panel
    ) -> None:
        QApplication.processEvents()
        idle_height = loaded_panel.sizeHint().height()

        loaded_panel._layers_progress.show()
        loaded_panel._begin_masks_work()
        loaded_panel._begin_atlas_work()
        QApplication.processEvents()

        assert loaded_panel.sizeHint().height() == idle_height


class TestProgrammaticLoading:
    """`load_atlas` / `load_template` run the button paths without the widgets."""

    @pytest.fixture
    def offline(self, monkeypatch, atlas_ds, sample_voxeldata_3d):
        """Serve every fetch from fixtures so the workers never touch the network."""
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.is_brainglobe_atlas_cached",
            lambda name: True,
        )
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.fetch_brainglobe_atlas",
            lambda name, **kwargs: atlas_ds.assign_attrs(name=name),
        )
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.fetch_template",
            lambda name, **kwargs: sample_voxeldata_3d.chunk({"k": 2}),
        )

    def test_uncached_atlas_shows_download_notice_while_fetching(
        self, qtbot, panel, offline, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "confusius._napari._atlas._panel.is_brainglobe_atlas_cached",
            lambda name: False,
        )
        seen: list[str] = []
        monkeypatch.setattr(
            panel,
            "_on_atlas_downloading",
            lambda name: (
                seen.append(name),
                AtlasPanel._on_atlas_downloading(panel, name),
            ),
        )

        panel.load_atlas("allen_mouse_10um")
        qtbot.waitUntil(lambda: panel._atlas is not None, timeout=10_000)

        assert seen == ["allen_mouse_10um"]
        assert not panel._atlas_notice.isVisibleTo(panel)

    def test_load_atlas_adds_layers(self, qtbot, panel, viewer, offline) -> None:
        panel.load_atlas("allen_mouse_25um", add_layers=True)
        qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=10_000)

        assert panel._atlas_name == "allen_mouse_25um"
        assert [layer.name for layer in viewer.layers] == [
            "allen_mouse_25um template",
            "allen_mouse_25um annotation",
            "allen_mouse_25um hemispheres",
        ]
        assert panel._atlas_progress.isHidden()
        assert panel._pending_layers is False

    def test_second_atlas_replaces_first(self, qtbot, panel, viewer, offline) -> None:
        panel.load_atlas("allen_mouse_25um")
        qtbot.waitUntil(lambda: panel._atlas is not None, timeout=10_000)

        panel.load_atlas("whs_sd_rat_39um")
        assert panel._atlas is None  # Removed before the new fetch starts.
        qtbot.waitUntil(lambda: panel._atlas is not None, timeout=10_000)

        assert panel._atlas_name == "whs_sd_rat_39um"
        assert panel._atlas.attrs["name"] == "whs_sd_rat_39um"
        assert len(viewer.layers) == 0
        qtbot.waitUntil(lambda: panel._atlas_progress.isHidden(), timeout=10_000)

    def test_load_while_loading_is_ignored(self, qtbot, panel, offline) -> None:
        panel.load_atlas("allen_mouse_25um")
        panel.load_atlas("whs_sd_rat_39um")
        qtbot.waitUntil(lambda: panel._atlas is not None, timeout=10_000)

        assert panel._atlas_name == "allen_mouse_25um"
        qtbot.waitUntil(lambda: panel._atlas_progress.isHidden(), timeout=10_000)
        assert panel._atlas_name == "allen_mouse_25um"

    def test_loading_the_loaded_atlas_is_a_noop(self, loaded_panel, offline) -> None:
        loaded_panel.load_atlas("mock")
        assert loaded_panel._atlas_loading is False

    def test_load_template_adds_in_memory_layer(
        self, qtbot, panel, viewer, offline
    ) -> None:
        panel.load_template("pepe_mariani_2026")
        qtbot.waitUntil(lambda: len(viewer.layers) == 1, timeout=10_000)

        assert viewer.layers[0].name == "pepe_mariani_2026"
        assert isinstance(viewer.layers[0].data, np.ndarray)
        qtbot.waitUntil(lambda: panel._template_progress.isHidden(), timeout=10_000)

    def test_load_lazily_box_keeps_dask_layer(
        self, qtbot, panel, viewer, offline
    ) -> None:
        import dask.array as da

        panel._template_lazy_check.setChecked(True)
        panel._load_template_btn.click()
        qtbot.waitUntil(lambda: len(viewer.layers) == 1, timeout=10_000)

        assert isinstance(viewer.layers[0].data, da.Array)
        qtbot.waitUntil(lambda: panel._template_progress.isHidden(), timeout=10_000)

    def test_unknown_template_raises(self, panel) -> None:
        with pytest.raises(KeyError, match="Unknown fUSI template"):
            panel.load_template("nope")
