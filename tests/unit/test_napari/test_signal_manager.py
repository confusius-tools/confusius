"""Unit tests for the imported signals manager dialog."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from confusius._napari._signals._manager import SignalsManagerDialog


def test_manager_roi_controls_and_mouse_color_column(
    make_napari_viewer, sample_voxeldata_3dt, signals_store, qtbot, monkeypatch
):
    from matplotlib.colors import to_hex
    from qtpy.QtWidgets import QPushButton

    from confusius._utils.colors import RED
    from confusius.plotting.napari import plot_napari

    viewer = make_napari_viewer()
    plot_napari(
        sample_voxeldata_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    points = viewer.add_points([[1, 1, 1]], name="ROI A")
    second = viewer.add_points([[2, 2, 2]], name="ROI B")
    controller = signals_store.live_sources(viewer)
    dialog = SignalsManagerDialog(signals_store, live_sources=controller)
    qtbot.addWidget(dialog)
    assert dialog._sources_list.count() == 2
    first_item, second_item = dialog._sources_list.item(0), dialog._sources_list.item(1)
    assert first_item is not None and second_item is not None
    assert {first_item.text(), second_item.text()} == {points.name, second.name}
    assert not any(
        button.text() == "Remove source" for button in dialog.findChildren(QPushButton)
    )
    first_item.setCheckState(Qt.CheckState.Unchecked)
    assert not next(s.enabled for s in controller.sources() if s.layer is points)
    assert next(s.enabled for s in controller.sources() if s.layer is second)
    assert next(s.enabled for s in controller.sources() if s.kind == "mouse")
    assert dialog._sources_label.text() == "ROI layers"
    assert not any(
        button.text() == "Default mouse colour"
        for button in dialog.findChildren(QPushButton)
    )
    mouse = next(s for s in signals_store.live_signals() if s.source_type == "mouse")
    assert mouse.color == RED
    roi_color = to_hex(points.face_color[0])
    monkeypatch.setattr(
        "confusius._napari._signals._manager.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    color_button = dialog._table.cellWidget(0, 2)
    assert isinstance(color_button, QPushButton)
    color_button.click()
    assert (
        next(s.color for s in signals_store.live_signals() if s.source_type == "mouse")
        == "#123456"
    )
    updated_button = dialog._table.cellWidget(0, 2)
    assert updated_button is not None
    assert "#123456" in updated_button.styleSheet()
    assert to_hex(points.face_color[0]) == roi_color
    controller.close()


def test_manager_roi_color_sync_style_and_rename_controls(
    make_napari_viewer, sample_voxeldata_3dt, signals_store, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QComboBox, QPushButton

    from confusius.plotting.napari import plot_napari

    viewer = make_napari_viewer()
    plot_napari(
        sample_voxeldata_3dt, viewer=viewer, show_colorbar=False, show_scale_bar=False
    )
    points = viewer.add_points([[1, 1, 1]], face_color="red")
    controller = signals_store.live_sources(viewer)
    seed = next(s for s in signals_store.live_signals() if s.source_type == "point")
    dialog = SignalsManagerDialog(signals_store, live_sources=controller)
    qtbot.addWidget(dialog)
    row = next(
        i
        for i in range(dialog._table.rowCount())
        if (item := dialog._table.item(i, 0)) is not None
        and item.data(Qt.ItemDataRole.UserRole) == seed.id
    )
    sync = dialog._table.item(row, 4)
    assert sync is not None and sync.checkState() == Qt.CheckState.Checked
    monkeypatch.setattr(
        "confusius._napari._signals._manager.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    color_button = dialog._table.cellWidget(row, 2)
    assert isinstance(color_button, QPushButton)
    color_button.click()
    assert signals_store.get_live_signal(seed.id).color == "#123456"
    sync = dialog._table.item(row, 4)
    assert sync is not None and sync.checkState() == Qt.CheckState.Unchecked
    points.face_color = "yellow"
    assert signals_store.get_live_signal(seed.id).color == "#123456"
    style = dialog._table.cellWidget(row, 5)
    assert isinstance(style, QComboBox)
    style.setCurrentText("Dotted")
    assert signals_store.get_live_signal(seed.id).line_style == ":"
    sync = dialog._table.item(row, 4)
    assert sync is not None
    sync.setCheckState(Qt.CheckState.Checked)
    assert signals_store.get_live_signal(seed.id).color == "#ffff00"
    assert signals_store.get_live_signal(seed.id).line_style == ":"
    name = dialog._table.item(row, 1)
    assert name is not None
    name.setText("My ROI")
    assert signals_store.get_live_signal(seed.id).display_name == "My ROI"
    source = dialog._table.item(row, 3)
    assert source is not None and source.text() == seed.default_name
    assert not any(
        button.text() == "Clear All Stored"
        for button in dialog.findChildren(QPushButton)
    )
    controller.close()


def test_manager_refreshes_rows_from_store(qtbot, signals_store, signals_csv):
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 2
    name_item = dialog._table.item(0, 1)
    file_item = dialog._table.item(1, 3)
    assert name_item is not None
    assert file_item is not None
    assert name_item.text() == "a"
    assert file_item.text() == "series.csv"


def test_manager_applies_store_mutations(
    qtbot, signals_store, signals_csv, monkeypatch
):
    imported = signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    name_item = dialog._table.item(0, 1)
    assert name_item is not None
    name_item.setText("baseline")
    assert signals_store.stored_signals()[0].name == "baseline"

    visible_item = dialog._table.item(0, 0)
    assert visible_item is not None
    visible_item.setCheckState(Qt.CheckState.Unchecked)
    assert signals_store.stored_signals()[0].visible is False

    monkeypatch.setattr(
        "confusius._napari._signals._manager.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    dialog._choose_color(imported[0].id)
    assert signals_store.stored_signals()[0].color == "#123456"

    dialog._table.selectRow(0)
    dialog._remove_selected()
    # Should have removed only the first signal, leaving the second.
    assert len(signals_store.stored_signals()) == 1
    assert signals_store.stored_signals()[0].name == "b"


def test_manager_updates_on_store_change(qtbot, signals_store, signals_csv):
    """Test that manager refreshes when store changes externally."""
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 0

    signals_store.import_file(signals_csv)

    assert dialog._table.rowCount() == 2


def test_manager_can_remove_all_stored_signals_by_selection(
    qtbot, signals_store, signals_csv
):
    """Select-all plus Remove Selected replaces the separate clear action."""
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 2

    dialog._table.selectAll()
    dialog._remove_btn.click()

    assert signals_store.stored_signals() == []
    assert dialog._table.rowCount() == 0


def test_manager_handles_multiple_selection(qtbot, signals_store, signals_csv):
    """Test removing multiple selected signals."""
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert len(signals_store.stored_signals()) == 2

    dialog._table.selectAll()
    dialog._remove_selected()

    assert signals_store.stored_signals() == []


def test_manager_export_selected_writes_only_selected_signals(
    qtbot, signals_store, signals_csv, monkeypatch, tmp_path
):
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    out_path = tmp_path / "selected.csv"
    monkeypatch.setattr(
        "confusius._napari._signals._manager.prompt_delimited_export_path",
        lambda *args, **kwargs: (out_path, ","),
    )

    dialog._table.selectRow(0)
    dialog._export_selected()

    content = out_path.read_text()
    assert "a" in content
    assert "b" not in content.split("\n")[0]


def test_manager_export_selected_shows_error_when_nothing_selected(
    qtbot, signals_store, signals_csv, monkeypatch
):
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)
    dialog._table.clearSelection()

    errors = []
    monkeypatch.setattr("confusius._napari._signals._manager.show_error", errors.append)

    dialog._export_selected()

    assert errors == ["Select one or more stored signals to export."]


def test_manager_imports_multiple_files(qtbot, signals_store, monkeypatch, tmp_path):
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    path1 = tmp_path / "a.csv"
    path1.write_text("time,a\n0,1\n1,2\n")
    path2 = tmp_path / "b.csv"
    path2.write_text("time,b\n0,3\n1,4\n")

    monkeypatch.setattr(
        "confusius._napari._signals._manager.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([str(path1), str(path2)], ""),
    )

    dialog._import_file()

    assert [signal.name for signal in signals_store.stored_signals()] == ["a", "b"]


def test_manager_import_dialog_filters_supported_formats(
    qtbot, signals_store, monkeypatch
):
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    def _fake_get_open_file_names(parent, caption, directory, file_filter):
        assert "*.tsv" in file_filter
        assert "*.tsv.gz" in file_filter
        assert "*.csv" in file_filter
        return [str(Path("/tmp/does-not-matter.tsv.gz"))], ""

    monkeypatch.setattr(
        "confusius._napari._signals._manager.QFileDialog.getOpenFileNames",
        _fake_get_open_file_names,
    )
    monkeypatch.setattr(signals_store, "import_file", lambda path: [])

    dialog._import_file()
