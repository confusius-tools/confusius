"""Unit tests for the imported signals manager dialog."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from confusius._napari._signals._manager import SignalsManagerDialog


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
    assert signals_store.imported_signals()[0].name == "baseline"

    visible_item = dialog._table.item(0, 0)
    assert visible_item is not None
    visible_item.setCheckState(Qt.CheckState.Unchecked)
    assert signals_store.imported_signals()[0].visible is False

    monkeypatch.setattr(
        "confusius._napari._signals._manager.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    dialog._choose_color(imported[0].id)
    assert signals_store.imported_signals()[0].color == "#123456"

    dialog._table.selectRow(0)
    dialog._remove_selected()
    # Should have removed only the first signal, leaving the second.
    assert len(signals_store.imported_signals()) == 1
    assert signals_store.imported_signals()[0].name == "b"


def test_manager_updates_on_store_change(qtbot, signals_store, signals_csv):
    """Test that manager refreshes when store changes externally."""
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 0

    signals_store.import_file(signals_csv)

    assert dialog._table.rowCount() == 2


def test_manager_clear_all_button(qtbot, signals_store, signals_csv):
    """Test the Clear All button removes all signals."""
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 2

    dialog._clear_btn.click()

    assert signals_store.imported_signals() == []
    assert dialog._table.rowCount() == 0


def test_manager_handles_multiple_selection(qtbot, signals_store, signals_csv):
    """Test removing multiple selected signals."""
    signals_store.import_file(signals_csv)
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    assert len(signals_store.imported_signals()) == 2

    dialog._table.selectAll()
    dialog._remove_selected()

    assert signals_store.imported_signals() == []


def test_manager_import_dialog_filters_supported_formats(
    qtbot, signals_store, monkeypatch
):
    dialog = SignalsManagerDialog(signals_store)
    qtbot.addWidget(dialog)

    def _fake_get_open_file_name(parent, caption, directory, file_filter):
        assert "*.tsv" in file_filter
        assert "*.tsv.gz" in file_filter
        assert "*.csv" in file_filter
        return str(Path("/tmp/does-not-matter.tsv.gz")), ""

    monkeypatch.setattr(
        "confusius._napari._signals._manager.QFileDialog.getOpenFileName",
        _fake_get_open_file_name,
    )
    monkeypatch.setattr(signals_store, "import_file", lambda path: [])

    dialog._import_file()
