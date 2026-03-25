"""Unit tests for the imported time-series manager dialog."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from confusius._napari._time_series._manager import TimeSeriesManagerDialog


def test_manager_refreshes_rows_from_store(qtbot, time_series_store, time_series_csv):
    time_series_store.import_file(time_series_csv)
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 2
    assert dialog._table.item(0, 1).text() == "a"
    assert dialog._table.item(1, 3).text() == "series.csv"


def test_manager_applies_store_mutations(
    qtbot, time_series_store, time_series_csv, monkeypatch
):
    imported = time_series_store.import_file(time_series_csv)
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    name_item = dialog._table.item(0, 1)
    name_item.setText("baseline")
    assert time_series_store.imported_series()[0].name == "baseline"

    visible_item = dialog._table.item(0, 0)
    visible_item.setCheckState(Qt.CheckState.Unchecked)
    assert time_series_store.imported_series()[0].visible is False

    monkeypatch.setattr(
        "confusius._napari._time_series._manager.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )
    dialog._choose_color(imported[0].id)
    assert time_series_store.imported_series()[0].color == "#123456"

    dialog._table.selectRow(0)
    dialog._remove_selected()
    # Should have removed only the first series, leaving the second.
    assert len(time_series_store.imported_series()) == 1
    assert time_series_store.imported_series()[0].name == "b"


def test_manager_updates_on_store_change(qtbot, time_series_store, time_series_csv):
    """Test that manager refreshes when store changes externally."""
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 0

    time_series_store.import_file(time_series_csv)

    assert dialog._table.rowCount() == 2


def test_manager_clear_all_button(qtbot, time_series_store, time_series_csv):
    """Test the Clear All button removes all series."""
    time_series_store.import_file(time_series_csv)
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 2

    dialog._clear_btn.click()

    assert time_series_store.imported_series() == []
    assert dialog._table.rowCount() == 0


def test_manager_handles_multiple_selection(qtbot, time_series_store, time_series_csv):
    """Test removing multiple selected series."""
    imported = time_series_store.import_file(time_series_csv)
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    assert len(time_series_store.imported_series()) == 2

    dialog._table.selectAll()
    dialog._remove_selected()

    assert time_series_store.imported_series() == []
