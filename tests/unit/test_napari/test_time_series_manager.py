"""Unit tests for the time-series manager dialog."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from confusius._napari._time_series._manager import TimeSeriesManagerDialog
from confusius._napari._time_series._store import LiveSeries


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
    time_series_store.import_file(time_series_csv)
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    assert len(time_series_store.imported_series()) == 2

    dialog._table.selectAll()
    dialog._remove_selected()

    assert time_series_store.imported_series() == []


# -- Live series tests -------------------------------------------------------


def _make_live(
    sid="point-0", name="Point 0", color="#ff0000", source_type="point", source_id=0
):
    """Create a LiveSeries for tests."""
    return LiveSeries(
        id=sid,
        name=name,
        color=color,
        visible=True,
        source_type=source_type,
        source_id=source_id,
    )


def test_manager_shows_live_and_imported(qtbot, time_series_store, time_series_csv):
    """Live series at top, imported below."""
    time_series_store.register_live_series([_make_live("point-0")])
    time_series_store.import_file(time_series_csv)

    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    assert dialog._table.rowCount() == 3  # 1 live + 2 imported
    # First row is the live series.
    assert dialog._table.item(0, 1).text() == "Point 0"
    assert dialog._table.item(0, 3).text() == "Points layer"
    # Imported rows follow.
    assert dialog._table.item(1, 1).text() == "a"
    assert dialog._table.item(2, 3).text() == "series.csv"


def test_manager_live_rename(qtbot, time_series_store):
    """Renaming a live series in the table updates the store."""
    time_series_store.register_live_series([_make_live("point-0")])
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    dialog._table.item(0, 1).setText("Barrel cortex")
    assert time_series_store.get_live_series("point-0").name == "Barrel cortex"


def test_manager_live_visibility(qtbot, time_series_store):
    """Toggling a live series checkbox updates the store."""
    time_series_store.register_live_series([_make_live("point-0")])
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    dialog._table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert time_series_store.get_live_series("point-0").visible is False


def test_manager_live_color(qtbot, time_series_store, monkeypatch):
    """Changing a live series color via the dialog updates the store."""
    time_series_store.register_live_series([_make_live("point-0")])
    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    monkeypatch.setattr(
        "confusius._napari._time_series._manager.QColorDialog.getColor",
        lambda *args, **kwargs: QColor("#00ff00"),
    )
    dialog._choose_color("point-0")
    assert time_series_store.get_live_series("point-0").color == "#00ff00"


def test_manager_remove_skips_live_series(qtbot, time_series_store, time_series_csv):
    """Removing selected rows should skip live series."""
    time_series_store.register_live_series([_make_live("point-0")])
    time_series_store.import_file(time_series_csv)

    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    dialog._table.selectAll()
    dialog._remove_selected()

    # Live series still present, imported gone.
    assert len(time_series_store.live_series()) == 1
    assert time_series_store.imported_series() == []


def test_manager_clear_all_does_not_remove_live(
    qtbot, time_series_store, time_series_csv
):
    """Clear All Imported should not affect live series."""
    time_series_store.register_live_series([_make_live("point-0")])
    time_series_store.import_file(time_series_csv)

    dialog = TimeSeriesManagerDialog(time_series_store)
    qtbot.addWidget(dialog)

    dialog._clear_btn.click()

    assert len(time_series_store.live_series()) == 1
    assert time_series_store.imported_series() == []
    assert dialog._table.rowCount() == 1  # Only live row remains
