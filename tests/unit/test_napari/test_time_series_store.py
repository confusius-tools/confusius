"""Unit tests for imported napari time-series storage."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from confusius._napari._time_series._store import TimeSeriesStore


def test_import_csv_creates_one_series_per_value_column(
    time_series_store, time_series_csv
):
    imported = time_series_store.import_file(time_series_csv)

    assert [series.name for series in imported] == ["a", "b"]
    npt.assert_array_equal(imported[0].x, np.array([0, 1, 2]))
    npt.assert_array_equal(imported[0].y, np.array([1, 2, 3]))
    npt.assert_array_equal(imported[1].y, np.array([4, 5, 6]))


def test_import_tsv_preserves_duplicate_time_values(time_series_store, time_series_tsv):
    imported = time_series_store.import_file(time_series_tsv)

    npt.assert_array_equal(imported[0].x, np.array([0, 0, 1]))
    npt.assert_array_equal(imported[0].y, np.array([1.0, 2.0, 3.0]))


def test_import_rejects_missing_time_column(time_series_store, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("frame,a\n0,1\n1,2\n")

    with pytest.raises(ValueError, match="contain a 'time' column"):
        time_series_store.import_file(path)


def test_import_rejects_missing_value_columns(time_series_store, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("time\n0\n1\n")

    with pytest.raises(ValueError, match="at least one value column"):
        time_series_store.import_file(path)


def test_import_rejects_non_numeric_value_columns(time_series_store, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("time,a,label\n0,1,foo\n1,2,bar\n")

    with pytest.raises(ValueError, match="must be numeric"):
        time_series_store.import_file(path)


def test_store_can_rename_toggle_recolor_and_remove_series(
    time_series_store, time_series_csv
):
    imported = time_series_store.import_file(time_series_csv)

    time_series_store.rename_series(imported[0].id, "baseline")
    time_series_store.set_series_visible(imported[1].id, False)
    time_series_store.set_series_color(imported[1].id, "#123456")

    updated = time_series_store.imported_series()
    assert updated[0].name == "baseline"
    assert updated[1].visible is False
    assert updated[1].color == "#123456"

    time_series_store.remove_series([imported[0].id])
    remaining = time_series_store.imported_series()
    assert len(remaining) == 1
    assert remaining[0].id == imported[1].id


def test_store_rejects_empty_renames(time_series_store, time_series_csv):
    imported = time_series_store.import_file(time_series_csv)

    with pytest.raises(ValueError, match="cannot be empty"):
        time_series_store.rename_series(imported[0].id, "   ")


def test_store_rejects_empty_color(time_series_store, time_series_csv):
    imported = time_series_store.import_file(time_series_csv)

    with pytest.raises(ValueError, match="cannot be empty"):
        time_series_store.set_series_color(imported[0].id, "")


def test_store_generates_unique_ids_after_removal(
    time_series_store, time_series_csv, tmp_path
):
    """IDs should be unique even after removing and importing new series."""
    imported1 = time_series_store.import_file(time_series_csv)
    id1 = imported1[0].id

    time_series_store.remove_series([id1])

    path2 = tmp_path / "series2.csv"
    path2.write_text("time,c\n0,1\n1,2\n")
    imported2 = time_series_store.import_file(path2)
    id2 = imported2[0].id

    assert id1 != id2


def test_store_clear_removes_all_series(time_series_store, time_series_csv):
    time_series_store.import_file(time_series_csv)
    assert len(time_series_store.imported_series()) == 2

    time_series_store.clear()
    assert time_series_store.imported_series() == []


class TestStoreSignals:
    """Test that signals are emitted correctly."""

    def test_import_emits_both_signals(
        self, time_series_store, time_series_csv, signal_spy
    ):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.import_file(time_series_csv)

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_clear_emits_both_signals(
        self, time_series_store, time_series_csv, signal_spy
    ):
        time_series_store.import_file(time_series_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.clear()

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_clear_no_change_emits_nothing(self, time_series_store, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.clear()

        assert changed_spy.count == 0
        assert plot_spy.count == 0

    def test_remove_emits_both_signals(
        self, time_series_store, time_series_csv, signal_spy
    ):
        imported = time_series_store.import_file(time_series_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.remove_series([imported[0].id])

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_remove_no_change_emits_nothing(self, time_series_store, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.remove_series(["non-existent"])

        assert changed_spy.count == 0
        assert plot_spy.count == 0

    def test_rename_emits_changed_signal(
        self, time_series_store, time_series_csv, signal_spy
    ):
        imported = time_series_store.import_file(time_series_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.rename_series(imported[0].id, "new_name")

        assert changed_spy.count == 1
        assert plot_spy.count == 0  # Rename doesn't affect plot data

    def test_set_visible_emits_both_signals(
        self, time_series_store, time_series_csv, signal_spy
    ):
        imported = time_series_store.import_file(time_series_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.set_series_visible(imported[0].id, False)

        assert changed_spy.count == 1
        assert plot_spy.count == 1  # Visibility affects plot data

    def test_set_color_emits_changed_signal(
        self, time_series_store, time_series_csv, signal_spy
    ):
        imported = time_series_store.import_file(time_series_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.set_series_color(imported[0].id, "#123456")

        assert changed_spy.count == 1
        assert plot_spy.count == 0  # Color change doesn't affect plot data
