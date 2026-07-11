"""Unit tests for imported napari signals storage."""

from __future__ import annotations

import gzip
import json

import numpy as np
import numpy.testing as npt
import pytest


def test_import_csv_creates_one_signal_per_value_column(signals_store, signals_csv):
    imported = signals_store.import_file(signals_csv)

    assert [signal.name for signal in imported] == ["a", "b"]
    npt.assert_array_equal(imported[0].x, np.array([0, 1, 2]))
    npt.assert_array_equal(imported[0].y, np.array([1, 2, 3]))
    npt.assert_array_equal(imported[1].y, np.array([4, 5, 6]))


def test_import_tsv_preserves_duplicate_time_values(signals_store, signals_tsv):
    imported = signals_store.import_file(signals_tsv)

    npt.assert_array_equal(imported[0].x, np.array([0, 0, 1]))
    npt.assert_array_equal(imported[0].y, np.array([1.0, 2.0, 3.0]))


def test_import_bids_physio_tsv_gz_uses_sidecar_columns_and_time(
    signals_store, tmp_path
):
    path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("1\t10\n2\t11\n3\t12\n")

    sidecar = path.with_suffix("").with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "Columns": ["pulse", "resp"],
                "SamplingFrequency": 2.0,
                "StartTime": -0.5,
            }
        )
    )

    imported = signals_store.import_file(path)

    assert [signal.name for signal in imported] == ["pulse", "resp"]
    npt.assert_allclose(imported[0].x, np.array([-0.5, 0.0, 0.5]))
    npt.assert_array_equal(imported[0].y, np.array([1.0, 2.0, 3.0]))
    npt.assert_array_equal(imported[1].y, np.array([10.0, 11.0, 12.0]))


def test_import_bids_physio_rejects_column_count_mismatch(signals_store, tmp_path):
    path = tmp_path / "sub-01_task-rest_recording-cardiac_physio.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("1\t10\n2\t11\n")

    path.with_suffix("").with_suffix(".json").write_text(
        json.dumps({"Columns": ["pulse"], "SamplingFrequency": 10.0})
    )

    with pytest.raises(ValueError, match="Columns' length does not match TSV width"):
        signals_store.import_file(path)


def test_import_csv_ignores_unrelated_json_sidecar(signals_store, tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text("time,a\n0,1\n1,2\n")
    path.with_suffix(".json").write_text(
        json.dumps({"Columns": ["pulse"], "SamplingFrequency": 10.0})
    )

    imported = signals_store.import_file(path)

    assert [signal.name for signal in imported] == ["a"]
    npt.assert_array_equal(imported[0].x, np.array([0, 1]))
    npt.assert_array_equal(imported[0].y, np.array([1, 2]))


def test_import_rejects_missing_time_column(signals_store, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("frame,a\n0,1\n1,2\n")

    with pytest.raises(ValueError, match="contain a 'time' column"):
        signals_store.import_file(path)


def test_import_rejects_missing_value_columns(signals_store, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("time\n0\n1\n")

    with pytest.raises(ValueError, match="at least one value column"):
        signals_store.import_file(path)


def test_import_rejects_non_numeric_value_columns(signals_store, tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("time,a,label\n0,1,foo\n1,2,bar\n")

    with pytest.raises(ValueError, match="must be numeric"):
        signals_store.import_file(path)


def test_store_can_rename_toggle_recolor_and_remove_signals(signals_store, signals_csv):
    imported = signals_store.import_file(signals_csv)

    signals_store.rename_signal(imported[0].id, "baseline")
    signals_store.set_signal_visible(imported[1].id, False)
    signals_store.set_signal_color(imported[1].id, "#123456")

    updated = signals_store.imported_signals()
    assert updated[0].name == "baseline"
    assert updated[1].visible is False
    assert updated[1].color == "#123456"

    signals_store.remove_signals([imported[0].id])
    remaining = signals_store.imported_signals()
    assert len(remaining) == 1
    assert remaining[0].id == imported[1].id


def test_store_rejects_empty_renames(signals_store, signals_csv):
    imported = signals_store.import_file(signals_csv)

    with pytest.raises(ValueError, match="cannot be empty"):
        signals_store.rename_signal(imported[0].id, "   ")


def test_store_rejects_empty_color(signals_store, signals_csv):
    imported = signals_store.import_file(signals_csv)

    with pytest.raises(ValueError, match="cannot be empty"):
        signals_store.set_signal_color(imported[0].id, "")


def test_store_generates_unique_ids_after_removal(signals_store, signals_csv, tmp_path):
    """IDs should be unique even after removing and importing new signals."""
    imported1 = signals_store.import_file(signals_csv)
    id1 = imported1[0].id

    signals_store.remove_signals([id1])

    path2 = tmp_path / "series2.csv"
    path2.write_text("time,c\n0,1\n1,2\n")
    imported2 = signals_store.import_file(path2)
    id2 = imported2[0].id

    assert id1 != id2


def test_store_clear_removes_all_signals(signals_store, signals_csv):
    signals_store.import_file(signals_csv)
    assert len(signals_store.imported_signals()) == 2

    signals_store.clear()
    assert signals_store.imported_signals() == []


class TestStoreSignals:
    """Test that signals are emitted correctly."""

    def test_import_emits_both_signals(self, signals_store, signals_csv, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.import_file(signals_csv)

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_clear_emits_both_signals(self, signals_store, signals_csv, signal_spy):
        signals_store.import_file(signals_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.clear()

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_clear_no_change_emits_nothing(self, signals_store, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.clear()

        assert changed_spy.count == 0
        assert plot_spy.count == 0

    def test_remove_emits_both_signals(self, signals_store, signals_csv, signal_spy):
        imported = signals_store.import_file(signals_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.remove_signals([imported[0].id])

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_remove_no_change_emits_nothing(self, signals_store, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.remove_signals(["non-existent"])

        assert changed_spy.count == 0
        assert plot_spy.count == 0

    def test_rename_emits_changed_signal(self, signals_store, signals_csv, signal_spy):
        imported = signals_store.import_file(signals_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.rename_signal(imported[0].id, "new_name")

        assert changed_spy.count == 1
        assert plot_spy.count == 0  # Rename doesn't affect plot data

    def test_set_visible_emits_both_signals(
        self, signals_store, signals_csv, signal_spy
    ):
        imported = signals_store.import_file(signals_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.set_signal_visible(imported[0].id, False)

        assert changed_spy.count == 1
        assert plot_spy.count == 1  # Visibility affects plot data

    def test_set_color_emits_changed_signal(
        self, signals_store, signals_csv, signal_spy
    ):
        imported = signals_store.import_file(signals_csv)

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        signals_store.changed.connect(changed_spy)
        signals_store.plot_data_changed.connect(plot_spy)

        signals_store.set_signal_color(imported[0].id, "#123456")

        assert changed_spy.count == 1
        assert plot_spy.count == 0  # Color change doesn't affect plot data
