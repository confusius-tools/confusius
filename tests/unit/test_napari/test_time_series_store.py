"""Unit tests for imported and live napari time-series storage."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from confusius._napari._time_series._store import LiveSeries


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


# -- Live series tests --------------------------------------------------------


def _make_live(
    sid="point-0",
    name="Point 0",
    color="#ff0000",
    source_type="point",
    source_id=0,
    visible=True,
):
    """Create a LiveSeries with sensible defaults for tests."""
    return LiveSeries(
        id=sid,
        name=name,
        color=color,
        visible=visible,
        source_type=source_type,
        source_id=source_id,
    )


class TestLiveSeriesRegistration:
    """Test register, clear, and smart re-registration."""

    def test_register_creates_entries(self, time_series_store):
        series = [
            _make_live("point-0"),
            _make_live("point-1", name="Point 1", source_id=1),
        ]
        time_series_store.register_live_series(series)

        result = time_series_store.live_series()
        assert len(result) == 2
        assert result[0].id == "point-0"
        assert result[1].id == "point-1"

    def test_register_preserves_user_overrides(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.rename_live_series("point-0", "Cortex ROI")
        time_series_store.set_live_series_color("point-0", "#00ff00")
        time_series_store.set_live_series_visible("point-0", False)

        # Re-register with different defaults — overrides survive.
        time_series_store.register_live_series(
            [
                _make_live("point-0", name="Point 0", color="#ff0000"),
            ]
        )

        updated = time_series_store.get_live_series("point-0")
        assert updated.name == "Cortex ROI"
        assert updated.color == "#00ff00"
        assert updated.visible is False

    def test_register_removes_disappeared_ids(self, time_series_store):
        time_series_store.register_live_series(
            [
                _make_live("point-0"),
                _make_live("point-1", name="Point 1", source_id=1),
            ]
        )

        # Re-register with only point-1.
        time_series_store.register_live_series(
            [
                _make_live("point-1", name="Point 1", source_id=1),
            ]
        )

        assert time_series_store.get_live_series("point-0") is None
        assert time_series_store.get_live_series("point-1") is not None

    def test_register_adds_new_ids(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.register_live_series(
            [
                _make_live("point-0"),
                _make_live("point-1", name="Point 1", source_id=1),
            ]
        )

        assert len(time_series_store.live_series()) == 2

    def test_clear_removes_all_live_series(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.clear_live_series()

        assert time_series_store.live_series() == []

    def test_clear_does_not_affect_imported(self, time_series_store, time_series_csv):
        time_series_store.import_file(time_series_csv)
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.clear_live_series()

        assert len(time_series_store.imported_series()) == 2
        assert time_series_store.live_series() == []

    def test_imported_clear_does_not_affect_live(
        self, time_series_store, time_series_csv
    ):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.import_file(time_series_csv)
        time_series_store.clear()

        assert time_series_store.imported_series() == []
        assert len(time_series_store.live_series()) == 1


class TestLiveSeriesMutations:
    """Test rename, recolor, and visibility changes."""

    def test_rename(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.rename_live_series("point-0", "Barrel cortex")

        assert time_series_store.get_live_series("point-0").name == "Barrel cortex"

    def test_rename_strips_whitespace(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.rename_live_series("point-0", "  Cortex  ")

        assert time_series_store.get_live_series("point-0").name == "Cortex"

    def test_rename_rejects_empty(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])

        with pytest.raises(ValueError, match="cannot be empty"):
            time_series_store.rename_live_series("point-0", "   ")

    def test_rename_rejects_unknown_id(self, time_series_store):
        with pytest.raises(ValueError, match="Unknown live series"):
            time_series_store.rename_live_series("no-such", "name")

    def test_set_color(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.set_live_series_color("point-0", "#abcdef")

        assert time_series_store.get_live_series("point-0").color == "#abcdef"

    def test_set_color_rejects_empty(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])

        with pytest.raises(ValueError, match="cannot be empty"):
            time_series_store.set_live_series_color("point-0", "")

    def test_set_visible(self, time_series_store):
        time_series_store.register_live_series([_make_live("point-0")])
        time_series_store.set_live_series_visible("point-0", False)

        assert time_series_store.get_live_series("point-0").visible is False
        assert time_series_store.visible_live_series() == []

    def test_visible_filters_correctly(self, time_series_store):
        time_series_store.register_live_series(
            [
                _make_live("point-0"),
                _make_live("point-1", name="Point 1", source_id=1),
            ]
        )
        time_series_store.set_live_series_visible("point-0", False)

        visible = time_series_store.visible_live_series()
        assert len(visible) == 1
        assert visible[0].id == "point-1"


class TestLiveSeriesSignals:
    """Test signal emissions for live series operations."""

    def test_register_emits_both_signals(self, time_series_store, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.register_live_series([_make_live("point-0")])

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_register_no_change_emits_nothing(self, time_series_store, signal_spy):
        time_series_store.register_live_series([_make_live("point-0")])

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        # Same registration again.
        time_series_store.register_live_series([_make_live("point-0")])

        assert changed_spy.count == 0
        assert plot_spy.count == 0

    def test_clear_emits_both_signals(self, time_series_store, signal_spy):
        time_series_store.register_live_series([_make_live("point-0")])

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.clear_live_series()

        assert changed_spy.count == 1
        assert plot_spy.count == 1

    def test_clear_empty_emits_nothing(self, time_series_store, signal_spy):
        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.clear_live_series()

        assert changed_spy.count == 0
        assert plot_spy.count == 0

    def test_rename_emits_changed_only(self, time_series_store, signal_spy):
        time_series_store.register_live_series([_make_live("point-0")])

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.rename_live_series("point-0", "New name")

        assert changed_spy.count == 1
        assert plot_spy.count == 0

    def test_set_color_emits_changed_only(self, time_series_store, signal_spy):
        time_series_store.register_live_series([_make_live("point-0")])

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.set_live_series_color("point-0", "#abcdef")

        assert changed_spy.count == 1
        assert plot_spy.count == 0

    def test_set_visible_emits_both_signals(self, time_series_store, signal_spy):
        time_series_store.register_live_series([_make_live("point-0")])

        changed_spy = signal_spy()
        plot_spy = signal_spy()
        time_series_store.changed.connect(changed_spy)
        time_series_store.plot_data_changed.connect(plot_spy)

        time_series_store.set_live_series_visible("point-0", False)

        assert changed_spy.count == 1
        assert plot_spy.count == 1
