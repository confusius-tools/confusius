"""Unit tests for the EventStore."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from confusius._napari._events._store import EventStore


@pytest.fixture
def store():
    """Return a fresh EventStore instance."""
    return EventStore()


class TestEventStore:
    """EventStore manages temporal events, colors, and display toggles."""

    def test_add_event_defaults_trial_type(self, store):
        """A blank trial type falls back to the default 'event' name."""
        store.add_event(1.0, 2.0, "")
        frame = store.events_dataframe()
        assert list(frame["trial_type"]) == ["event"]
        assert frame["onset"].tolist() == [1.0]
        assert frame["duration"].tolist() == [2.0]

    def test_add_event_rejects_non_positive_duration(self, store):
        """A zero or negative duration raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            store.add_event(1.0, 0.0, "stim")
        with pytest.raises(ValueError, match="positive"):
            store.add_event(1.0, -1.0, "stim")

    def test_color_is_stable_per_trial_type(self, store):
        """The same trial type always maps to the same color; distinct types differ."""
        first = store.color_for("stim")
        assert store.color_for("stim") == first
        assert store.color_for("cue") != first

    def test_active_events_half_open_interval(self, store):
        """An event is active on [onset, onset + duration)."""
        store.add_event(1.0, 2.0, "stim")  # active over [1, 3)
        assert store.active_events(0.5).empty
        assert list(store.active_events(1.0)["trial_type"]) == ["stim"]
        assert list(store.active_events(2.9)["trial_type"]) == ["stim"]
        assert store.active_events(3.0).empty

    def test_active_events_instantaneous(self, store, tmp_path):
        """A loaded zero-duration event is active only exactly at its onset."""
        # Zero-duration events cannot be created interactively, only loaded
        # from a BIDS events file.
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\ttrial_type\n5.0\t0.0\tblip\n")
        store.load_file(path)
        assert list(store.active_events(5.0)["trial_type"]) == ["blip"]
        assert store.active_events(5.001).empty

    def test_remove_and_clear(self, store):
        """Events can be removed by index and cleared entirely."""
        store.add_event(0.0, 1.0, "a")
        store.add_event(2.0, 1.0, "b")
        store.add_event(4.0, 1.0, "c")

        store.remove_events([1])
        assert list(store.events_dataframe()["trial_type"]) == ["a", "c"]

        store.clear()
        assert store.events_dataframe().empty

    def test_load_and_save_round_trip(self, store, tmp_path):
        """Saving then loading preserves events, including unused extra columns.

        `response_time` is never read by the plugin, but it must still survive a
        load -> save -> load cycle.
        """
        src = tmp_path / "in.tsv"
        src.write_text(
            "onset\tduration\ttrial_type\tresponse_time\n"
            "2.0\t1.0\tb\t0.30\n"
            "0.0\t0.5\ta\t0.10\n"
        )
        store.load_file(src)

        out = tmp_path / "out.tsv"
        store.save_file(out)

        other = EventStore()
        loaded = other.load_file(out)

        expected = pd.DataFrame(
            {
                "onset": [0.0, 2.0],
                "duration": [0.5, 1.0],
                "trial_type": ["a", "b"],
                "response_time": [0.10, 0.30],
            }
        )
        assert_frame_equal(loaded, expected)
        assert_frame_equal(other.events_dataframe(), expected)

    def test_changed_emitted_on_mutation(self, store, qtbot):
        """The changed signal fires when events are added."""
        with qtbot.waitSignal(store.changed, timeout=1000):
            store.add_event(0.0, 1.0, "stim")

    def test_toggle_setters_emit_only_on_change(self, store):
        """Display toggles emit changed only when the value actually changes."""
        received = []
        store.changed.connect(lambda: received.append(True))

        store.set_shade_signals(True)  # already True → no emit
        assert received == []

        store.set_shade_signals(False)
        store.set_show_in_overlay(False)
        assert len(received) == 2
