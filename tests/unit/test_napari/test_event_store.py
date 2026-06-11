"""Unit tests for the EventStore."""

import pytest

from confusius._napari._events._store import EventStore
from confusius.bids.events import BIDSEvent


@pytest.fixture
def store():
    """Return a fresh EventStore instance."""
    return EventStore()


class TestEventStore:
    """EventStore manages temporal events, colors, and display toggles."""

    def test_add_event_defaults_trial_type(self, store):
        """A blank trial type falls back to the default 'event' name."""
        event = store.add_event(1.0, 2.0, "")
        assert event == BIDSEvent(1.0, 2.0, "event")
        assert store.events() == [event]

    def test_add_event_rejects_negative_duration(self, store):
        """A negative duration raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            store.add_event(1.0, -1.0, "stim")

    def test_color_is_stable_per_trial_type(self, store):
        """The same trial type always maps to the same color; distinct types differ."""
        first = store.color_for("stim")
        assert store.color_for("stim") == first
        assert store.color_for("cue") != first

    def test_active_events_half_open_interval(self, store):
        """An event is active on [onset, onset + duration)."""
        store.add_event(1.0, 2.0, "stim")  # active over [1, 3)
        assert store.active_events(0.5) == []
        assert [e.trial_type for e in store.active_events(1.0)] == ["stim"]
        assert [e.trial_type for e in store.active_events(2.9)] == ["stim"]
        assert store.active_events(3.0) == []

    def test_active_events_instantaneous(self, store):
        """A zero-duration event is active only exactly at its onset."""
        store.add_event(5.0, 0.0, "blip")
        assert [e.trial_type for e in store.active_events(5.0)] == ["blip"]
        assert store.active_events(5.001) == []

    def test_remove_and_clear(self, store):
        """Events can be removed by index and cleared entirely."""
        store.add_event(0.0, 1.0, "a")
        store.add_event(2.0, 1.0, "b")
        store.add_event(4.0, 1.0, "c")

        store.remove_events([1])
        assert [e.trial_type for e in store.events()] == ["a", "c"]

        store.clear()
        assert store.events() == []

    def test_load_and_save_round_trip(self, store, tmp_path):
        """Saving then loading into a fresh store preserves events."""
        store.add_event(2.0, 1.0, "b")
        store.add_event(0.0, 0.5, "a")
        path = tmp_path / "events.tsv"
        store.save_file(path)

        other = EventStore()
        loaded = other.load_file(path)

        assert loaded == [BIDSEvent(0.0, 0.5, "a"), BIDSEvent(2.0, 1.0, "b")]
        assert other.events() == loaded

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
