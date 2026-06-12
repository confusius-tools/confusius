"""Tests for the confusius.bids.events module."""

import pytest

from confusius.bids.events import (
    DEFAULT_TRIAL_TYPE,
    BIDSEvent,
    read_events,
    write_events,
)


class TestReadEvents:
    """Reading BIDS events files."""

    def test_reads_required_columns(self, tmp_path):
        """onset, duration, and trial_type are parsed into events."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\ttrial_type\n1.0\t2.0\tstim\n4.5\t0.5\tcue\n")

        events = read_events(path)

        assert events == [
            BIDSEvent(1.0, 2.0, "stim"),
            BIDSEvent(4.5, 0.5, "cue"),
        ]

    def test_defaults_trial_type_when_column_absent(self, tmp_path):
        """Missing trial_type column yields the default trial type."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\n0.0\t1.0\n")

        events = read_events(path)

        assert events == [BIDSEvent(0.0, 1.0, DEFAULT_TRIAL_TYPE)]

    def test_defaults_trial_type_when_cell_missing(self, tmp_path):
        """An empty or n/a trial_type cell falls back to the default."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\ttrial_type\n0.0\t1.0\tn/a\n2.0\t1.0\t\n")

        events = read_events(path)

        assert [event.trial_type for event in events] == [
            DEFAULT_TRIAL_TYPE,
            DEFAULT_TRIAL_TYPE,
        ]

    def test_preserves_extra_columns(self, tmp_path):
        """Columns beyond the BIDS basics are kept on the event."""
        path = tmp_path / "events.tsv"
        path.write_text(
            "onset\tduration\ttrial_type\tresponse_time\n1.0\t2.0\tstim\t0.42\n"
        )

        events = read_events(path)

        assert events[0].extra == {"response_time": "0.42"}

    def test_missing_required_column_raises(self, tmp_path):
        """A file without a duration column is rejected."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\ttrial_type\n0.0\tstim\n")

        with pytest.raises(ValueError, match="duration"):
            read_events(path)

    def test_non_numeric_onset_raises(self, tmp_path):
        """A non-numeric onset value is rejected."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\nstart\t1.0\n")

        with pytest.raises(ValueError, match="numeric"):
            read_events(path)


class TestWriteEvents:
    """Writing BIDS events files."""

    def test_writes_tab_separated_sorted_by_onset(self, tmp_path):
        """Events are written tab-separated and ordered by onset."""
        path = tmp_path / "events.tsv"
        write_events(
            path,
            [BIDSEvent(4.0, 1.0, "late"), BIDSEvent(1.0, 2.0, "early")],
        )

        lines = path.read_text().splitlines()
        assert lines[0] == "onset\tduration\ttrial_type"
        assert lines[1].split("\t")[:3] == ["1.0", "2.0", "early"]
        assert lines[2].split("\t")[:3] == ["4.0", "1.0", "late"]

    def test_round_trip_preserves_events(self, tmp_path):
        """Writing then reading returns the same events (onset-sorted)."""
        path = tmp_path / "events.tsv"
        events = [
            BIDSEvent(2.0, 1.0, "b", {"stim_file": "b.png"}),
            BIDSEvent(0.0, 0.5, "a"),
        ]
        write_events(path, events)

        loaded = read_events(path)

        # The "a" event had no stim_file, so it is written as "n/a" and read back
        # as a missing (skipped) extra column.
        assert loaded[0] == BIDSEvent(0.0, 0.5, "a")
        assert loaded[1] == BIDSEvent(2.0, 1.0, "b", {"stim_file": "b.png"})

    def test_empty_events_writes_header_only(self, tmp_path):
        """An empty event list still writes a valid header."""
        path = tmp_path / "events.tsv"
        write_events(path, [])

        assert path.read_text().splitlines() == ["onset\tduration\ttrial_type"]
