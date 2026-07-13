"""Tests for the confusius.bids.events module."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from confusius.bids.events import (
    DEFAULT_TRIAL_TYPE,
    DURATION_COLUMN,
    ONSET_COLUMN,
    TRIAL_TYPE_COLUMN,
    load_events,
    save_events,
)


def test_exports_canonical_bids_column_names():
    """The module exposes the shared BIDS events column names."""
    assert ONSET_COLUMN == "onset"
    assert DURATION_COLUMN == "duration"
    assert TRIAL_TYPE_COLUMN == "trial_type"


class TestReadEvents:
    """Reading BIDS events files into a DataFrame."""

    def test_reads_required_columns(self, tmp_path):
        """onset, duration, and trial_type are parsed into a typed table."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\ttrial_type\n1.0\t2.0\tstim\n4.5\t0.5\tcue\n")

        events = load_events(path)

        expected = pd.DataFrame(
            {
                "onset": [1.0, 4.5],
                "duration": [2.0, 0.5],
                "trial_type": ["stim", "cue"],
            }
        )
        assert_frame_equal(events, expected)

    def test_defaults_trial_type_when_column_absent(self, tmp_path):
        """Missing trial_type column yields the default trial type."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\n0.0\t1.0\n")

        events = load_events(path)

        assert list(events["trial_type"]) == [DEFAULT_TRIAL_TYPE]

    def test_defaults_trial_type_when_cell_missing(self, tmp_path):
        """An empty or n/a trial_type cell falls back to the default."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\ttrial_type\n0.0\t1.0\tn/a\n2.0\t1.0\t\n")

        events = load_events(path)

        assert list(events["trial_type"]) == [DEFAULT_TRIAL_TYPE, DEFAULT_TRIAL_TYPE]

    def test_preserves_extra_columns(self, tmp_path):
        """Columns beyond the BIDS basics are kept, after the canonical three."""
        path = tmp_path / "events.tsv"
        path.write_text(
            "onset\tduration\ttrial_type\tresponse_time\n1.0\t2.0\tstim\t0.42\n"
        )

        events = load_events(path)

        assert list(events.columns) == [
            "onset",
            "duration",
            "trial_type",
            "response_time",
        ]
        assert events["response_time"].tolist() == [0.42]

    def test_missing_required_column_raises(self, tmp_path):
        """A file without a duration column is rejected."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\ttrial_type\n0.0\tstim\n")

        with pytest.raises(ValueError, match="duration"):
            load_events(path)

    def test_non_numeric_onset_raises(self, tmp_path):
        """A non-numeric onset value is rejected."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\nstart\t1.0\n")

        with pytest.raises(ValueError, match="numeric"):
            load_events(path)

    def test_negative_duration_raises(self, tmp_path):
        """A negative duration value is rejected."""
        path = tmp_path / "events.tsv"
        path.write_text("onset\tduration\n0.0\t-1.0\n")

        with pytest.raises(ValueError, match="non-negative"):
            load_events(path)


class TestWriteEvents:
    """Writing BIDS events DataFrames."""

    def test_writes_tab_separated_sorted_by_onset(self, tmp_path):
        """Events are written tab-separated and ordered by onset."""
        path = tmp_path / "events.tsv"
        events = pd.DataFrame(
            {
                "onset": [4.0, 1.0],
                "duration": [1.0, 2.0],
                "trial_type": ["late", "early"],
            }
        )

        save_events(path, events)

        lines = path.read_text().splitlines()
        assert lines[0] == "onset\tduration\ttrial_type"
        assert lines[1].split("\t")[:3] == ["1.0", "2.0", "early"]
        assert lines[2].split("\t")[:3] == ["4.0", "1.0", "late"]

    def test_round_trip_preserves_events_and_extra_columns(self, tmp_path):
        """Writing then reading returns the same table, including extra columns."""
        path = tmp_path / "events.tsv"
        events = pd.DataFrame(
            {
                "onset": [2.0, 0.0],
                "duration": [1.0, 0.5],
                "trial_type": ["b", "a"],
                # The "a" row has no stim_file; it round-trips as "n/a" -> NaN.
                "stim_file": ["b.png", None],
            }
        )

        save_events(path, events)
        loaded = load_events(path)

        expected = pd.DataFrame(
            {
                "onset": [0.0, 2.0],
                "duration": [0.5, 1.0],
                "trial_type": ["a", "b"],
                "stim_file": [np.nan, "b.png"],
            }
        )
        assert_frame_equal(loaded, expected)

    def test_empty_events_writes_header_only(self, tmp_path):
        """An empty events table still writes a valid header."""
        path = tmp_path / "events.tsv"
        events = pd.DataFrame({"onset": [], "duration": [], "trial_type": []})

        save_events(path, events)

        assert path.read_text().splitlines() == ["onset\tduration\ttrial_type"]

    def test_rejects_non_dataframe(self, tmp_path):
        """A non-DataFrame events argument is rejected."""
        path = tmp_path / "events.tsv"
        with pytest.raises(TypeError, match="DataFrame"):
            save_events(path, [(0.0, 1.0, "stim")])  # ty: ignore[invalid-argument-type]

    def test_rejects_missing_required_column(self, tmp_path):
        """A DataFrame without a duration column is rejected."""
        path = tmp_path / "events.tsv"
        events = pd.DataFrame({"onset": [0.0], "trial_type": ["stim"]})
        with pytest.raises(ValueError, match="duration"):
            save_events(path, events)
