"""Read and write BIDS events files (`.tsv`).

This module provides a small, dependency-light representation of a BIDS task
events table together with helpers to read and write it. A BIDS events file is a
tab-separated table where each row describes one event in time. The `onset` and
`duration` columns (both in seconds) are required; the optional `trial_type`
column names the kind of event and defaults to `"event"` when absent. Any
additional columns are preserved on a round trip.

See the BIDS specification for the full definition of the events file:
https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_TRIAL_TYPE = "event"
"""Trial type assigned to events whose `trial_type` is missing or empty."""

_ONSET_COLUMN = "onset"
"""Required BIDS column holding the event onset in seconds."""

_DURATION_COLUMN = "duration"
"""Required BIDS column holding the event duration in seconds."""

_TRIAL_TYPE_COLUMN = "trial_type"
"""Optional BIDS column naming the kind of event."""

_REQUIRED_COLUMNS = (_ONSET_COLUMN, _DURATION_COLUMN)
"""BIDS columns that every events file must contain."""


@dataclass(frozen=True, slots=True)
class BIDSEvent:
    """A single BIDS task event spanning a period of time.

    Attributes
    ----------
    onset : float
        Event onset in seconds, measured from the start of the recording.
    duration : float
        Event duration in seconds. Zero denotes an instantaneous event.
    trial_type : str, default: "event"
        Name of the kind of event.
    extra : dict[str, str]
        Additional BIDS columns (e.g. `response_time`, `stim_file`) kept as
        strings so they survive a read/write round trip. Empty by default.
    """

    onset: float
    duration: float
    trial_type: str = DEFAULT_TRIAL_TYPE
    extra: dict[str, str] = field(default_factory=dict)


def normalize_trial_type(trial_type: object) -> str:
    """Return a non-empty trial type, falling back to the default.

    Parameters
    ----------
    trial_type : object
        Candidate trial type. Missing values (`None`, NaN) and blank strings
        are replaced by the default trial type.

    Returns
    -------
    str
        The stripped trial type, or `DEFAULT_TRIAL_TYPE` when the input is
        missing or empty.
    """
    if trial_type is None or (isinstance(trial_type, float) and pd.isna(trial_type)):
        return DEFAULT_TRIAL_TYPE
    text = str(trial_type).strip()
    return text or DEFAULT_TRIAL_TYPE


def read_events(path: str | Path) -> list[BIDSEvent]:
    """Read a BIDS events file into a list of events.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a tab-separated BIDS events file.

    Returns
    -------
    list[BIDSEvent]
        One event per data row, in file order.

    Raises
    ------
    ValueError
        If the `onset` or `duration` column is missing, or if any `onset` or
        `duration` value is missing or non-numeric.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep="\t")

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        names = ", ".join(repr(column) for column in missing)
        raise ValueError(f"BIDS events file is missing required column(s): {names}.")

    onset = pd.to_numeric(frame[_ONSET_COLUMN], errors="coerce")
    duration = pd.to_numeric(frame[_DURATION_COLUMN], errors="coerce")
    if onset.isna().any() or duration.isna().any():
        raise ValueError(
            "BIDS events 'onset' and 'duration' must all be numeric and present."
        )

    if _TRIAL_TYPE_COLUMN in frame.columns:
        trial_types = [
            normalize_trial_type(value) for value in frame[_TRIAL_TYPE_COLUMN]
        ]
    else:
        trial_types = [DEFAULT_TRIAL_TYPE] * len(frame)

    extra_columns = [
        column
        for column in frame.columns
        if column not in (_ONSET_COLUMN, _DURATION_COLUMN, _TRIAL_TYPE_COLUMN)
    ]

    events = []
    for i in range(len(frame)):
        extra = {
            column: str(frame[column].iloc[i])
            for column in extra_columns
            if not pd.isna(frame[column].iloc[i])
        }
        events.append(
            BIDSEvent(
                onset=float(onset.iloc[i]),
                duration=float(duration.iloc[i]),
                trial_type=trial_types[i],
                extra=extra,
            )
        )
    return events


def write_events(path: str | Path, events: list[BIDSEvent]) -> None:
    """Write events to a BIDS events file.

    Rows are sorted by onset, as recommended by BIDS. Extra columns present on
    any event are written for every row, using `"n/a"` where absent.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path for the tab-separated events file.
    events : list[BIDSEvent]
        Events to write. An empty list writes a header-only file.

    Returns
    -------
    None
        This function writes to disk and returns nothing.
    """
    path = Path(path)

    extra_keys: list[str] = []
    for event in events:
        for key in event.extra:
            if key not in extra_keys:
                extra_keys.append(key)

    columns = [_ONSET_COLUMN, _DURATION_COLUMN, _TRIAL_TYPE_COLUMN, *extra_keys]
    rows = []
    for event in sorted(events, key=lambda event: event.onset):
        row: dict[str, object] = {
            _ONSET_COLUMN: event.onset,
            _DURATION_COLUMN: event.duration,
            _TRIAL_TYPE_COLUMN: event.trial_type,
        }
        for key in extra_keys:
            row[key] = event.extra.get(key, "n/a")
        rows.append(row)

    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")
