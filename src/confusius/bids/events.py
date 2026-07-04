"""Read and write BIDS events files (`.tsv`).

A BIDS events file is a tab-separated table where each row describes one event in
time. The `onset` and `duration` columns (both in seconds) are required; the
optional `trial_type` column names the kind of event and defaults to `"event"`
when absent. Any additional columns are preserved on a round trip.

Events are represented as a `pandas.DataFrame` with columns ordered
`onset`, `duration`, `trial_type`, then any extra columns. This is the same
representation consumed by the GLM design-matrix tools (see
[make_first_level_design_matrix][confusius.glm.make_first_level_design_matrix]),
so an events table read here can be fed to the GLM without conversion.

See the BIDS specification for the full definition of the events file:
https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from confusius._utils.bids import DEFAULT_TRIAL_TYPE, normalize_trial_type

__all__ = [
    "DEFAULT_TRIAL_TYPE",
    "normalize_trial_type",
    "ONSET_COLUMN",
    "DURATION_COLUMN",
    "TRIAL_TYPE_COLUMN",
    "read_events",
    "write_events",
]

ONSET_COLUMN = "onset"
"""Required BIDS column holding the event onset in seconds."""

DURATION_COLUMN = "duration"
"""Required BIDS column holding the event duration in seconds."""

TRIAL_TYPE_COLUMN = "trial_type"
"""Optional BIDS column naming the kind of event."""

_REQUIRED_COLUMNS = (ONSET_COLUMN, DURATION_COLUMN)
"""BIDS columns that every events file must contain."""


def read_events(path: str | Path) -> pd.DataFrame:
    """Read a BIDS events file into an events table.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a tab-separated BIDS events file.

    Returns
    -------
    pandas.DataFrame
        Events table with columns ordered `onset` (float), `duration` (float),
        `trial_type` (str), then any extra columns from the file (preserved
        verbatim). A missing `trial_type` column or blank/`n/a` cell defaults to
        `DEFAULT_TRIAL_TYPE`.

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

    onset = pd.to_numeric(frame[ONSET_COLUMN], errors="coerce")
    duration = pd.to_numeric(frame[DURATION_COLUMN], errors="coerce")
    if onset.isna().any() or duration.isna().any():
        raise ValueError(
            "BIDS events 'onset' and 'duration' must all be numeric and present."
        )
    if (duration < 0).any():
        raise ValueError("BIDS events 'duration' values must be non-negative.")
    frame[ONSET_COLUMN] = onset.astype(float)
    frame[DURATION_COLUMN] = duration.astype(float)
    if TRIAL_TYPE_COLUMN in frame.columns:
        frame[TRIAL_TYPE_COLUMN] = [
            normalize_trial_type(value) for value in frame[TRIAL_TYPE_COLUMN]
        ]
    else:
        frame[TRIAL_TYPE_COLUMN] = DEFAULT_TRIAL_TYPE

    extra = [
        column
        for column in frame.columns
        if column not in (ONSET_COLUMN, DURATION_COLUMN, TRIAL_TYPE_COLUMN)
    ]
    ordered = [ONSET_COLUMN, DURATION_COLUMN, TRIAL_TYPE_COLUMN, *extra]
    return frame[ordered]


def write_events(path: str | Path, events: pd.DataFrame) -> None:
    """Write an events table to a BIDS events file.

    Rows are sorted by onset, as recommended by BIDS, and columns are ordered
    `onset`, `duration`, `trial_type` (when present), then any extra columns.
    Missing values are written as `"n/a"`.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path for the tab-separated events file.
    events : pandas.DataFrame
        Events table containing at least `onset` and `duration` columns. An
        empty table writes a header-only file.

    Returns
    -------
    None
        This function writes to disk and returns nothing.

    Raises
    ------
    TypeError
        If `events` is not a `pandas.DataFrame`.
    ValueError
        If `events` is missing the `onset` or `duration` column.
    """
    path = Path(path)
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")

    missing = [column for column in _REQUIRED_COLUMNS if column not in events.columns]
    if missing:
        names = ", ".join(repr(column) for column in missing)
        raise ValueError(f"events DataFrame is missing required column(s): {names}.")

    leading = [ONSET_COLUMN, DURATION_COLUMN]
    if TRIAL_TYPE_COLUMN in events.columns:
        leading.append(TRIAL_TYPE_COLUMN)
    ordered = leading + [column for column in events.columns if column not in leading]

    frame = events[ordered].sort_values(ONSET_COLUMN, kind="stable")
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")
