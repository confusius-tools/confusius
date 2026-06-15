"""Shared store for BIDS temporal events in the napari plugin."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from qtpy.QtCore import QObject, Signal

from confusius._napari._utils import CATEGORICAL_COLORS
from confusius.bids.events import (
    DEFAULT_TRIAL_TYPE,
    read_events,
    write_events,
)

ONSET_COLUMN = "onset"
"""Events DataFrame column holding the event onset in seconds."""

DURATION_COLUMN = "duration"
"""Events DataFrame column holding the event duration in seconds."""

TRIAL_TYPE_COLUMN = "trial_type"
"""Events DataFrame column naming the kind of event."""


class EventStore(QObject):
    """Store the temporal events shared between the event panel, plotter and overlay.

    The store owns the events table as a `pandas.DataFrame` with `onset`,
    `duration`, and `trial_type` columns (plus any extra columns carried in from a
    loaded BIDS events file). This is the same representation consumed by the GLM
    design-matrix tools, so the events can be fed to the GLM without conversion. The
    store assigns a stable color per trial type and tracks the two display toggles
    (shade events on the signal plot, show active events in the time overlay). It
    emits `changed` whenever its contents or toggles change so the plotter and
    overlay can refresh.

    Parameters
    ----------
    parent : QObject, optional
        Optional Qt parent.
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._events: pd.DataFrame = _empty_events()
        self._colors: dict[str, str] = {}
        self._color_counter: int = 0
        self._shade_signals: bool = True
        self._show_in_overlay: bool = True

    # -- display toggles -------------------------------------------------------

    @property
    def shade_signals(self) -> bool:
        """Whether event bands should be shaded on the signal plot.

        Returns
        -------
        bool
            Whether shading is enabled.
        """
        return self._shade_signals

    @property
    def show_in_overlay(self) -> bool:
        """Whether active events should be named in the time overlay.

        Returns
        -------
        bool
            Whether the overlay readout is enabled.
        """
        return self._show_in_overlay

    def set_shade_signals(self, enabled: bool) -> None:
        """Enable or disable shading event bands on the signal plot.

        Parameters
        ----------
        enabled : bool
            Whether to shade event bands.
        """
        if self._shade_signals != enabled:
            self._shade_signals = enabled
            self.changed.emit()

    def set_show_in_overlay(self, enabled: bool) -> None:
        """Enable or disable naming active events in the time overlay.

        Parameters
        ----------
        enabled : bool
            Whether to show active events in the overlay.
        """
        if self._show_in_overlay != enabled:
            self._show_in_overlay = enabled
            self.changed.emit()

    # -- queries ---------------------------------------------------------------

    def events_dataframe(self) -> pd.DataFrame:
        """Return a copy of the events table.

        Returns
        -------
        pandas.DataFrame
            The stored events with columns `onset`, `duration`, `trial_type`, then
            any extra columns. The row order matches insertion order. Suitable to
            pass directly as the `events` argument of
            [make_first_level_design_matrix][confusius.glm.make_first_level_design_matrix].
        """
        return self._events.copy()

    def trial_types(self) -> list[str]:
        """Return the distinct trial types in first-seen order.

        Returns
        -------
        list[str]
            Unique trial types across all stored events.
        """
        return list(dict.fromkeys(self._events[TRIAL_TYPE_COLUMN]))

    def color_for(self, trial_type: str) -> str:
        """Return a stable hex color for a trial type, assigning one if needed.

        Parameters
        ----------
        trial_type : str
            Trial type to look up.

        Returns
        -------
        str
            Hex color string assigned to this trial type.
        """
        if trial_type not in self._colors:
            self._colors[trial_type] = CATEGORICAL_COLORS[
                self._color_counter % len(CATEGORICAL_COLORS)
            ]
            self._color_counter += 1
        return self._colors[trial_type]

    def active_events(self, time: float) -> pd.DataFrame:
        """Return the events that are ON at a given time.

        An event is active over the half-open interval ``[onset, onset + duration)``.
        Instantaneous events (zero duration) are active only exactly at their onset.

        Parameters
        ----------
        time : float
            Time value to test, in the same units as the events' onsets.

        Returns
        -------
        pandas.DataFrame
            The subset of the events table active at *time*, preserving columns and
            row order.
        """
        onsets = self._events[ONSET_COLUMN]
        durations = self._events[DURATION_COLUMN]
        ends = onsets + durations
        instantaneous = durations <= 0
        mask = ((onsets <= time) & (time < ends)) | (instantaneous & (onsets == time))
        return self._events[mask]

    # -- mutations -------------------------------------------------------------

    def add_event(
        self, onset: float, duration: float, trial_type: str | None = None
    ) -> None:
        """Create and store a new event.

        Parameters
        ----------
        onset : float
            Event onset in seconds.
        duration : float
            Event duration in seconds. Must be positive.
        trial_type : str, optional
            Trial type name. Missing or blank values use the default trial type.

        Raises
        ------
        ValueError
            If `duration` is negative or zero.
        """
        if duration <= 0:
            raise ValueError("Event duration must be positive.")
        name = (trial_type or "").strip() or DEFAULT_TRIAL_TYPE
        self.color_for(name)
        new_row = pd.DataFrame(
            {
                ONSET_COLUMN: [float(onset)],
                DURATION_COLUMN: [float(duration)],
                TRIAL_TYPE_COLUMN: [name],
            }
        )
        self._events = self._concat(new_row)
        self.changed.emit()

    def remove_events(self, indices: list[int]) -> None:
        """Remove events by positional index.

        Parameters
        ----------
        indices : list[int]
            Indices into the current event list to remove.
        """
        drop = set(indices)
        keep = [i for i in range(len(self._events)) if i not in drop]
        if len(keep) != len(self._events):
            self._events = self._events.iloc[keep].reset_index(drop=True)
            self.changed.emit()

    def clear(self) -> None:
        """Remove all events."""
        if len(self._events):
            self._events = _empty_events()
            self.changed.emit()

    def load_file(self, path: str | Path) -> pd.DataFrame:
        """Load events from a BIDS events file and append them to the store.

        Every column from the file is retained, including ones the plugin does not
        display, so they survive a later [save_file][confusius._napari._events._store.EventStore.save_file].

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a tab-separated BIDS events file.

        Returns
        -------
        pandas.DataFrame
            The events read from the file.

        Raises
        ------
        ValueError
            If the file is not a valid BIDS events file (propagated from
            `confusius.bids.events.read_events`).
        """
        loaded = read_events(path)
        for trial_type in loaded[TRIAL_TYPE_COLUMN]:
            self.color_for(trial_type)
        self._events = self._concat(loaded)
        self.changed.emit()
        return loaded

    def save_file(self, path: str | Path) -> None:
        """Write all stored events to a BIDS events file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path for the tab-separated events file.
        """
        write_events(path, self._events)

    def _concat(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Append rows to the events table, aligning columns.

        Parameters
        ----------
        rows : pandas.DataFrame
            One or more event rows to append.

        Returns
        -------
        pandas.DataFrame
            The combined table with a fresh range index. Concatenation aligns on
            column name, so extra columns present on either side are filled with
            NaN where absent.
        """
        # Avoid pandas' deprecated all-NA/empty concatenation path by short-
        # circuiting when the store is still empty.
        if self._events.empty:
            return rows.reset_index(drop=True)
        return pd.concat([self._events, rows], ignore_index=True)


def _empty_events() -> pd.DataFrame:
    """Return an empty events table with the canonical typed columns.

    Returns
    -------
    pandas.DataFrame
        A zero-row frame with float `onset`/`duration` and object `trial_type`
        columns.
    """
    return pd.DataFrame(
        {
            ONSET_COLUMN: pd.Series(dtype=float),
            DURATION_COLUMN: pd.Series(dtype=float),
            TRIAL_TYPE_COLUMN: pd.Series(dtype=object),
        }
    )
