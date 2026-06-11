"""Shared store for BIDS temporal events in the napari plugin."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QObject, Signal

from confusius.bids.events import (
    DEFAULT_TRIAL_TYPE,
    BIDSEvent,
    read_events,
    write_events,
)

_TRIAL_TYPE_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]
"""Qualitative palette cycled across distinct trial types."""


class EventStore(QObject):
    """Store the temporal events shared between the event panel, plotter and overlay.

    The store owns the list of `confusius.bids.events.BIDSEvent` objects, assigns a
    stable color per trial type, and tracks the two display toggles (shade events on
    the signal plot, show active events in the time overlay). It emits `changed`
    whenever its contents or toggles change so the plotter and overlay can refresh.

    Parameters
    ----------
    parent : QObject, optional
        Optional Qt parent.
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._events: list[BIDSEvent] = []
        self._colors: dict[str, str] = {}
        self._color_counter: int = 0
        self._shade_signals: bool = True
        self._show_in_overlay: bool = True

    # -- display toggles -------------------------------------------------------

    @property
    def shade_signals(self) -> bool:
        """Whether event bands should be shaded on the signal plot."""
        return self._shade_signals

    @property
    def show_in_overlay(self) -> bool:
        """Whether active events should be named in the time overlay."""
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

    def events(self) -> list[BIDSEvent]:
        """Return all events in insertion order.

        Returns
        -------
        list[BIDSEvent]
            The stored events.
        """
        return list(self._events)

    def trial_types(self) -> list[str]:
        """Return the distinct trial types in first-seen order.

        Returns
        -------
        list[str]
            Unique trial types across all stored events.
        """
        return list(dict.fromkeys(event.trial_type for event in self._events))

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
            self._colors[trial_type] = _TRIAL_TYPE_COLORS[
                self._color_counter % len(_TRIAL_TYPE_COLORS)
            ]
            self._color_counter += 1
        return self._colors[trial_type]

    def active_events(self, time: float) -> list[BIDSEvent]:
        """Return events that are ON at a given time.

        An event is active over the half-open interval ``[onset, onset + duration)``.
        Instantaneous events (zero duration) are active only exactly at their onset.

        Parameters
        ----------
        time : float
            Time value to test, in the same units as the events' onsets.

        Returns
        -------
        list[BIDSEvent]
            Events active at *time*.
        """
        active = []
        for event in self._events:
            end = event.onset + event.duration
            if event.duration <= 0:
                if time == event.onset:
                    active.append(event)
            elif event.onset <= time < end:
                active.append(event)
        return active

    # -- mutations -------------------------------------------------------------

    def add_event(
        self, onset: float, duration: float, trial_type: str | None = None
    ) -> BIDSEvent:
        """Create and store a new event.

        Parameters
        ----------
        onset : float
            Event onset in seconds.
        duration : float
            Event duration in seconds. Must be non-negative.
        trial_type : str, optional
            Trial type name. Missing or blank values use the default trial type.

        Returns
        -------
        BIDSEvent
            The newly created event.

        Raises
        ------
        ValueError
            If `duration` is negative.
        """
        if duration < 0:
            raise ValueError("Event duration must be non-negative.")
        name = (trial_type or "").strip() or DEFAULT_TRIAL_TYPE
        event = BIDSEvent(onset=float(onset), duration=float(duration), trial_type=name)
        self.color_for(name)
        self._events.append(event)
        self.changed.emit()
        return event

    def remove_events(self, indices: list[int]) -> None:
        """Remove events by index.

        Parameters
        ----------
        indices : list[int]
            Indices into the current event list to remove.
        """
        drop = set(indices)
        kept = [event for i, event in enumerate(self._events) if i not in drop]
        if len(kept) != len(self._events):
            self._events = kept
            self.changed.emit()

    def clear(self) -> None:
        """Remove all events."""
        if self._events:
            self._events.clear()
            self.changed.emit()

    def load_file(self, path: str | Path) -> list[BIDSEvent]:
        """Load events from a BIDS events file and append them to the store.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a tab-separated BIDS events file.

        Returns
        -------
        list[BIDSEvent]
            The events read from the file.

        Raises
        ------
        ValueError
            If the file is not a valid BIDS events file (propagated from
            `confusius.bids.events.read_events`).
        """
        loaded = read_events(path)
        for event in loaded:
            self.color_for(event.trial_type)
        self._events.extend(loaded)
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
