"""Shared store for stored (imported or pinned) napari signals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from qtpy.QtCore import QObject, Signal

from confusius._napari._utils import CATEGORICAL_COLORS
from confusius.bids import load_physio

_STORED_SIGNAL_COLORS = CATEGORICAL_COLORS
"""Palette cycled across stored signal columns."""


@dataclass(frozen=True, slots=True)
class StoredSignal:
    """Persistent signal stored for overlay in the plotter.

    Unlike `LiveSignal`, a stored signal owns its data outright and survives
    source-mode switches. It either comes from an imported file, or is a
    snapshot pinned from a live signal (see
    [`SignalStore.pin_signal`][confusius._napari._signals._store.SignalStore.pin_signal]).

    Attributes
    ----------
    id : str
        Stable signal identifier.
    name : str
        Display name used in legends and exports.
    x : numpy.ndarray
        Time values.
    y : numpy.ndarray
        Signal values.
    visible : bool
        Whether the signal should be plotted.
    color : str
        Hex color used for plotting.
    source_label : str
        Human-readable source description (a file name, or e.g. `"Pinned voxel"`).
    file_path : pathlib.Path | None
        Original imported file, or `None` for a pinned signal.
    original_column_name : str | None
        Column name from the imported file, or `None` for a pinned signal.
    pin_origin : str | None, optional
        Identifier of the live signal this was pinned from (e.g. `"label-3"`,
        `"mouse-12-34-56"`), or `None` for a signal imported from a file.
        Re-pinning the same origin updates this entry in place instead of
        creating a duplicate, and a pinned entry is hidden from the plot
        whenever a live signal with a matching id is currently active.
    """

    id: str
    name: str
    x: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    visible: bool
    color: str
    source_label: str
    file_path: Path | None
    original_column_name: str | None
    pin_origin: str | None = None


@dataclass(frozen=True, slots=True)
class LiveSignal:
    """Live signal backed by a napari layer.

    Unlike `StoredSignal`, a live signal does not own its data: the plotter extracts
    it from layers on each update, and it only exists while its source mode is active
    — switching source modes replaces the whole live set. The store tracks only
    presentation metadata (name, color, visibility) so the user can customise the
    plot; to keep a live signal's current values around after switching modes, pin it
    with [`SignalStore.pin_signal`][confusius._napari._signals._store.SignalStore.pin_signal].

    Attributes
    ----------
    id : str
        Stable identifier (e.g. `"mouse-0"`, `"point-3"`, `"label-5"`).
    name : str
        Display name used in legends (editable by the user).
    color : str
        Hex color for the plot line.
    visible : bool
        Whether the signal should be plotted.
    source_type : `"mouse"` | `"point"` | `"label"`
        Kind of napari source that produces this signal.
    source_id : int | None
        `None` for mouse, point index for points, label integer for labels.
    """

    id: str
    name: str
    color: str
    visible: bool
    source_type: Literal["mouse", "point", "label"]
    source_id: int | None


class SignalStore(QObject):
    """Store stored and live signals shared between the panel and plotter.

    The store is the single source of truth for signal presentation metadata (name,
    color, visibility). It is shared between the panel, plotter, and manager dialog.

    Parameters
    ----------
    parent : QObject | None, optional
        Optional Qt parent.
    """

    changed = Signal()
    plot_data_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._stored_signals: list[StoredSignal] = []
        self._live_signals: dict[str, LiveSignal] = {}
        self._id_counter: int = 0

    def stored_signals(self) -> list[StoredSignal]:
        """Return all stored signals.

        Returns
        -------
        list[StoredSignal]
            Stored signals in insertion order.
        """
        return list(self._stored_signals)

    def visible_stored_signals(self) -> list[StoredSignal]:
        """Return only stored signals marked as visible.

        Returns
        -------
        list[StoredSignal]
            Visible stored signals.
        """
        return [signal for signal in self._stored_signals if signal.visible]

    def _emit_changed(self, *, plot_data: bool = False) -> None:
        """Emit changed signals.

        Parameters
        ----------
        plot_data : bool, optional
            Whether to also emit `plot_data_changed`. Defaults to False.
        """
        if plot_data:
            self.plot_data_changed.emit()
        self.changed.emit()

    def clear(self) -> None:
        """Remove all stored signals."""
        if not self._stored_signals:
            return
        self._stored_signals.clear()
        self._emit_changed(plot_data=True)

    def rename_signal(self, signal_id: str, new_name: str) -> None:
        """Rename one stored signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        new_name : str
            New display name.

        Raises
        ------
        ValueError
            If the name is empty or the signal does not exist.
        """
        stripped = new_name.strip()
        if not stripped:
            raise ValueError("Stored signal name cannot be empty.")
        self._replace_signal(signal_id, name=stripped)

    def set_signal_visible(self, signal_id: str, visible: bool) -> None:
        """Update the visible flag for one stored signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        visible : bool
            Whether the signal should be plotted.
        """
        self._replace_signal(signal_id, plot_data=True, visible=visible)

    def set_signal_color(self, signal_id: str, color: str) -> None:
        """Update the plot color for one stored signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        color : str
            Hex color string.
        """
        if not color:
            raise ValueError("Stored signal color cannot be empty.")
        self._replace_signal(signal_id, color=color)

    def remove_signals(self, signal_ids: list[str]) -> None:
        """Remove selected stored signals.

        Parameters
        ----------
        signal_ids : list[str]
            Identifiers of signals to remove.
        """
        if not signal_ids:
            return

        ids = set(signal_ids)
        kept = [signal for signal in self._stored_signals if signal.id not in ids]
        if len(kept) == len(self._stored_signals):
            return
        self._stored_signals = kept
        self._emit_changed(plot_data=True)

    def import_file(self, path: Path) -> list[StoredSignal]:
        """Import one CSV or TSV file into the store.

        Parameters
        ----------
        path : pathlib.Path
            File to import.

        Returns
        -------
        list[StoredSignal]
            Signals created from the file.

        Raises
        ------
        ValueError
            If the file does not contain a valid `time` column and numeric value columns.
        """
        frame = self._read_signals_table(path)
        imported = self._signal_from_frame(frame, path)
        self._stored_signals.extend(imported)
        self._emit_changed(plot_data=True)
        return imported

    def pin_signal(
        self,
        origin: str,
        name: str,
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        color: str,
        source_label: str,
    ) -> StoredSignal:
        """Pin a live signal's current values as a persistent stored signal.

        Re-pinning the same `origin` updates that entry's data in place rather than
        creating a duplicate, so repeatedly pinning e.g. the same label id just
        refreshes its snapshot (useful after repainting the label mask).

        Parameters
        ----------
        origin : str
            Identifier of the live signal being pinned (e.g. `"label-3"`,
            `"mouse-12-34-56"`). Matches the corresponding `LiveSignal.id` so the
            plotter can hide the pinned copy while that live signal is active.
        name : str
            Display name for the pinned signal.
        x : numpy.ndarray
            Time values.
        y : numpy.ndarray
            Signal values.
        color : str
            Hex color used for plotting.
        source_label : str
            Human-readable description of where this was pinned from.

        Returns
        -------
        StoredSignal
            The created or updated pinned signal.
        """
        x = np.asarray(x, dtype=float).copy()
        y = np.asarray(y, dtype=float).copy()

        for index, signal in enumerate(self._stored_signals):
            if signal.pin_origin == origin:
                updated = replace(signal, x=x, y=y)
                self._stored_signals[index] = updated
                self._emit_changed(plot_data=True)
                return updated

        signal_id = f"pinned-{self._id_counter}"
        self._id_counter += 1
        pinned = StoredSignal(
            id=signal_id,
            name=name,
            x=x,
            y=y,
            visible=True,
            color=color,
            source_label=source_label,
            file_path=None,
            original_column_name=None,
            pin_origin=origin,
        )
        self._stored_signals.append(pinned)
        self._emit_changed(plot_data=True)
        return pinned

    def _read_signals_table(self, path: Path) -> pd.DataFrame:
        """Read one CSV or TSV signals table from disk."""
        if self._is_bids_physio_path(path):
            return load_physio(path)

        _SEP: dict[str, str] = {".csv": ",", ".tsv": "\t"}
        sep = _SEP.get(self._table_suffix(path))
        return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")

    def _is_bids_physio_path(self, path: Path) -> bool:
        """Return whether `path` looks like a BIDS physio table."""
        if self._table_suffix(path) != ".tsv":
            return False
        name = path.name.removesuffix(".gz")
        return name.endswith("_physio.tsv")

    def _table_suffix(self, path: Path) -> str:
        """Return the logical table suffix, ignoring a trailing `.gz`."""
        if path.suffix.lower() == ".gz" and len(path.suffixes) >= 2:
            return path.suffixes[-2].lower()
        return path.suffix.lower()

    def _signal_from_frame(self, frame: pd.DataFrame, path: Path) -> list[StoredSignal]:
        """Convert a dataframe into stored signal entries."""
        if "time" not in frame.columns:
            raise ValueError("Imported file must contain a 'time' column.")

        value_columns = [column for column in frame.columns if column != "time"]
        if not value_columns:
            raise ValueError(
                "Imported file must contain at least one value column besides 'time'."
            )

        non_numeric = [
            column
            for column in value_columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            columns = ", ".join(repr(column) for column in non_numeric)
            raise ValueError(f"Imported value columns must be numeric: {columns}.")

        time_values = frame["time"].to_numpy(copy=True)
        imported = []

        for offset, column in enumerate(value_columns):
            color = _STORED_SIGNAL_COLORS[
                (self._id_counter + offset) % len(_STORED_SIGNAL_COLORS)
            ]
            signal_id = f"imported-{self._id_counter + offset}"
            imported.append(
                StoredSignal(
                    id=signal_id,
                    name=str(column),
                    x=time_values.copy(),
                    y=frame[column].to_numpy(dtype=float, copy=True),
                    visible=True,
                    color=color,
                    source_label=path.name,
                    file_path=path,
                    original_column_name=str(column),
                )
            )

        self._id_counter += len(value_columns)
        return imported

    # -- Live signal management ------------------------------------------------

    def live_signals(self) -> list[LiveSignal]:
        """Return all live signals in insertion order.

        Returns
        -------
        list[LiveSignal]
            All registered live signals.
        """
        return list(self._live_signals.values())

    def visible_live_signals(self) -> list[LiveSignal]:
        """Return only live signals marked as visible.

        Returns
        -------
        list[LiveSignal]
            Visible live signals.
        """
        return [s for s in self._live_signals.values() if s.visible]

    def get_live_signal(self, signal_id: str) -> LiveSignal | None:
        """Look up a single live signal by ID.

        Parameters
        ----------
        signal_id : str
            Signal identifier.

        Returns
        -------
        LiveSignal | None
            The signal, or `None` if not found.
        """
        return self._live_signals.get(signal_id)

    def register_live_signals(self, signals: list[LiveSignal]) -> None:
        """Register live signals, preserving user overrides for existing IDs.

        New IDs are added with the supplied defaults. Existing IDs keep their current
        name, color, and visibility.  IDs not present in `signals` are removed.

        Parameters
        ----------
        signals : list[LiveSignal]
            Live signals to register.
        """
        new_by_id = {s.id: s for s in signals}
        merged: dict[str, LiveSignal] = {}
        for sid, new in new_by_id.items():
            old = self._live_signals.get(sid)
            if old is not None:
                # Preserve user overrides, but update source metadata.
                merged[sid] = replace(
                    new,
                    name=old.name,
                    color=old.color,
                    visible=old.visible,
                )
            else:
                merged[sid] = new

        if merged != self._live_signals:
            self._live_signals = merged
            self._emit_changed(plot_data=True)

    def clear_live_signals(self) -> None:
        """Remove all live signals."""
        if not self._live_signals:
            return
        self._live_signals.clear()
        self._emit_changed(plot_data=True)

    def rename_live_signal(self, signal_id: str, new_name: str) -> None:
        """Rename one live signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        new_name : str
            New display name.

        Raises
        ------
        ValueError
            If the name is empty or the signal does not exist.
        """
        stripped = new_name.strip()
        if not stripped:
            raise ValueError("Live signal name cannot be empty.")
        self._replace_live(signal_id, name=stripped)

    def set_live_signal_visible(self, signal_id: str, visible: bool) -> None:
        """Update the visible flag for one live signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        visible : bool
            Whether the signal should be plotted.
        """
        self._replace_live(signal_id, plot_data=True, visible=visible)

    def set_live_signal_color(self, signal_id: str, color: str) -> None:
        """Update the plot color for one live signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        color : str
            Hex color string.
        """
        if not color:
            raise ValueError("Live signal color cannot be empty.")
        self._replace_live(signal_id, color=color)

    # -- Private helpers ------------------------------------------------------

    def _replace_live(
        self,
        signal_id: str,
        *,
        plot_data: bool = False,
        **changes,
    ) -> None:
        """Replace one live signal while preserving order."""
        signal = self._live_signals.get(signal_id)
        if signal is None:
            raise ValueError(f"Unknown live signal id: {signal_id!r}.")
        self._live_signals[signal_id] = replace(signal, **changes)
        self._emit_changed(plot_data=plot_data)

    def _replace_signal(
        self,
        signal_id: str,
        *,
        plot_data: bool = False,
        **changes,
    ) -> None:
        """Replace one stored signal while preserving order."""
        for index, signal in enumerate(self._stored_signals):
            if signal.id != signal_id:
                continue
            self._stored_signals[index] = replace(signal, **changes)
            self._emit_changed(plot_data=plot_data)
            return

        raise ValueError(f"Unknown stored signal id: {signal_id!r}.")
