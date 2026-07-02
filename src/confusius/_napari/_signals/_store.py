"""Shared store for imported napari signals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from qtpy.QtCore import QObject, Signal

_IMPORTED_SIGNAL_COLORS = [
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


@dataclass(frozen=True, slots=True)
class ImportedSignal:
    """Imported signal stored for overlay in the plotter.

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
        Human-readable source description, typically the file name.
    file_path : pathlib.Path
        Original imported file.
    original_column_name : str
        Column name from the imported file.
    """

    id: str
    name: str
    x: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]
    visible: bool
    color: str
    source_label: str
    file_path: Path
    original_column_name: str


@dataclass(frozen=True, slots=True)
class LiveSignal:
    """Live signal backed by a napari layer.

    Unlike `ImportedSignal`, a live signal does not own its data: the plotter extracts
    it from layers on each update. The store tracks only presentation metadata (name,
    color, visibility) so the user can customise the plot.

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
    """Store imported and live signals shared between the panel and plotter.

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
        self._imported_signals: list[ImportedSignal] = []
        self._live_signals: dict[str, LiveSignal] = {}
        self._id_counter: int = 0

    def imported_signals(self) -> list[ImportedSignal]:
        """Return all imported signals.

        Returns
        -------
        list[ImportedSignal]
            Imported signals in insertion order.
        """
        return list(self._imported_signals)

    def visible_imported_signals(self) -> list[ImportedSignal]:
        """Return only imported signals marked as visible.

        Returns
        -------
        list[ImportedSignal]
            Visible imported signals.
        """
        return [signal for signal in self._imported_signals if signal.visible]

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
        """Remove all imported signals."""
        if not self._imported_signals:
            return
        self._imported_signals.clear()
        self._emit_changed(plot_data=True)

    def rename_signal(self, signal_id: str, new_name: str) -> None:
        """Rename one imported signal.

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
            raise ValueError("Imported signal name cannot be empty.")
        self._replace_signal(signal_id, name=stripped)

    def set_signal_visible(self, signal_id: str, visible: bool) -> None:
        """Update the visible flag for one imported signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        visible : bool
            Whether the signal should be plotted.
        """
        self._replace_signal(signal_id, plot_data=True, visible=visible)

    def set_signal_color(self, signal_id: str, color: str) -> None:
        """Update the plot color for one imported signal.

        Parameters
        ----------
        signal_id : str
            Signal identifier.
        color : str
            Hex color string.
        """
        if not color:
            raise ValueError("Imported signal color cannot be empty.")
        self._replace_signal(signal_id, color=color)

    def remove_signals(self, signal_ids: list[str]) -> None:
        """Remove selected imported signals.

        Parameters
        ----------
        signal_ids : list[str]
            Identifiers of signals to remove.
        """
        if not signal_ids:
            return

        ids = set(signal_ids)
        kept = [signal for signal in self._imported_signals if signal.id not in ids]
        if len(kept) == len(self._imported_signals):
            return
        self._imported_signals = kept
        self._emit_changed(plot_data=True)

    def import_file(self, path: Path) -> list[ImportedSignal]:
        """Import one CSV or TSV file into the store.

        Parameters
        ----------
        path : pathlib.Path
            File to import.

        Returns
        -------
        list[ImportedSignal]
            Imported signal created from the file.

        Raises
        ------
        ValueError
            If the file does not contain a valid `time` column and numeric value columns.
        """
        frame = self._read_signals_table(path)
        imported = self._signal_from_frame(frame, path)
        self._imported_signals.extend(imported)
        self._emit_changed(plot_data=True)
        return imported

    def _read_signals_table(self, path: Path) -> pd.DataFrame:
        """Read one CSV or TSV signals table from disk."""
        _SEP: dict[str, str] = {".csv": ",", ".tsv": "\t"}
        sep = _SEP.get(path.suffix.lower())
        return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")

    def _signal_from_frame(
        self, frame: pd.DataFrame, path: Path
    ) -> list[ImportedSignal]:
        """Convert a dataframe into imported signal entries."""
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
            color = _IMPORTED_SIGNAL_COLORS[
                (self._id_counter + offset) % len(_IMPORTED_SIGNAL_COLORS)
            ]
            signal_id = f"imported-{self._id_counter + offset}"
            imported.append(
                ImportedSignal(
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
        """Replace one imported signal while preserving order."""
        for index, signal in enumerate(self._imported_signals):
            if signal.id != signal_id:
                continue
            self._imported_signals[index] = replace(signal, **changes)
            self._emit_changed(plot_data=plot_data)
            return

        raise ValueError(f"Unknown imported signal id: {signal_id!r}.")
