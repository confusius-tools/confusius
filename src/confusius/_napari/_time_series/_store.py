"""Shared store for imported napari time series."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from qtpy.QtCore import QObject, Signal

_IMPORTED_SERIES_COLORS = [
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
class ImportedSeries:
    """Imported time series stored for overlay in the plotter.

    Attributes
    ----------
    id : str
        Stable series identifier.
    name : str
        Display name used in legends and exports.
    x : numpy.ndarray
        Time values.
    y : numpy.ndarray
        Series values.
    visible : bool
        Whether the series should be plotted.
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


class TimeSeriesStore(QObject):
    """Store imported time series shared between the panel and plotter.

    Parameters
    ----------
    parent : QObject | None, optional
        Optional Qt parent.
    """

    changed = Signal()
    plot_data_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._imported_series: list[ImportedSeries] = []
        self._id_counter: int = 0

    def imported_series(self) -> list[ImportedSeries]:
        """Return all imported series.

        Returns
        -------
        list[ImportedSeries]
            Imported series in insertion order.
        """
        return list(self._imported_series)

    def visible_imported_series(self) -> list[ImportedSeries]:
        """Return only imported series marked as visible.

        Returns
        -------
        list[ImportedSeries]
            Visible imported series.
        """
        return [series for series in self._imported_series if series.visible]

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
        """Remove all imported series."""
        if not self._imported_series:
            return
        self._imported_series.clear()
        self._emit_changed(plot_data=True)

    def rename_series(self, series_id: str, new_name: str) -> None:
        """Rename one imported series.

        Parameters
        ----------
        series_id : str
            Series identifier.
        new_name : str
            New display name.

        Raises
        ------
        ValueError
            If the name is empty or the series does not exist.
        """
        stripped = new_name.strip()
        if not stripped:
            raise ValueError("Imported series name cannot be empty.")
        self._replace_series(series_id, name=stripped)

    def set_series_visible(self, series_id: str, visible: bool) -> None:
        """Update the visible flag for one imported series.

        Parameters
        ----------
        series_id : str
            Series identifier.
        visible : bool
            Whether the series should be plotted.
        """
        self._replace_series(series_id, plot_data=True, visible=visible)

    def set_series_color(self, series_id: str, color: str) -> None:
        """Update the plot color for one imported series.

        Parameters
        ----------
        series_id : str
            Series identifier.
        color : str
            Hex color string.
        """
        if not color:
            raise ValueError("Imported series color cannot be empty.")
        self._replace_series(series_id, color=color)

    def remove_series(self, series_ids: list[str]) -> None:
        """Remove selected imported series.

        Parameters
        ----------
        series_ids : list[str]
            Identifiers of series to remove.
        """
        if not series_ids:
            return

        ids = set(series_ids)
        kept = [series for series in self._imported_series if series.id not in ids]
        if len(kept) == len(self._imported_series):
            return
        self._imported_series = kept
        self._emit_changed(plot_data=True)

    def import_file(self, path: Path) -> list[ImportedSeries]:
        """Import one CSV or TSV file into the store.

        Parameters
        ----------
        path : pathlib.Path
            File to import.

        Returns
        -------
        list[ImportedSeries]
            Imported series created from the file.

        Raises
        ------
        ValueError
            If the file does not contain a valid `time` column and numeric value columns.
        """
        frame = self._read_series_table(path)
        imported = self._series_from_frame(frame, path)
        self._imported_series.extend(imported)
        self._emit_changed(plot_data=True)
        return imported

    def _read_series_table(self, path: Path) -> pd.DataFrame:
        """Read one CSV or TSV time-series table from disk."""
        _SEP: dict[str, str] = {".csv": ",", ".tsv": "\t"}
        sep = _SEP.get(path.suffix.lower())
        return pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")

    def _series_from_frame(
        self, frame: pd.DataFrame, path: Path
    ) -> list[ImportedSeries]:
        """Convert a dataframe into imported series entries."""
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
            color = _IMPORTED_SERIES_COLORS[
                (self._id_counter + offset) % len(_IMPORTED_SERIES_COLORS)
            ]
            series_id = f"imported-{self._id_counter + offset}"
            imported.append(
                ImportedSeries(
                    id=series_id,
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

    def _replace_series(
        self,
        series_id: str,
        *,
        plot_data: bool = False,
        **changes,
    ) -> None:
        """Replace one imported series while preserving order."""
        for index, series in enumerate(self._imported_series):
            if series.id != series_id:
                continue
            self._imported_series[index] = replace(series, **changes)
            self._emit_changed(plot_data=plot_data)
            return

        raise ValueError(f"Unknown imported series id: {series_id!r}.")
