"""Data structures and serialization for CSV/TSV time series export."""

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from qtpy.QtWidgets import QFileDialog, QWidget


@dataclass(frozen=True, slots=True)
class PlotSeries:
    """A time series ready for plotting with color.

    Attributes
    ----------
    label : str
        Series name for legend.
    x_values : numpy.ndarray
        X-axis values (typically time).
    y_values : numpy.ndarray
        Y-axis values (intensity/z-score).
    color : str | None
        Hex color string, or None to use default.
    """

    label: str
    x_values: npt.NDArray[np.floating]
    y_values: npt.NDArray[np.floating]
    color: str | None = None


@dataclass(frozen=True, slots=True)
class ExportSeries:
    """A time series ready for CSV/TSV export (no color).

    Unlike `PlotSeries`, this excludes color information since exports are
    data-only.

    Attributes
    ----------
    label : str
        Series name for the header.
    x_values : numpy.ndarray
        X-axis values (typically time).
    y_values : numpy.ndarray
        Y-axis values (intensity/z-score).
    """

    label: str
    x_values: npt.NDArray[np.floating]
    y_values: npt.NDArray[np.floating]


def prepare_export_series(series: list[ExportSeries]) -> list[ExportSeries]:
    """Normalize plotted series into copied 1D NumPy arrays.

    Parameters
    ----------
    series : list[ExportSeries]
        Plotted series to normalize.

    Returns
    -------
    list[ExportSeries]
        Series with copied, flattened NumPy arrays.

    Raises
    ------
    ValueError
        If any `x`/`y` pair has mismatched lengths.
    """
    prepared = []
    for s in series:
        x_array = np.ravel(np.asarray(s.x_values)).copy()
        y_array = np.ravel(np.asarray(s.y_values)).copy()
        if len(x_array) != len(y_array):
            raise ValueError(
                f"Cannot export {s.label!r}: time and value arrays have different lengths."
            )
        prepared.append(ExportSeries(s.label, x_array, y_array))
    return prepared


def prompt_delimited_export_path(
    parent: QWidget, title: str, default_filename: str
) -> tuple[Path, str] | None:
    """Open a save dialog for CSV/TSV export.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the save dialog.
    title : str
        Dialog title.
    default_filename : str
        Default file path shown when the dialog opens.

    Returns
    -------
    tuple[pathlib.Path, str] | None
        Selected path and delimiter, or `None` if the dialog was cancelled.
    """
    path_str, selected_filter = QFileDialog.getSaveFileName(
        parent,
        title,
        default_filename,
        "TSV (*.tsv);;CSV (*.csv);;All files (*)",
    )
    if not path_str:
        return None
    return resolve_delimited_export_path(path_str, selected_filter)


def resolve_delimited_export_path(
    path_str: str, selected_filter: str
) -> tuple[Path, str]:
    """Resolve output path and delimiter from a save dialog selection.

    Parameters
    ----------
    path_str : str
        Path returned by the save dialog.
    selected_filter : str
        Selected name filter returned by the save dialog.

    Returns
    -------
    path : pathlib.Path
        Output path with an inferred suffix when needed.
    delimiter : str
        Delimiter corresponding to the chosen file format.
    """
    path = Path(path_str)
    filter_lower = selected_filter.lower()
    suffix = path.suffix.lower()

    if "csv" in filter_lower or suffix == ".csv":
        if not suffix:
            path = path.with_suffix(".csv")
        return path, ","

    if not suffix:
        path = path.with_suffix(".tsv")
    return path, "\t"


def format_export_value(value: object) -> str:
    """Format a scalar value for CSV or TSV export.

    Parameters
    ----------
    value : object
        Scalar value from the export table.

    Returns
    -------
    str
        String representation suitable for delimited text output.
    """
    if value is None:
        return ""
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value)
    if isinstance(value, np.generic):
        return format_export_value(value.item())
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.16g}"
    return str(value)


def write_delimited_series(
    path: Path, series: list[ExportSeries], delimiter: str, time_header: str = "time"
) -> None:
    """Write one or more plotted series to a delimited text file.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    series : list[ExportSeries]
        Plotted series to write.
    delimiter : str
        Delimiter to use when writing the file.
    time_header : str, default: "time"
        Header label for the first column.

    Raises
    ------
    ValueError
        If no series are provided or any `x`/`y` pair has mismatched lengths.
    """
    if not series:
        raise ValueError("No plotted data available to export.")

    prepared_series = prepare_export_series(series)
    labels = _deduplicate_export_labels([s.label for s in prepared_series])

    # Build one DataFrame per series, then outer-merge on the time column so series with
    # different time axes are aligned correctly.
    merged = pd.DataFrame(
        {
            time_header: prepared_series[0].x_values,
            labels[0]: prepared_series[0].y_values,
        }
    )
    for s, label in zip(prepared_series[1:], labels[1:], strict=True):
        right = pd.DataFrame({time_header: s.x_values, label: s.y_values})
        merged = merged.merge(right, on=time_header, how="outer")

    merged = merged.sort_values(time_header, kind="stable").reset_index(drop=True)

    # Format all values for text output (NaN → empty string, float precision, etc.).
    for col in merged.columns:
        merged[col] = merged[col].map(format_export_value)

    merged.to_csv(path, sep=delimiter, index=False)


def _deduplicate_export_labels(labels: list[str]) -> list[str]:
    """Return unique export column labels while preserving order."""
    counts: dict[str, int] = {}
    unique_labels = []
    for label in labels:
        base_label = label or "series"
        count = counts.get(base_label, 0)
        unique_labels.append(base_label if count == 0 else f"{base_label}_{count + 1}")
        counts[base_label] = count + 1
    return unique_labels
