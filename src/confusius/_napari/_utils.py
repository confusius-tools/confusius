"""Shared utilities for the ConfUSIus napari plugin."""

import datetime as dt
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import pandas as pd
from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer as _QSvgRenderer
from qtpy.QtWidgets import QApplication, QFileDialog, QToolButton, QWidget

ExportSeries: TypeAlias = tuple[str, np.ndarray, np.ndarray]
"""Type alias for a plotted series, as a tuple of `(label, x_values, y_values)`."""

_ASSETS_DIR = Path(__file__).parent / "assets"
"""Directory containing SVG icon assets for export buttons."""


def get_napari_colors(theme_name: str) -> dict:
    """Extract plot-relevant colors from a napari theme.

    Falls back to sensible defaults if the napari theme API is unavailable.

    Parameters
    ----------
    theme_name : str
        Name of the napari theme (e.g. `"dark"` or `"light"`).

    Returns
    -------
    dict
        Dictionary with the following keys:

        `"bg"`
            Background hex color string.
        `"fg"`
            Foreground (text) hex color string.
        `"accent"`
            Accent hex color string for plot lines.
        `"cursor"`
            Hex color string for the time cursor line.
        `"is_dark"`
            `True` when the background luminance is below 50 %.
    """
    try:
        from napari.utils.theme import get_theme

        t = get_theme(theme_name)

        def _h(c) -> str:
            h = c.as_hex()
            # as_hex() may include an alpha channel (#rrggbbaa); strip it.
            return h[:7]

        bg = _h(t.background)
        fg = _h(t.text)
    except Exception:  # noqa: BLE001
        _dark = theme_name != "light"
        bg = "#262930" if _dark else "#f0f0f0"
        fg = "#bbbbbb" if _dark else "#2c2c2c"

    r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
    is_dark = (0.299 * r + 0.587 * g + 0.114 * b) < 128

    return {
        "bg": bg,
        "fg": fg,
        "accent": "#ffd33d" if is_dark else "#c49a0a",
        "cursor": "#ff6b6b" if is_dark else "#cc2200",
        "is_dark": is_dark,
    }


def recolor_toolbar_icons(toolbar: QWidget, color: str) -> None:
    """Tint every action icon in *toolbar* with *color*.

    Matplotlib toolbar icons are black PNGs. Using `CompositionMode_SourceIn` keeps the
    alpha mask of each icon but floods all opaque pixels with *color*, producing
    theme-aware icons while preserving enabled and disabled states.

    Parameters
    ----------
    toolbar : QWidget
        The matplotlib `NavigationToolbar2QT` instance whose icons to recolor.
    color : str
        CSS hex color string (e.g. `"#ffffff"`).
    """
    qcolor = QColor(color)
    disabled_color = QColor(color)
    disabled_color.setAlphaF(0.4)

    icon_cache = cast(
        dict[object, QIcon] | None, toolbar.property("_confusius_original_icons")
    )
    if icon_cache is None:
        icon_cache = {}
        toolbar.setProperty("_confusius_original_icons", icon_cache)

    for action in toolbar.actions():
        current_icon = action.icon()
        if current_icon.isNull():
            continue

        icon_cache.setdefault(action, QIcon(current_icon))
        base_icon = icon_cache[action]
        sizes = base_icon.availableSizes()
        size = sizes[0] if sizes else QSize(24, 24)

        tinted_icon = QIcon()
        for mode, tint in [
            (QIcon.Mode.Normal, qcolor),
            (QIcon.Mode.Active, qcolor),
            (QIcon.Mode.Selected, qcolor),
            (QIcon.Mode.Disabled, disabled_color),
        ]:
            for state in [QIcon.State.Off, QIcon.State.On]:
                src = base_icon.pixmap(size, mode, state)
                if src.isNull():
                    continue
                tinted_icon.addPixmap(_tint_pixmap(src, tint), mode, state)

        action.setIcon(tinted_icon)


def style_plot_toolbar(toolbar: QWidget, colors: dict) -> None:
    """Apply consistent theme styling to a matplotlib navigation toolbar.

    Parameters
    ----------
    toolbar : QWidget
        Matplotlib navigation toolbar to style.
    colors : dict
        Napari theme color mapping produced by
        [`get_napari_colors`][confusius._napari._utils.get_napari_colors].
    """
    toolbar.setStyleSheet(
        " ".join(
            [
                f"background: {colors['bg']};",
                "border: none;",
                "QToolBar {",
                f"background: {colors['bg']};",
                "border: none;",
                "spacing: 0px;",
                "}",
            ]
        )
    )
    recolor_toolbar_icons(toolbar, colors["fg"])


def _tint_pixmap(src: QPixmap, color: QColor) -> QPixmap:
    """Return a pixmap tinted with the requested color.

    Parameters
    ----------
    src : QPixmap
        Source pixmap.
    color : QColor
        Tint color to apply while preserving alpha.

    Returns
    -------
    QPixmap
        Tinted pixmap.
    """
    dst = QPixmap(src.size())
    dst.setDevicePixelRatio(src.devicePixelRatio())
    dst.fill(Qt.GlobalColor.transparent)
    painter = QPainter(dst)
    painter.drawPixmap(0, 0, src)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(dst.rect(), color)
    painter.end()
    return dst


def rgba_css(hex_color: str, alpha: float) -> str:
    """Convert a hex color to an `rgba(...)` CSS string.

    Parameters
    ----------
    hex_color : str
        CSS hex color string.
    alpha : float
        Alpha channel value between 0 and 1.

    Returns
    -------
    str
        CSS `rgba(...)` string.
    """
    qcolor = QColor(hex_color)
    return f"rgba({qcolor.red()}, {qcolor.green()}, {qcolor.blue()}, {alpha:.3f})"


def make_lucide_icon(name: str, color: str, size: int = 16) -> QIcon:
    """Render a Lucide SVG icon tinted with the requested color.

    Parameters
    ----------
    name : str
        Stem of the SVG asset inside `src/confusius/_napari/assets`.
    color : str
        CSS hex color string used to replace `currentColor` in the SVG.
    size : int, default: 16
        Target icon size in logical pixels.

    Returns
    -------
    QIcon
        Rendered icon, or an empty icon if the asset file is missing.
    """
    svg_path = _ASSETS_DIR / f"{name}.svg"
    if not svg_path.exists():
        return QIcon()

    svg_bytes = svg_path.read_bytes().replace(b"currentColor", color.encode())

    app = cast(QApplication | None, QApplication.instance())
    screen = app.primaryScreen() if app is not None else None
    dpr = screen.devicePixelRatio() if screen is not None else 1.0
    px = round(size * dpr)

    renderer = _QSvgRenderer(svg_bytes)
    image = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()

    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(dpr)
    return QIcon(pixmap)


def create_export_button(
    toolbar: QWidget, object_name: str, on_export, text: str = "Export"
) -> QToolButton:
    """Create an export button for a matplotlib toolbar.

    Parameters
    ----------
    toolbar : QWidget
        Parent toolbar widget.
    object_name : str
        Qt object name used for styling and inspection.
    on_export : callable
        Callback triggered when the button is clicked.
    text : str, default: "Export"
        Button label shown next to the export icon.

    Returns
    -------
    QToolButton
        Configured export button.
    """
    button = QToolButton(toolbar)
    button.setObjectName(object_name)
    button.setText(text)
    button.setToolTip("Export the plotted data.")
    button.setAutoRaise(True)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.clicked.connect(on_export)

    return button


def style_export_button(button: QToolButton, colors: dict) -> None:
    """Apply themed styling to an export button.

    Parameters
    ----------
    button : QToolButton
        Export button to style.
    colors : dict
        Napari theme color mapping produced by
        [`get_napari_colors`][confusius._napari._utils.get_napari_colors].
    """
    button.setIcon(make_lucide_icon("download", colors["accent"], size=22))
    button.setIconSize(QSize(22, 22))
    button.setToolButtonStyle(
        Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        if button.text()
        else Qt.ToolButtonStyle.ToolButtonIconOnly
    )
    button.setStyleSheet(
        " ".join(
            [
                "QToolButton {",
                "background: transparent;",
                "border: none;",
                f"color: {colors['accent']};",
                "padding: 3px 5px;",
                "margin-left: 6px;",
                "border-radius: 4px;",
                "font-weight: 600;",
                "}",
                "QToolButton:disabled {",
                "background: transparent;",
                f"color: {rgba_css(colors['fg'], 0.4)};",
                "}",
            ]
        )
    )


def prepare_export_series(series: list[ExportSeries]) -> list[ExportSeries]:
    """Normalize plotted series into copied 1D NumPy arrays.

    Parameters
    ----------
    series : list[ExportSeries]
        Plotted series as `(label, x_values, y_values)` tuples.

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
    for label, x_values, y_values in series:
        x_array = np.ravel(np.asarray(x_values)).copy()
        y_array = np.ravel(np.asarray(y_values)).copy()
        if len(x_array) != len(y_array):
            raise ValueError(
                f"Cannot export {label!r}: time and value arrays have different lengths."
            )
        prepared.append((label, x_array, y_array))
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
        Plotted series as `(label, x_values, y_values)` tuples.
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
    labels = _deduplicate_export_labels([label for label, _, _ in prepared_series])

    row_lookup: dict[tuple[object, int], int] = {}
    row_times: list[object] = []
    column_values = [dict() for _ in prepared_series]

    for column_index, (_, x_array, y_array) in enumerate(prepared_series):
        occurrence_counts: dict[object, int] = {}
        for x_value, y_value in zip(x_array, y_array, strict=True):
            key = _export_key(x_value)
            occurrence = occurrence_counts.get(key, 0)
            occurrence_counts[key] = occurrence + 1
            row_key = (key, occurrence)

            row_index = row_lookup.get(row_key)
            if row_index is None:
                row_index = len(row_times)
                row_lookup[row_key] = row_index
                row_times.append(x_value)

            column_values[column_index][row_index] = y_value

    rows = []
    for row_index, time_value in enumerate(row_times):
        row = {time_header: format_export_value(time_value)}
        for label, values in zip(labels, column_values, strict=True):
            row[label] = format_export_value(values.get(row_index))
        rows.append(row)

    pd.DataFrame(rows, columns=[time_header, *labels]).to_csv(
        path, sep=delimiter, index=False
    )


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


def _export_key(value: object) -> object:
    """Return a stable hashable key for export row alignment."""
    if isinstance(value, np.datetime64):
        return ("datetime64", np.datetime_as_string(value, unit="ns"))
    if isinstance(value, np.generic):
        return value.item()
    return value
