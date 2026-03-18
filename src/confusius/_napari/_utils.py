"""Shared utilities for the ConfUSIus napari plugin."""

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import QWidget


def napari_colors(theme_name: str) -> dict:
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

    Matplotlib toolbar icons are black PNGs. Using `CompositionMode_SourceIn`
    keeps the alpha mask of each icon but floods all opaque pixels with *color*,
    producing white icons on dark themes and black icons on light themes.

    Parameters
    ----------
    toolbar : QWidget
        The matplotlib `NavigationToolbar2QT` instance whose icons to recolor.
    color : str
        CSS hex color string (e.g. `"#ffffff"`).
    """
    qcolor = QColor(color)
    for action in toolbar.actions():
        icon = action.icon()
        if icon.isNull():
            continue
        sizes = icon.availableSizes()
        sz = sizes[0] if sizes else QSize(24, 24)
        src = icon.pixmap(sz)
        dst = QPixmap(src.size())
        dst.setDevicePixelRatio(src.devicePixelRatio())
        dst.fill(Qt.GlobalColor.transparent)
        p = QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(dst.rect(), qcolor)
        p.end()
        action.setIcon(QIcon(dst))


def show_napari_error(message: str) -> None:
    """Display *message* as a napari error notification.

    Silently swallows any exception so callers never crash on a missing or
    broken napari notification API.

    Parameters
    ----------
    message : str
        Error message to display.
    """
    try:
        from napari.utils.notifications import show_error

        show_error(message)
    except Exception:  # noqa: BLE001
        pass
