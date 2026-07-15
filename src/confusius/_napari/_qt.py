"""Shared Qt helpers for internal napari panels."""

from __future__ import annotations

from qtpy.QtWidgets import QMainWindow, QWidget


def find_main_window(widget: QWidget) -> QMainWindow | None:
    """Return the ancestor `QMainWindow` for a widget, if present.

    Parameters
    ----------
    widget : QWidget
        Starting widget to search from.

    Returns
    -------
    QMainWindow or None
        The containing main window, or `None` if no ancestor main window is
        found or the Qt object was already deleted.
    """
    try:
        parent = widget.parent()
    except RuntimeError:
        return None
    while parent is not None:
        if isinstance(parent, QMainWindow):
            return parent
        try:
            parent = parent.parent()
        except RuntimeError:
            return None
    return None
