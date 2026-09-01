"""Shared Qt helpers for internal napari panels."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QMainWindow,
    QWidget,
)


class _NoScrollWheelFilter(QObject):
    """Forward wheel events to the enclosing scroll area instead of the control.

    Without this, scrolling the sidebar with the cursor over a combo box or spin
    box changes that control's value instead of scrolling the sidebar — installed
    on every such control so the whole scroll area behaves like one continuous
    surface. Unconditional (regardless of focus): these controls are also
    reachable via their up/down arrows or by typing, so trading away
    wheel-to-adjust entirely removes any risk of the sidebar scroll still being
    hijacked by a control that Qt considers focused.
    """

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # type: ignore
        """Redirect the watched control's wheel events to its scroll area."""
        if (
            event is not None
            and event.type() == QEvent.Type.Wheel
            and isinstance(watched, QWidget)
        ):
            ancestor = watched.parentWidget()
            while ancestor is not None and not isinstance(
                ancestor, QAbstractScrollArea
            ):
                ancestor = ancestor.parentWidget()
            if ancestor is not None:
                # QAbstractScrollArea only reacts to wheel events delivered to its
                # viewport, not to the scroll area widget itself.
                QApplication.sendEvent(ancestor.viewport(), event)
            return True
        return super().eventFilter(watched, event)


def install_no_scroll_wheel_filter(root: QWidget) -> None:
    """Stop combo/spin boxes under `root` from capturing sidebar scroll events.

    Recursively installs a shared `_NoScrollWheelFilter` on every `QComboBox` and
    `QAbstractSpinBox` descendant of `root` (the latter covers both `QSpinBox` and
    `QDoubleSpinBox`). Call once after a panel's widgets are constructed — new
    descendants added later are not covered.

    Parameters
    ----------
    root : QWidget
        Widget to search for combo/spin box descendants.
    """
    wheel_filter = _NoScrollWheelFilter(root)
    for widget in root.findChildren((QComboBox, QAbstractSpinBox)):
        widget.installEventFilter(wheel_filter)


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
