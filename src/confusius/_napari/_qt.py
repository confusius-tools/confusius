"""Shared Qt helpers for internal napari panels."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from confusius._utils.colors import RED

SEGMENT_BUTTON_STYLE = f"""
QPushButton {{
    border-radius: 0;
}}
QPushButton:checked {{
    background: {RED};
    color: white;
    font-weight: bold;
}}
"""
"""QSS shared by every button of a segmented control; the end buttons add rounded
outer corners on top of it."""


def make_segmented_buttons(
    labels: list[str], parent: QWidget
) -> tuple[QHBoxLayout, list[QPushButton]]:
    """Build a row of mutually exclusive checkable buttons, the first one checked.

    This is the page switcher used at the top of the Registration and Atlas panels.

    Parameters
    ----------
    labels : list[str]
        Button captions, left to right. At least two.
    parent : qtpy.QtWidgets.QWidget
        Owner of the exclusive `QButtonGroup` that keeps one button checked.

    Returns
    -------
    row : qtpy.QtWidgets.QHBoxLayout
        Layout holding the buttons with no spacing, ready to add to a panel.
    buttons : list[qtpy.QtWidgets.QPushButton]
        The buttons, in `labels` order; connect to their `toggled` signals.
    """
    group = QButtonGroup(parent)
    row = QHBoxLayout()
    row.setSpacing(0)
    buttons: list[QPushButton] = []
    last = len(labels) - 1
    for i, label in enumerate(labels):
        button = QPushButton(label)
        button.setCheckable(True)
        button.setChecked(i == 0)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        style = SEGMENT_BUTTON_STYLE
        if i == 0:
            style += "QPushButton { border-top-left-radius: 3px; border-bottom-left-radius: 3px; }"
        if i == last:
            style += "QPushButton { border-top-right-radius: 3px; border-bottom-right-radius: 3px; }"
        button.setStyleSheet(style)
        group.addButton(button)
        row.addWidget(button)
        buttons.append(button)
    return row, buttons


class _NoScrollWheelFilter(QObject):
    """Forward wheel events to the enclosing scroll area instead of the control.

    Without this, scrolling the sidebar with the cursor over a combo box or spin
    box changes that control's value instead of scrolling the sidebar. Installed
    on every such control so the whole scroll area behaves like one continuous
    surface.
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
