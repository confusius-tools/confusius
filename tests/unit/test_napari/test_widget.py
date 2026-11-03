"""Unit tests for the main napari widget."""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWIDGETSIZE_MAX, QApplication, QPushButton

from confusius._napari._widget import ConfUSIusWidget


def test_events_section_precedes_signals(make_napari_viewer, qtbot):
    widget = ConfUSIusWidget(make_napari_viewer())
    qtbot.addWidget(widget)
    headers = [
        button.text()
        for button in widget.findChildren(QPushButton)
        if button.objectName() == "accordion_header"
    ]
    assert headers.index("Events") + 1 == headers.index("Signals")


def test_dock_fills_available_height_under_napari_0_9(make_napari_viewer):
    """The main widget should not leave a blank band below the sidebar."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    dock = viewer.window.add_dock_widget(widget, area="right")

    QTimer.singleShot(0, lambda: None)
    QApplication.processEvents()

    assert dock.maximumHeight() == QWIDGETSIZE_MAX
