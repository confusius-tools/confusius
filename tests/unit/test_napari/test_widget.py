"""Unit tests for the main napari widget."""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWIDGETSIZE_MAX, QApplication

from confusius._napari._transform._panel import TransformPanel
from confusius._napari._widget import ConfUSIusWidget


def test_accordion_has_a_transform_section(make_napari_viewer):
    """The transform panel sits in the accordion, right after registration."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)

    titles = list(widget._accordion_panels)

    assert titles.index("Transform") == titles.index("Registration") + 1
    assert isinstance(widget._accordion_panels["Transform"], TransformPanel)


def test_dock_fills_available_height_under_napari_0_9(make_napari_viewer):
    """The main widget should not leave a blank band below the sidebar."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    dock = viewer.window.add_dock_widget(widget, area="right")

    QTimer.singleShot(0, lambda: None)
    QApplication.processEvents()

    assert dock.maximumHeight() == QWIDGETSIZE_MAX
