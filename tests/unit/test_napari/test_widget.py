"""Unit tests for the main napari widget."""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWIDGETSIZE_MAX, QApplication, QGroupBox

from confusius._napari._data._load_panel import DataPanel
from confusius._napari._video._video_panel import VideoPanel
from confusius._napari._widget import ConfUSIusWidget


def test_video_loading_is_part_of_data_io(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)

    labels = [button.text() for button, _icon in widget._accordion_btns]
    assert labels == [
        "Data I/O",
        "Signals",
        "Registration",
        "Events",
        "Quality Control",
    ]
    video_panel = widget.findChild(VideoPanel)
    assert video_panel is not None
    data_io = widget._accordion_panels["Data I/O"]
    assert data_io.isAncestorOf(video_panel)
    assert widget.findChild(DataPanel).video_panel is video_panel
    assert [group.title() for group in data_io.findChildren(QGroupBox)] == [
        "Load Data",
        "Layer Export",
    ]


def test_dock_fills_available_height_under_napari_0_9(make_napari_viewer):
    """The main widget should not leave a blank band below the sidebar."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    dock = viewer.window.add_dock_widget(widget, area="right")

    QTimer.singleShot(0, lambda: None)
    QApplication.processEvents()

    assert dock.maximumHeight() == QWIDGETSIZE_MAX
