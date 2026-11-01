"""Unit tests for the main napari widget."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtGui import QPalette
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


@pytest.mark.parametrize(
    ("theme", "enabled_color", "disabled_color"),
    [("dark", "#c8c8d4", "#565668"), ("light", "#2c2c3a", "#a0a0b0")],
)
def test_video_reopen_buttons_look_disabled(
    make_napari_viewer, theme, enabled_color, disabled_color
):
    viewer = make_napari_viewer()
    viewer.theme = theme
    widget = ConfUSIusWidget(viewer)
    panel = widget.findChild(VideoPanel)

    for button in (panel._show_viewer_btn, panel._show_layer_list_btn):
        button.ensurePolished()
        assert not button.isEnabled()
        assert (
            button.palette()
            .color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
            .name()
            == disabled_color
        )
        button.setEnabled(True)
        assert (
            button.palette()
            .color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText)
            .name()
            == enabled_color
        )
        button.setEnabled(False)
        assert (
            button.palette()
            .color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
            .name()
            == disabled_color
        )


def test_dock_fills_available_height_under_napari_0_9(make_napari_viewer):
    """The main widget should not leave a blank band below the sidebar."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    dock = viewer.window.add_dock_widget(widget, area="right")

    QTimer.singleShot(0, lambda: None)
    QApplication.processEvents()

    assert dock.maximumHeight() == QWIDGETSIZE_MAX
