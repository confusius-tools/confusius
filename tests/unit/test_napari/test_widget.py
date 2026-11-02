"""Unit tests for the main napari widget."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, QRect, QTimer
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QWIDGETSIZE_MAX, QApplication, QGroupBox

from confusius._napari._data._load_panel import DataPanel
from confusius._napari._tour import build_default_tour
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


def test_data_tour_highlights_video_reopen_buttons(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    viewer.window.add_dock_widget(widget, name="ConfUSIus")
    window = widget.window()
    window.setUpdatesEnabled(False)
    window.resize(1600, 900)
    window.show()
    QApplication.processEvents()
    tour = build_default_tour(widget)
    try:
        steps = {step.title: step for step in tour._steps}
        load_step = steps["Load Data"]
        assert load_step.target() is widget.findChild(DataPanel)._file_group
        assert "fUSI" in load_step.body
        assert "videos" in load_step.body

        video_step = steps["Video Viewer and List"]
        video_step.pre_action()
        panel = widget.findChild(VideoPanel)
        assert video_step.target() is panel
        buttons = (panel._show_viewer_btn, panel._show_layer_list_btn)
        rects = [
            QRect(button.mapTo(window, QPoint(0, 0)), button.size())
            for button in buttons
        ]
        assert video_step.spotlight_rect() == rects[0].united(rects[1])
        for button in buttons:
            assert button.text() in video_step.body
        load_index = tour._steps.index(load_step)
        assert tour._steps[load_index + 1] is video_step
    finally:
        tour.close_tour()


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
