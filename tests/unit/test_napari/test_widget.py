"""Tests for the main ConfUSIus napari plugin widget."""

from __future__ import annotations

from qtpy.QtWidgets import QComboBox

from confusius._napari._widget import ConfUSIusWidget


def test_section_selector_switches_visible_panel(qtbot, make_napari_viewer) -> None:
    """The top dropdown should drive the visible plugin section."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    qtbot.addWidget(widget)

    selector = widget.findChild(QComboBox, "section_selector")
    assert selector is not None
    assert [selector.itemText(i) for i in range(selector.count())] == [
        "Data I/O",
        "Video",
        "Signals",
        "Quality Control",
    ]
    assert selector.currentText() == "Data I/O"
    assert widget._section_stack.currentWidget() is widget._accordion_panels["Data I/O"]

    selector.setCurrentText("Signals")

    assert widget._section_stack.currentWidget() is widget._accordion_panels["Signals"]
