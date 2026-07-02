"""Tests for the main ConfUSIus napari plugin widget."""

from __future__ import annotations

from confusius._napari._widget import ConfUSIusWidget


def test_section_rail_switches_visible_panel(qtbot, make_napari_viewer) -> None:
    """The icon rail buttons should drive the visible plugin section."""
    viewer = make_napari_viewer()
    widget = ConfUSIusWidget(viewer)
    qtbot.addWidget(widget)

    labels = [btn.text() for btn, _icon in widget._accordion_btns]
    assert labels == ["Data I/O", "Video", "Signals", "Quality Control"]
    data_btn = widget._accordion_btns[0][0]
    assert data_btn.isChecked()
    assert widget._section_stack.currentWidget() is widget._accordion_panels["Data I/O"]

    signals_btn = next(
        btn for btn, _icon in widget._accordion_btns if btn.text() == "Signals"
    )
    signals_btn.click()

    assert widget._section_stack.currentWidget() is widget._accordion_panels["Signals"]
    assert signals_btn.isChecked()
    assert not data_btn.isChecked()
