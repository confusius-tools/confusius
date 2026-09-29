"""Unit tests for shared Qt helpers."""

from __future__ import annotations

from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QApplication, QComboBox, QScrollArea, QVBoxLayout, QWidget

from confusius._napari._qt import install_no_scroll_wheel_filter


def _wheel_event() -> QWheelEvent:
    return QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_wheel_over_combo_box_scrolls_area_instead_of_changing_value(qtbot):
    scroll = QScrollArea()
    content = QWidget()
    layout = QVBoxLayout(content)
    combo = QComboBox()
    combo.addItems(["a", "b", "c"])
    layout.addWidget(combo)
    # Pad well past the scroll area's viewport so there is something to scroll.
    for _ in range(50):
        pad = QWidget()
        pad.setMinimumHeight(20)
        layout.addWidget(pad)
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)
    scroll.resize(200, 100)
    qtbot.addWidget(scroll)
    scroll.show()
    qtbot.waitExposed(scroll)
    QApplication.processEvents()

    install_no_scroll_wheel_filter(content)

    vbar = scroll.verticalScrollBar()
    assert vbar is not None
    assert vbar.maximum() > 0  # sanity: there is actually something to scroll
    before_index, before_scroll = combo.currentIndex(), vbar.value()

    QApplication.sendEvent(combo, _wheel_event())
    QApplication.processEvents()

    assert combo.currentIndex() == before_index
    assert vbar.value() != before_scroll


def test_combo_box_without_scroll_area_ancestor_still_swallows_wheel(qtbot):
    # No QAbstractScrollArea anywhere in the ancestry: the filter must still
    # swallow the event (leaving the combo's value untouched) rather than error
    # out looking for somewhere to forward it.
    parent = QWidget()
    layout = QVBoxLayout(parent)
    combo = QComboBox()
    combo.addItems(["a", "b", "c"])
    layout.addWidget(combo)
    qtbot.addWidget(parent)
    install_no_scroll_wheel_filter(parent)

    before_index = combo.currentIndex()
    QApplication.sendEvent(combo, _wheel_event())

    assert combo.currentIndex() == before_index
