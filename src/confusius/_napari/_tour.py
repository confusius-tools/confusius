"""Interactive guided tour for the ConfUSIus napari plugin.

Displays a step-by-step overlay that highlights individual widgets and shows
explanatory tooltips, similar to onboarding tours in web applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QMouseEvent, QPaintEvent


@dataclass(frozen=True)
class TourStep:
    """A single step in the guided tour.

    Attributes
    ----------
    target : Callable[[], QWidget | None]
        Callable returning the widget to spotlight. Using a callable (rather
        than a direct reference) lets steps target widgets that may not exist
        yet when the tour is defined, or that are created lazily.
    title : str
        Short heading displayed in the tooltip.
    body : str
        Longer explanatory text.
    anchor : str
        Preferred tooltip placement relative to the target: ``"right"``,
        ``"left"``, ``"above"``, or ``"below"``.
    spotlight_rect : Callable[[], QRect | None] | None
        Optional callable returning a custom spotlight rectangle in window
        coordinates. Use this when a step should highlight multiple related
        widgets together rather than a single QWidget.
    tooltip_target : Callable[[], QWidget | None] | None
        Optional callable returning the widget used only for tooltip placement.
        This lets the spotlight track a small control while the tooltip is
        positioned relative to a larger container such as the napari dock.
    pre_action : Callable[[], None] | None
        Optional callback executed before this step is shown (e.g. to expand
        an accordion section so the target widget becomes visible).
    """

    target: Callable[[], QWidget | None]
    title: str
    body: str
    anchor: str = "right"
    spotlight_rect: Callable[[], QRect | None] | None = None
    tooltip_target: Callable[[], QWidget | None] | None = None
    pre_action: Callable[[], None] | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed tooltip width so word-wrapped QLabels can compute their height
# correctly via heightForWidth() when adjustSize() is called.
_TOOLTIP_WIDTH = 300
_SPOTLIGHT_PADDING = 6
_SPOTLIGHT_RADIUS = 8
_SPOTLIGHT_BORDER_WIDTH = 2


# ---------------------------------------------------------------------------
# Tooltip bubble
# ---------------------------------------------------------------------------


class _TourTooltip(QWidget):
    """Tooltip shown above the overlay as a regular child widget.

    Keeping the tooltip as a sibling of the overlay avoids repainting the full
    scrim every time a button hover state changes, which made navigation feel
    sluggish inside napari.
    """

    next_clicked = Signal()
    back_clicked = Signal()
    skip_clicked = Signal()

    _PADDING = 14

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Plain QWidget ignores stylesheet background/border unless this is set.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("tour_tooltip")
        # Fixed width lets the layout call heightForWidth(_TOOLTIP_WIDTH) when
        # adjustSize() is invoked, giving the correct height for wrapped text.
        self.setFixedWidth(_TOOLTIP_WIDTH)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self._PADDING, self._PADDING, self._PADDING, self._PADDING
        )
        layout.setSpacing(8)

        self._title_label = QLabel()
        self._title_label.setObjectName("tour_tooltip_title")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        self._title_label.setFont(f)
        self._title_label.setTextFormat(Qt.TextFormat.RichText)
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._body_label = QLabel()
        self._body_label.setTextFormat(Qt.TextFormat.RichText)
        self._body_label.setWordWrap(True)
        layout.addWidget(self._body_label)

        nav = QHBoxLayout()
        nav.setSpacing(8)
        self._counter = QLabel()
        self._counter.setStyleSheet("font-size: 11px;")
        nav.addWidget(self._counter)
        nav.addStretch()
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self.back_clicked)
        nav.addWidget(self._back_btn)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self.next_clicked)
        nav.addWidget(self._next_btn)
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self.skip_clicked)
        nav.addWidget(self._skip_btn)
        layout.addLayout(nav)

    def apply_theme(self, *, is_dark: bool) -> None:
        """Apply theme colours."""
        accent = "#ffd33d" if is_dark else "#c49a0a"
        accent_fg = "#1c1c27" if is_dark else "#ffffff"
        bg = "#2d2d3a" if is_dark else "#ffffff"
        fg = "#c8c8d4" if is_dark else "#2c2c3a"
        border = "#3d3d4a" if is_dark else "#d0d0d8"
        btn_bg = "#38384a" if is_dark else "#e0e0e8"
        self.setStyleSheet(f"""
            QWidget#tour_tooltip {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QWidget#tour_tooltip QLabel {{
                color: {fg};
                background: transparent;
                border: none;
            }}
            QWidget#tour_tooltip QLabel#tour_tooltip_title {{
                color: {accent};
            }}
            QWidget#tour_tooltip QPushButton {{
                background: {btn_bg};
                color: {fg};
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
            }}
            QWidget#tour_tooltip QPushButton:hover {{
                background: {accent};
                color: {accent_fg};
            }}
        """)

    def set_content(self, title: str, body: str, step: int, total: int) -> None:
        """Update text and navigation state, then recalculate size."""
        self._title_label.setText(title)
        self._body_label.setText(body)
        self._counter.setText(f"{step}/{total}")
        self._next_btn.setText("Finish" if step == total else "Next")
        self._back_btn.setVisible(step > 1)
        self._skip_btn.setVisible(step < total)

        layout = self.layout()
        if layout is None:
            return
        margins = layout.contentsMargins()
        content_width = self.width() - margins.left() - margins.right()

        # Qt's layout size hints are unreliable here with wrapped labels inside
        # dock widgets. Compute the wrapped label heights explicitly from the
        # available text width so longer tour copy cannot be clipped.
        self._title_label.setFixedWidth(content_width)
        self._body_label.setFixedWidth(content_width)
        self._title_label.setFixedHeight(
            self._title_label.heightForWidth(content_width)
        )
        self._body_label.setFixedHeight(self._body_label.heightForWidth(content_width))

        layout.activate()
        self.setFixedHeight(layout.sizeHint().height())

    def place(self, target_rect: QRect, anchor: str, bounds: QRect) -> None:
        """Position near target_rect, clipped to bounds (window coordinates)."""
        gap = 12
        w, h = self.width(), self.height()
        if anchor == "right":
            x, y = target_rect.right() + gap, target_rect.center().y() - h // 2
        elif anchor == "left":
            x, y = target_rect.left() - w - gap, target_rect.top()
        elif anchor == "above":
            x, y = target_rect.center().x() - w // 2, target_rect.top() - h - gap
        else:  # below
            x, y = target_rect.center().x() - w // 2, target_rect.bottom() + gap
        x = max(bounds.left() + 8, min(x, bounds.right() - w - 8))
        y = max(bounds.top() + 8, min(y, bounds.bottom() - h - 8))
        self.move(x, y)


# ---------------------------------------------------------------------------
# Overlay (dark scrim with spotlight cutout)
# ---------------------------------------------------------------------------


class _TourOverlay(QWidget):
    """Full-window dark scrim with a spotlight cutout.

    Mouse events are consumed here so the underlying UI is not accidentally
    activated during the tour. The tooltip itself is rendered as a sibling
    widget above this overlay.
    """

    _OPACITY = 0.55

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._spotlight: QRect | None = None

    def set_spotlight(self, rect: QRect | None) -> None:
        """Update the cutout region (in parent-widget coordinates)."""
        self._spotlight = rect
        self.update()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # type: ignore[invalid-method-override]
        # Consume clicks outside the tooltip so underlying widgets are not accidentally
        # activated during the tour.
        if event is not None:
            event.accept()

    def paintEvent(self, _event: QPaintEvent | None) -> None:  # type: ignore[invalid-method-override]  # noqa: N802
        painter = QPainter()
        if not painter.begin(self):
            return

        try:
            overlay_color = QColor(0, 0, 0, round(255 * self._OPACITY))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            if self._spotlight is not None:
                rect = self._spotlight.adjusted(
                    -_SPOTLIGHT_PADDING,
                    -_SPOTLIGHT_PADDING,
                    _SPOTLIGHT_PADDING,
                    _SPOTLIGHT_PADDING,
                )
                painter.fillRect(0, 0, self.width(), rect.top(), overlay_color)
                painter.fillRect(
                    0,
                    rect.bottom() + 1,
                    self.width(),
                    self.height() - rect.bottom() - 1,
                    overlay_color,
                )
                painter.fillRect(
                    0, rect.top(), rect.left(), rect.height(), overlay_color
                )
                painter.fillRect(
                    rect.right() + 1,
                    rect.top(),
                    self.width() - rect.right() - 1,
                    rect.height(),
                    overlay_color,
                )
                painter.setPen(QPen(QColor(255, 211, 61, 230), _SPOTLIGHT_BORDER_WIDTH))
                painter.drawRoundedRect(rect, _SPOTLIGHT_RADIUS, _SPOTLIGHT_RADIUS)
            else:
                painter.fillRect(self.rect(), overlay_color)
        finally:
            painter.end()


# ---------------------------------------------------------------------------
# Tour controller
# ---------------------------------------------------------------------------


class GuidedTour(QObject):
    """Manages a multi-step overlay tour anchored to a top-level window.

    Parameters
    ----------
    steps : list[TourStep]
        Ordered tour steps.
    parent_window : QWidget
        The top-level window (e.g. napari's main window) to overlay.
    is_dark : bool
        Whether the current theme is dark.
    on_close : Callable[[], None] | None
        Optional callback invoked when the tour ends (skip or finish). Use
        this to restore any UI state changed by the tour's pre-actions.
    """

    finished = Signal()

    def __init__(
        self,
        steps: list[TourStep],
        parent_window: QWidget,
        *,
        is_dark: bool = True,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent_window)
        self._steps = steps
        self._window = parent_window
        self._current = 0
        # Incremented on every _show_step call so stale QTimer callbacks from
        # rapid next/back clicks are discarded.
        self._generation = 0
        self._on_close = on_close
        self._active = False
        self._event_filter_installed = False

        self._overlay = _TourOverlay(parent_window)
        self._tooltip = _TourTooltip(parent_window)
        self._tooltip.apply_theme(is_dark=is_dark)
        self._tooltip.next_clicked.connect(self._on_next)
        self._tooltip.back_clicked.connect(self._on_back)
        self._tooltip.skip_clicked.connect(self.close_tour)

    # -- Public API ----------------------------------------------------------

    def start(self) -> None:
        """Show the overlay and display the first step."""
        self._overlay.setGeometry(self._window.rect())
        self._overlay.show()
        self._overlay.raise_()
        self._tooltip.show()
        self._tooltip.raise_()
        self._active = True
        # Watch the top-level window for resize events so the scrim, spotlight,
        # and tooltip stay aligned when the napari window or its docks change
        # size while the tour is active.
        self._window.installEventFilter(self)
        self._event_filter_installed = True
        # Defer one event-loop iteration so the tooltip completes its initial
        # layout pass before set_content calls adjustSize().
        QTimer.singleShot(0, lambda: self._show_step(0))

    def close_tour(self) -> None:
        """Tear down the overlay and restore the UI to its pre-tour state."""
        self._active = False
        self._generation += 1  # Invalidate any pending timers.
        if self._event_filter_installed:
            self._window.removeEventFilter(self)
            self._event_filter_installed = False
        self._overlay.hide()
        self._tooltip.hide()
        if self._on_close is not None:
            self._on_close()
        self._overlay.deleteLater()
        self._tooltip.deleteLater()
        self.finished.emit()
        self.deleteLater()

    # -- Event handling ------------------------------------------------------

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # type: ignore[invalid-method-override]  # noqa: N802
        """Reposition the overlay when the watched window is resized.

        Parameters
        ----------
        watched : qtpy.QtCore.QObject or None
            The object the event was sent to; only ``self._window`` is acted
            on.
        event : qtpy.QtCore.QEvent or None
            The intercepted event. Only `QEvent.Type.Resize` triggers a
            reposition; all others fall through unchanged.

        Returns
        -------
        bool
            Always the base class result — the event is never consumed so
            napari's own layout handlers still run.
        """
        if (
            event is not None
            and watched is self._window
            and event.type() == QEvent.Type.Resize
            and self._active
        ):
            self._reposition_current()
        return super().eventFilter(watched, event)

    # -- Navigation ----------------------------------------------------------

    def _on_next(self) -> None:
        if self._current < len(self._steps) - 1:
            self._show_step(self._current + 1)
        else:
            self.close_tour()

    def _on_back(self) -> None:
        if self._current > 0:
            self._show_step(self._current - 1)

    def _show_step(self, index: int) -> None:
        self._current = index
        self._generation += 1
        gen = self._generation
        step = self._steps[index]

        if step.pre_action is not None:
            step.pre_action()
            # Let Qt finish any visibility or layout changes before measuring.
            QTimer.singleShot(0, lambda g=gen: self._position_step(index, g))
        else:
            self._position_step(index, gen)

    def _position_step(self, index: int, generation: int) -> None:
        """Measure target geometry and place the spotlight + tooltip."""
        # Stale callback — the user navigated away before we fired.
        if generation != self._generation:
            return

        step = self._steps[index]
        target = step.target()
        if target is None or not target.isVisible():
            if index < len(self._steps) - 1:
                self._show_step(index + 1)
            else:
                self.close_tour()
            return

        self._tooltip.set_content(step.title, step.body, index + 1, len(self._steps))
        self._apply_geometry(step, target)
        self._tooltip.show()
        self._tooltip.raise_()

    def _reposition_current(self) -> None:
        """Recompute geometry for the active step without re-running pre_action."""
        if not self._active:
            return
        if not (0 <= self._current < len(self._steps)):
            return
        step = self._steps[self._current]
        target = step.target()
        if target is None or not target.isVisible():
            return
        self._apply_geometry(step, target)

    def _apply_geometry(self, step: TourStep, target: QWidget) -> None:
        """Resize the scrim and place the spotlight + tooltip for `step`."""
        # Keep overlay geometry in sync with the window (handles resizes).
        self._overlay.setGeometry(self._window.rect())

        # Convert the highlighted widget into window coordinates.
        top_left = target.mapTo(self._window, QPoint(0, 0))
        target_rect = QRect(top_left, target.size())
        if step.spotlight_rect is not None:
            custom_rect = step.spotlight_rect()
            if custom_rect is not None and not custom_rect.isNull():
                target_rect = custom_rect
        self._overlay.set_spotlight(target_rect)

        anchor_target = (
            step.tooltip_target() if step.tooltip_target is not None else target
        )
        anchor_rect = target_rect
        if anchor_target is not None and anchor_target.isVisible():
            anchor_top_left = anchor_target.mapTo(self._window, QPoint(0, 0))
            anchor_rect = QRect(anchor_top_left, anchor_target.size())

        placement_rect = anchor_rect
        if step.tooltip_target is not None and step.anchor == "left":
            placement_rect = QRect(anchor_rect)
            placement_rect.moveTop(target_rect.top())

        self._tooltip.place(placement_rect, step.anchor, self._window.rect())


# ---------------------------------------------------------------------------
# Convenience: build the default tour for the ConfUSIus widget
# ---------------------------------------------------------------------------


def build_default_tour(
    plugin_widget: QWidget,
    *,
    is_dark: bool = True,
) -> GuidedTour:
    """Create the standard ConfUSIus tour.

    Parameters
    ----------
    plugin_widget : QWidget
        The `ConfUSIusWidget` instance.
    is_dark : bool
        Current theme brightness.

    Returns
    -------
    GuidedTour
        Ready-to-start tour instance.
    """
    from confusius._napari._data._load_panel import DataPanel
    from confusius._napari._data._save_panel import SavePanel
    from confusius._napari._qc._panel import QCPanel
    from confusius._napari._signals._panel import SignalPanel

    window = plugin_widget.window() or plugin_widget

    def _dock_widget() -> QWidget | None:
        parent = plugin_widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QDockWidget):
                return parent
            parent = parent.parentWidget()

        main_window = plugin_widget.window()
        if main_window is None:
            return None

        for dock in main_window.findChildren(QDockWidget):
            if dock.isAncestorOf(plugin_widget):
                return dock
        return None

    def _accordion_panel(label: str) -> Callable[[], QWidget | None]:
        def _find() -> QWidget | None:
            panels = getattr(plugin_widget, "_accordion_panels", {})
            return panels.get(label)

        return _find

    def _accordion_button(label: str) -> Callable[[], QWidget | None]:
        def _find() -> QWidget | None:
            for btn, _icon in getattr(plugin_widget, "_accordion_btns", []):
                if btn.text() == label:
                    return btn
            return None

        return _find

    def _widget_rect(widget: QWidget | None) -> QRect | None:
        if widget is None or not widget.isVisible():
            return None
        top_left = widget.mapTo(window, QPoint(0, 0))
        return QRect(top_left, widget.size())

    def _united_rect(*widgets: QWidget | None) -> QRect | None:
        rects = [
            rect for widget in widgets if (rect := _widget_rect(widget)) is not None
        ]
        if not rects:
            return None
        united = QRect(rects[0])
        for rect in rects[1:]:
            united = united.united(rect)
        return united

    def _accordion_tab_rect(label: str) -> Callable[[], QRect | None]:
        def _find() -> QRect | None:
            return _united_rect(_accordion_button(label)(), _accordion_panel(label)())

        return _find

    def _panel_descendant(
        label: str,
        widget_type: type[QWidget],
    ) -> Callable[[], QWidget | None]:
        def _find() -> QWidget | None:
            panel = _accordion_panel(label)()
            if panel is None:
                return None
            if isinstance(panel, widget_type):
                return panel
            return panel.findChild(widget_type)

        return _find

    def _panel_attr(
        label: str, widget_type: type[QWidget], attr: str
    ) -> Callable[[], QWidget | None]:
        def _find() -> QWidget | None:
            panel = _panel_descendant(label, widget_type)()
            if panel is None:
                return None
            widget = getattr(panel, attr, None)
            return widget if isinstance(widget, QWidget) else None

        return _find

    def _panel_attr_rect(
        label: str,
        widget_type: type[QWidget],
        *attrs: str,
    ) -> Callable[[], QRect | None]:
        def _find() -> QRect | None:
            widgets = [_panel_attr(label, widget_type, attr)() for attr in attrs]
            return _united_rect(*widgets)

        return _find

    def _expand_section(label: str) -> Callable[[], None]:
        def _action() -> None:
            for btn, _icon in getattr(plugin_widget, "_accordion_btns", []):
                if btn.text() == label and not btn.isChecked():
                    btn.click()

        return _action

    # Record the open panel before the tour starts so we can restore it when
    # the tour is closed or skipped.
    initial_open: str | None = next(
        (
            btn.text()
            for btn, _ in getattr(plugin_widget, "_accordion_btns", [])
            if btn.isChecked()
        ),
        None,
    )

    def _restore_state() -> None:
        if initial_open is None:
            return
        for btn, _ in getattr(plugin_widget, "_accordion_btns", []):
            if btn.text() == initial_open and not btn.isChecked():
                btn.click()
                break

    steps = [
        TourStep(
            target=lambda: plugin_widget.findChild(QWidget, "confusius_header"),
            title="Welcome to ConfUSIus!",
            body=(
                "ConfUSIus offers a napari plugin to help you load fUSI data, "
                "explore signals, and run quality control.<br><br>"
                "This tour will point out the different features of ConfUSIus and how "
                "to use them."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
        ),
        TourStep(
            target=_accordion_panel("Data I/O"),
            title="Data I/O",
            body=(
                "This section is where data enters and leaves ConfUSIus. Load fUSI "
                "datasets from disk, then save processed layers or annotations back "
                "out with the coordinates you want to preserve."
            ),
            anchor="left",
            spotlight_rect=_accordion_tab_rect("Data I/O"),
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Data I/O"),
        ),
        TourStep(
            target=_panel_attr("Data I/O", DataPanel, "_file_group"),
            title="File Loading",
            body=(
                "Open NIfTI, SCAN, or Zarr files here. <b>Browse</b> selects a dataset "
                "and starts loading it right away. Tick <b>Load lazily</b> to keep "
                "large arrays on disk and load frames as you view them."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Data I/O"),
        ),
        TourStep(
            target=_panel_attr("Data I/O", SavePanel, "_save_group"),
            title="Layer Saving",
            body=(
                "Save any napari layer from here. <b>Coordinates from</b> is useful "
                "when exporting labels or derived maps: it allows reusing the physical "
                "coordinates from a ConfUSIus-loaded image layer."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Data I/O"),
        ),
        TourStep(
            target=_accordion_panel("Signals"),
            title="Signals",
            body=(
                "Use this section to inspect signals from voxels, points, or regions "
                "compare them to imported signals, and export signals for use in other "
                "analyses or tools."
            ),
            anchor="left",
            spotlight_rect=_accordion_tab_rect("Signals"),
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Signals"),
        ),
        TourStep(
            target=_panel_attr("Signals", SignalPanel, "_source_group"),
            title="Choose a Signal Source",
            body=(
                "Signals can come from live <b>Shift-hovering</b> you mouse, from a "
                "<b>Points</b> layer, or from a <b>Labels</b> layer for region "
                "averages. The <b>+</b> buttons create helper layers so you can start "
                "annotating immediately."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Signals"),
        ),
        TourStep(
            target=_panel_attr("Signals", SignalPanel, "_axis_group"),
            title="Axis Parameters",
            body=(
                "These controls define what the plot shows: choose the <i>x</i>-axis, "
                "keep the <i>y</i>-range on autoscale, or lock it manually when you "
                "want stable comparisons across traces."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Signals"),
        ),
        TourStep(
            target=_panel_attr("Signals", SignalPanel, "_display_group"),
            title="Display Options",
            body=(
                "Use these options to add a grid, show the current <i>x</i>-axis "
                "cursor, or <i>z</i>-score traces."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Signals"),
        ),
        TourStep(
            target=_panel_attr("Signals", SignalPanel, "_show_btn"),
            title="Open and Manage Signals",
            body=(
                "<b>Show Signal Plot</b> opens the docked plot view below napari. "
                "<b>Manage Signals</b> lets you toggle, rename, recolor, import, and "
                "export traces."
            ),
            anchor="left",
            spotlight_rect=_panel_attr_rect(
                "Signals",
                SignalPanel,
                "_show_btn",
                "_manage_btn",
            ),
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Signals"),
        ),
        TourStep(
            target=_accordion_panel("Quality Control"),
            title="Quality Control",
            body=(
                "This section helps you perform quality control (QC) on a recording. "
                "Some QC outputs appear as plots, while others come back as viewer "
                "layers."
            ),
            anchor="left",
            spotlight_rect=_accordion_tab_rect("Quality Control"),
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Quality Control"),
        ),
        TourStep(
            target=_panel_attr("Quality Control", QCPanel, "_layer_group"),
            title="Choose Layers and Metrics",
            body=(
                "Pick the layer to evaluate, then choose temporal metrics like DVARS "
                "and carpet plots or spatial metrics like tSNR and CV."
            ),
            anchor="left",
            spotlight_rect=_panel_attr_rect(
                "Quality Control",
                QCPanel,
                "_layer_group",
                "_temporal_group",
                "_spatial_group",
            ),
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Quality Control"),
        ),
        TourStep(
            target=_panel_attr("Quality Control", QCPanel, "_compute_btn"),
            title="Run Quality Control",
            body=(
                "After choosing a layer and the metrics you want, <b>Compute</b> runs "
                "the analysis in the background. QC plots will be shown once computed, "
                "and a <b>Show QC plots</b> button will appear to bring them back "
                "without recomputing if you close them."
            ),
            anchor="left",
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Quality Control"),
        ),
        TourStep(
            target=lambda: plugin_widget,
            title="You're Ready to Explore!",
            body=(
                "You're all set to start exploring. Load a dataset in Data I/O, "
                "explore signals, run a few QC checks, and have fun digging into some "
                "fUSI data!"
            ),
            anchor="left",
            spotlight_rect=lambda: _widget_rect(plugin_widget),
            tooltip_target=_dock_widget,
            pre_action=_expand_section("Data I/O"),
        ),
    ]

    return GuidedTour(steps, window, is_dark=is_dark, on_close=_restore_state)
