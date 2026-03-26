"""Interactive guided tour for the ConfUSIus napari plugin.

Displays a step-by-step overlay that highlights individual widgets and shows
explanatory tooltips, similar to onboarding tours in web applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import QPoint, QRect, QTimer, Qt, Signal
from qtpy.QtGui import QColor, QFont, QPainter, QPainterPath
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class TourStep:
    """A single step in the guided tour.

    Attributes
    ----------
    target : Callable[[], QWidget | None]
        Callable returning the widget to spotlight. Using a callable (rather than a
        direct reference) lets steps target widgets that may not exist yet when the tour
        is defined, or that are created lazily.
    title : str
        Short heading displayed in the tooltip.
    body : str
        Longer explanatory text.
    anchor : str
        Preferred tooltip placement relative to the target: ``"right"``, ``"left"``,
        ``"above"``, or ``"below"``.
    pre_action : Callable[[], None] | None
        Optional callback executed before this step is shown (e.g. to expand an
        accordion section so the target widget becomes visible).
    """

    target: Callable[[], QWidget | None]
    title: str
    body: str
    anchor: str = "right"
    pre_action: Callable[[], None] | None = None


# ---------------------------------------------------------------------------
# Tooltip bubble
# ---------------------------------------------------------------------------


class _TourTooltip(QWidget):
    """Floating tooltip with title, body text, and navigation buttons."""

    next_clicked = Signal()
    back_clicked = Signal()
    skip_clicked = Signal()

    _PADDING = 16
    _MAX_WIDTH = 320

    def __init__(self, parent: QWidget | None = None) -> None:
        # No parent — fully independent top-level window. This avoids any
        # event routing through the main window or compositor quirks.
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setObjectName("tour_tooltip")
        self.setMaximumWidth(self._MAX_WIDTH)

        self._build_ui()

    # -- UI ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self._PADDING, self._PADDING, self._PADDING, self._PADDING
        )
        layout.setSpacing(8)

        self._title_label = QLabel()
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        self._title_label.setFont(font)
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._body_label = QLabel()
        self._body_label.setWordWrap(True)
        self._body_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )
        layout.addWidget(self._body_label)

        # Navigation row.
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

    # -- Public interface ----------------------------------------------------

    def apply_theme(self, *, is_dark: bool) -> None:
        """Apply theme colors once (not on every step change)."""
        accent = "#ffd33d" if is_dark else "#c49a0a"
        accent_fg = "#1c1c27" if is_dark else "#ffffff"
        bg = "#2d2d3a" if is_dark else "#ffffff"
        fg = "#c8c8d4" if is_dark else "#2c2c3a"
        border = "#3d3d4a" if is_dark else "#d0d0d8"
        btn_bg = "#38384a" if is_dark else "#e0e0e8"

        self.setStyleSheet(
            f"""
            #tour_tooltip {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            #tour_tooltip QLabel {{
                color: {fg};
                background: transparent;
            }}
            #tour_tooltip QPushButton {{
                background: {btn_bg};
                color: {fg};
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
            }}
            #tour_tooltip QPushButton:hover {{
                background: {accent};
                color: {accent_fg};
            }}
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 100 if is_dark else 60))
        self.setGraphicsEffect(shadow)

    def set_content(self, title: str, body: str, step: int, total: int) -> None:
        """Update the tooltip text and navigation state."""
        self._title_label.setText(title)
        self._body_label.setText(body)
        self._counter.setText(f"{step}/{total}")

        is_last = step == total
        self._next_btn.setText("Finish" if is_last else "Next")
        self._back_btn.setVisible(step > 1)

    def position_near(self, target_rect: QRect, anchor: str) -> None:
        """Move the tooltip near *target_rect* (in global coordinates)."""
        self.adjustSize()
        gap = 12
        tw, th = self.width(), self.height()

        if anchor == "right":
            x = target_rect.right() + gap
            y = target_rect.center().y() - th // 2
        elif anchor == "left":
            x = target_rect.left() - tw - gap
            y = target_rect.center().y() - th // 2
        elif anchor == "above":
            x = target_rect.center().x() - tw // 2
            y = target_rect.top() - th - gap
        else:  # below
            x = target_rect.center().x() - tw // 2
            y = target_rect.bottom() + gap

        # Keep the tooltip on-screen.
        screen = QApplication.primaryScreen()
        if screen is not None:
            sr = screen.availableGeometry()
            x = max(sr.left() + 8, min(x, sr.right() - tw - 8))
            y = max(sr.top() + 8, min(y, sr.bottom() - th - 8))

        self.move(x, y)


# ---------------------------------------------------------------------------
# Overlay (dark scrim with spotlight cutout)
# ---------------------------------------------------------------------------

_SPOTLIGHT_PADDING = 6
_SPOTLIGHT_RADIUS = 8


class _TourOverlay(QWidget):
    """Full-window semi-transparent overlay with a rounded-rect cutout.

    This widget is purely visual — all mouse events pass through to the
    widgets underneath. This avoids expensive QRegion mask recalculations
    and ensures the overlay doesn't interfere with event delivery.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Purely visual: all mouse/keyboard events pass through.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._spotlight: QRect | None = None
        self._opacity = 0.55

    def set_spotlight(self, rect: QRect | None) -> None:
        """Set the region to cut out, in parent-widget coordinates."""
        self._spotlight = rect
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Full dark wash.
        wash = QColor(0, 0, 0, round(255 * self._opacity))
        painter.fillRect(self.rect(), wash)

        # Punch out the spotlight.
        if self._spotlight is not None:
            path = QPainterPath()
            path.addRoundedRect(
                self._spotlight.adjusted(
                    -_SPOTLIGHT_PADDING,
                    -_SPOTLIGHT_PADDING,
                    _SPOTLIGHT_PADDING,
                    _SPOTLIGHT_PADDING,
                ),
                _SPOTLIGHT_RADIUS,
                _SPOTLIGHT_RADIUS,
            )
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillPath(path, Qt.GlobalColor.transparent)

        painter.end()


# ---------------------------------------------------------------------------
# Tour controller
# ---------------------------------------------------------------------------


class GuidedTour(QWidget):
    """Manages a multi-step overlay tour anchored to a top-level window.

    Parameters
    ----------
    steps : list[TourStep]
        Ordered tour steps.
    parent_window : QWidget
        The top-level window (e.g. napari's main window) to overlay.
    is_dark : bool
        Whether the current theme is dark.
    """

    finished = Signal()

    # Delay before measuring widget positions after a pre_action. Must
    # comfortably exceed the accordion animation duration (200 ms) plus the
    # layout reflow that happens after the collapse animation hides a panel.
    _SETTLE_MS = 350

    def __init__(
        self,
        steps: list[TourStep],
        parent_window: QWidget,
        *,
        is_dark: bool = True,
    ) -> None:
        super().__init__(parent_window)
        self._steps = steps
        self._window = parent_window
        self._is_dark = is_dark
        self._current = 0
        # Generation counter: incremented on every _show_step call so that
        # stale QTimer callbacks from rapid next/back clicks are discarded.
        self._generation = 0

        # Overlay covers the whole window.
        self._overlay = _TourOverlay(parent_window)

        # Tooltip is a parentless top-level window so it is fully independent
        # from the main window's event processing and compositor handling.
        self._tooltip = _TourTooltip()
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
        self._show_step(0)

    def close_tour(self) -> None:
        """Tear down the overlay and tooltip."""
        self._generation += 1  # Invalidate any pending timers.
        self._tooltip.hide()
        self._overlay.hide()
        self._tooltip.deleteLater()
        self._overlay.deleteLater()
        self.finished.emit()
        self.deleteLater()

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
            # Clear the spotlight so the overlay is a plain dark wash while
            # the accordion animates (no stale highlight). The overlay stays
            # visible so there is no jarring flash.
            self._overlay.set_spotlight(None)

            step.pre_action()

            # Defer position measurement until the accordion animation and
            # layout reflow have finished. The generation check discards this
            # callback if the user clicked next/back again in the meantime.
            QTimer.singleShot(
                self._SETTLE_MS,
                lambda g=gen: self._position_step(index, g),
            )
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

        # Ensure overlay covers the full window.
        self._overlay.setGeometry(self._window.rect())

        # Map target geometry to the overlay's coordinate system.
        top_left = target.mapTo(self._window, QPoint(0, 0))
        target_rect_local = QRect(top_left, target.size())
        self._overlay.set_spotlight(target_rect_local)

        # Tooltip uses global (screen) coordinates.
        target_rect_global = QRect(target.mapToGlobal(QPoint(0, 0)), target.size())
        self._tooltip.set_content(step.title, step.body, index + 1, len(self._steps))
        self._tooltip.position_near(target_rect_global, step.anchor)
        self._tooltip.raise_()


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
    window = plugin_widget.window()
    if window is None:
        window = plugin_widget

    # Helper to find accordion header buttons by label text.
    def _accordion_btn(label: str) -> Callable[[], QWidget | None]:
        def _find() -> QWidget | None:
            for btn, _icon in getattr(plugin_widget, "_accordion_btns", []):
                if btn.text() == label:
                    return btn
            return None

        return _find

    # Helper to return the panel content widget for a given accordion section.
    def _accordion_panel(label: str) -> Callable[[], QWidget | None]:
        def _find() -> QWidget | None:
            panels = getattr(plugin_widget, "_accordion_panels", {})
            return panels.get(label)

        return _find

    # Helper to click an accordion button to expand its section.
    def _expand_section(label: str) -> Callable[[], None]:
        def _action() -> None:
            for btn, _icon in getattr(plugin_widget, "_accordion_btns", []):
                if btn.text() == label and not btn.isChecked():
                    btn.click()

        return _action

    steps = [
        TourStep(
            target=lambda: plugin_widget.findChild(QWidget, "confusius_header"),
            title="Welcome to ConfUSIus!",
            body=(
                "This plugin lets you load, visualize, and analyze functional "
                "ultrasound imaging (fUSI) data directly inside napari.\n\n"
                "Let\u2019s take a quick tour of the main sections."
            ),
            anchor="below",
        ),
        TourStep(
            target=_accordion_panel("Data I/O"),
            title="Data I/O",
            body=(
                "Load fUSI volumes from NIfTI, Zarr, or Iconeus SCAN files, and "
                "save your results back. Click the header to expand the panel."
            ),
            anchor="right",
            pre_action=_expand_section("Data I/O"),
        ),
        TourStep(
            target=_accordion_panel("Time Series"),
            title="Time Series",
            body=(
                "Explore voxel time courses interactively. Hover with Shift held "
                "to see live traces, or pick points and label regions to compare "
                "signals across areas."
            ),
            anchor="right",
            pre_action=_expand_section("Time Series"),
        ),
        TourStep(
            target=_accordion_panel("Quality Control"),
            title="Quality Control",
            body=(
                "Compute QC metrics like DVARS, tSNR, and coefficient of "
                "variation to assess your data quality before analysis."
            ),
            anchor="right",
            pre_action=_expand_section("Quality Control"),
        ),
    ]

    return GuidedTour(steps, window, is_dark=is_dark)
