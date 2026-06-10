"""Main ConfUSIus napari plugin widget."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import (
    QByteArray,
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
)
from qtpy.QtGui import QFont, QImage, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer as _QSvgRenderer
from qtpy.QtWidgets import (
    QWIDGETSIZE_MAX,
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._theme import make_lucide_icon
from confusius._napari._time_overlay import _TimeOverlay

if TYPE_CHECKING:
    import napari

    from confusius._napari._tour import GuidedTour

_ASSETS_DIR = Path(__file__).parent / "assets"

ACCORDION_ANIMATION_DURATION_MS = 200
"""Duration of accordion expand/collapse animations, in milliseconds."""


def _build_stylesheet(is_dark: bool, napari_bg: str | None = None) -> str:  # noqa: C901
    """Return a full QSS stylesheet parametrised by theme brightness."""
    group_title_bg = napari_bg or ("#1c1c27" if is_dark else "#f0f0e8")
    if is_dark:
        header_bg = napari_bg or "#1c1c27"
        header_border = "#e94b5f"
        accent = "#e94b5f"
        accent_hover = "#f5728a"
        accent_fg = "#ffffff"
        tab_bg = "#2d2d3a"
        tab_selected_bg = "#38384a"
        tab_hover_bg = "#34344a"
        tab_fg = "#c8c8d4"
        placeholder_fg = "#565668"
        subtitle_fg = "#888898"
        version_fg = "#555565"
        input_bg = "#2d2d3a"
        input_fg = "#c8c8d4"
        input_border = "#3d3d4a"
        btn_bg = "#38384a"
        btn_fg = "#c8c8d4"
        btn_hover_bg = "#44445a"
        status_err = "#e05555"
    else:
        header_bg = napari_bg or "#f0f0e8"
        header_border = "#d93a54"
        accent = "#d93a54"
        accent_hover = "#c02845"
        accent_fg = "#ffffff"
        tab_bg = "#e0e0e8"
        tab_selected_bg = "#d4d4e0"
        tab_hover_bg = "#d8d8e8"
        tab_fg = "#2c2c3a"
        placeholder_fg = "#a0a0b0"
        subtitle_fg = "#505060"
        version_fg = "#909098"
        input_bg = "#e8e8f0"
        input_fg = "#2c2c3a"
        input_border = "#c0c0cc"
        btn_bg = "#d4d4e0"
        btn_fg = "#2c2c3a"
        btn_hover_bg = "#c8c8d8"
        status_err = "#b03030"

    return f"""
/* ---- Header ---- */
#confusius_header {{
    background: {header_bg};
    border-bottom: 2px solid {header_border};
}}
#confusius_title   {{ color: {accent};      background: transparent; }}
#confusius_subtitle {{ color: {subtitle_fg}; font-size: 11px; background: transparent; }}
#confusius_version  {{ color: {version_fg};  font-size: 10px; background: transparent; }}

/* ---- Accordion section headers ---- */
QPushButton#accordion_header {{
    background: {tab_bg};
    color: {tab_fg};
    border: none;
    border-radius: 0;
    padding: 8px 12px;
    font-weight: bold;
    font-size: 12px;
    text-align: left;
    margin-bottom: 0;
}}
QPushButton#accordion_header:hover:!checked {{
    background: {tab_hover_bg};
}}
QPushButton#accordion_header:checked {{
    background: {tab_selected_bg};
    color: {accent};
    border-left: 3px solid {accent};
    padding-left: 9px;
}}

/* ---- Placeholder labels ---- */
QLabel#placeholder {{
    color: {placeholder_fg};
    font-style: italic;
    font-size: 11px;
    padding: 24px 8px;
}}

/* ---- Inputs ---- */
QLineEdit {{
    background: {input_bg};
    color: {input_fg};
    border: 1px solid {input_border};
    border-radius: 3px;
    padding: 4px 6px;
}}
QComboBox {{
    background: {input_bg};
    color: {input_fg};
    border: 1px solid {input_border};
    border-radius: 3px;
    padding: 4px 6px;
}}

/* ---- Buttons ---- */
QPushButton {{
    background: {btn_bg};
    color: {btn_fg};
    border: none;
    border-radius: 3px;
    padding: 5px 10px;
}}
QPushButton:hover {{
    background: {btn_hover_bg};
}}
QPushButton#primary_btn {{
    background: {accent};
    color: {accent_fg};
    font-weight: bold;
    padding: 6px;
}}
QPushButton#primary_btn:hover {{
    background: {accent_hover};
}}
QPushButton#primary_btn:disabled {{
    background: {btn_bg};
    color: {placeholder_fg};
    font-weight: normal;
}}

/* ---- Group boxes ---- */
QGroupBox {{
    border: 1px solid {input_border};
    border-radius: 4px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-weight: bold;
    font-size: 11px;
    color: {subtitle_fg};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    top: 2px;
    left: 8px;
    padding: 0 4px;
    background: {group_title_bg};
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background: {input_border};
    border: none;
    border-radius: 2px;
    max-height: 4px;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 2px;
}}

/* ---- Status labels ---- */
QLabel#status_err {{ color: {status_err}; font-size: 11px; }}

/* ---- Tour button ---- */
QPushButton#tour_btn {{
    background: transparent;
    color: {accent};
    border: 1px solid {accent};
    border-radius: 4px;
    font-size: 11px;
    padding: 3px 8px;
}}
QPushButton#tour_btn:hover {{
    background: {accent};
    color: {accent_fg};
}}
"""


class ConfUSIusWidget(QWidget):
    """Main ConfUSIus napari plugin widget.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.setMinimumWidth(350)
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.Expanding,
        )
        self._active_tour: GuidedTour | None = None
        self._apply_theme()
        self._setup_ui()
        self.viewer.events.theme.connect(self._on_theme_changed)
        # Defer the title update so napari has time to fully configure the dock widget
        # (including installing its custom title bar).
        QTimer.singleShot(500, self._fix_dock_title)

        # Timestamp overlay — shown whenever time is being sliced.
        self._time_overlay = _TimeOverlay(self.viewer)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _is_dark(self) -> bool:
        return self.viewer.theme != "light"

    def _apply_theme(self) -> None:
        is_dark = self._is_dark()
        napari_bg: str | None = None
        try:
            from napari.utils.theme import get_theme

            t = get_theme(self.viewer.theme)
            h = t.background.as_hex()
            napari_bg = h[:7]
        except Exception:  # noqa: BLE001
            pass
        self.setStyleSheet(_build_stylesheet(is_dark, napari_bg=napari_bg))

    def _fix_dock_title(self) -> None:
        """Update the dock title to show the package version.

        Napari formats the dock name as `"{widget_display_name} ({plugin})"` and uses a
        custom `QtCustomTitleBar` whose visible `QLabel` is NOT updated by
        `QDockWidget.setWindowTitle`. We therefore look up our dock in napari's
        internal registry and update the label directly.
        """
        try:
            ver = version("confusius")
        except PackageNotFoundError:
            ver = "dev"
        title = f"ConfUSIus v{ver}"

        try:
            # napari stores all dock widgets in _wrapped_dock_widgets (on viewer.window,
            # NOT on viewer.window._qt_window).
            for dock in self.viewer.window._wrapped_dock_widgets.values():
                if not dock.isAncestorOf(self):
                    continue
                dock.setWindowTitle(title)
                # napari's QtCustomTitleBar stores the visible text in a `title`
                # attribute (a QLabel).
                tb = dock.titleBarWidget()
                if tb is not None:
                    set_text = getattr(getattr(tb, "title", None), "setText", None)
                    if callable(set_text):
                        set_text(title)
                return
        except Exception:  # noqa: BLE001
            pass

    def _on_theme_changed(self) -> None:
        self._apply_theme()
        accent = "#e94b5f" if self._is_dark() else "#d93a54"
        for btn, icon_name in getattr(self, "_accordion_btns", []):
            btn.setIcon(make_lucide_icon(icon_name, accent))

    # ------------------------------------------------------------------
    # Guided tour
    # ------------------------------------------------------------------

    def _start_tour(self) -> None:
        from confusius._napari._tour import build_default_tour

        # Ignore repeat clicks on the tour button while a tour is already
        # running so we don't spawn stacked overlays.
        if self._active_tour is not None:
            return

        tour = build_default_tour(self, is_dark=self._is_dark())
        self._active_tour = tour
        tour.finished.connect(self._on_tour_finished)
        tour.start()

    def _on_tour_finished(self) -> None:
        self._active_tour = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Wrap header + accordion in a scroll area so the sidebar dock can be
        # made arbitrarily short without forcing a tall minimum on the middle
        # band of the main window layout (which would cap how high the bottom
        # dock can grow). Content scrolls vertically when the dock is short.
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._make_header())
        content_layout.addWidget(self._make_accordion(), stretch=1)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        root.addWidget(scroll)

    def _make_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("confusius_header")

        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(2)

        tour_btn = QPushButton("Take a Tour")
        tour_btn.setObjectName("tour_btn")
        tour_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        tour_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        tour_btn.setFixedHeight(24)
        tour_btn.clicked.connect(self._start_tour)
        tour_btn.adjustSize()

        logo_widget = self._load_logo()
        logo_row = QHBoxLayout()
        logo_row.setContentsMargins(0, 0, 0, 6)
        logo_row.setSpacing(8)
        if logo_widget is not None:
            logo_row.addWidget(logo_widget)
        logo_row.addStretch()
        logo_row.addWidget(tour_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(logo_row)

        title = QLabel("ConfUSIus")
        title.setObjectName("confusius_title")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Functional Ultrasound Imaging Analysis")
        subtitle.setObjectName("confusius_subtitle")
        layout.addWidget(subtitle)

        return header

    def _load_logo(self) -> QWidget | None:
        """Return an SVG logo widget, or None if unavailable.

        Uses viewBoxF() (not defaultSize()) for the aspect ratio so that mm-unit
        SVG dimensions are handled correctly by Qt's renderer. An explicit clip
        rect on the painter prevents Qt from rendering strokes outside the image
        bounds when the SVG viewBox sits flush with the content edges.
        """
        import re

        svg_path = _ASSETS_DIR / "confusius-logo.svg"
        if not svg_path.exists():
            return None

        svg_bytes = svg_path.read_bytes()

        # The logo circle extends ~3.5 SVG units beyond the viewBox right edge
        # (content overhang) plus 1 unit for the half-stroke, so the viewBox
        # must be expanded by at least 5 units on that side. Using a symmetric
        # pad of 6 ensures all edges are covered without being too conservative.
        def _expand_viewbox(m: re.Match) -> bytes:
            x, y, w, h = (float(v) for v in m.group(1).split())
            pad = 6.0  # SVG user units (mm)
            return (
                f'viewBox="{x - pad:.4f} {y - pad:.4f} '
                f'{w + 2 * pad:.4f} {h + 2 * pad:.4f}"'
            ).encode()

        svg_bytes = re.sub(rb'viewBox="([^"]+)"', _expand_viewbox, svg_bytes, count=1)

        renderer = _QSvgRenderer()
        renderer.load(QByteArray(svg_bytes))
        if not renderer.isValid():
            return None

        target_height = 80
        vb = renderer.viewBoxF()
        aspect = vb.width() / vb.height() if vb.height() > 0 else 1.0
        target_width = round(target_height * aspect)

        dpr = QApplication.instance().devicePixelRatio()  # type: ignore[union-attr]
        px_w, px_h = round(target_width * dpr), round(target_height * dpr)

        image = QImage(px_w, px_h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setClipRect(0, 0, px_w, px_h)
        renderer.render(painter, QRectF(0, 0, px_w, px_h))
        painter.end()

        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(dpr)

        label = QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(target_width, target_height)
        label.setStyleSheet("background: transparent;")
        return label

    def _make_accordion(self) -> QWidget:
        """Build a stacked accordion where the open section fills all space."""
        from confusius._napari._data._load_panel import DataPanel
        from confusius._napari._data._save_panel import SavePanel
        from confusius._napari._qc._panel import QCPanel
        from confusius._napari._signals._panel import SignalPanel
        from confusius._napari._video._video_panel import VideoPanel

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Combined loading + saving panel.
        data_panel = QWidget()
        data_layout = QVBoxLayout(data_panel)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(0)
        data_layout.addWidget(DataPanel(self.viewer))
        data_layout.addWidget(SavePanel(self.viewer))
        data_layout.addStretch()

        # Video panel (own section).
        video_panel = VideoPanel(self.viewer)

        accent = "#e94b5f" if self._is_dark() else "#d93a54"
        tab_entries = [
            ("Data I/O", "file-input"),
            ("Video", "video"),
            ("Signals", "chart-line"),
            ("Quality Control", "clipboard-check"),
        ]
        panels = [
            data_panel,
            video_panel,
            SignalPanel(self.viewer),
            QCPanel(self.viewer),
        ]
        btns: list[QPushButton] = []

        for i, ((title, icon_name), panel) in enumerate(zip(tab_entries, panels)):
            btn = QPushButton(title)
            btn.setIcon(make_lucide_icon(icon_name, accent))
            btn.setIconSize(QSize(16, 16))
            btn.setObjectName("accordion_header")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(btn)
            # stretch=100 dominates the trailing spacer (stretch=1) so the visible panel
            # fills almost all remaining height. Hidden panels are excluded from
            # stretch allocation by Qt automatically.
            layout.addWidget(panel, stretch=100)
            panel.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            # Allow the collapse animation to shrink the panel below its
            # content's minimum size hint.
            panel.setMinimumHeight(0)
            panel.setVisible(i == 0)
            btns.append(btn)

        # Trailing spacer: takes all space when every panel is hidden so the header
        # buttons stack at the top of the accordion area.
        layout.addStretch(1)

        # Maps each panel to its in-flight animation; also keeps the Python-side
        # reference alive until the animation finishes or is replaced.
        panel_anims: dict[QWidget, QPropertyAnimation] = {}

        def _replace_panel_anim(
            panel: QWidget, anim: QPropertyAnimation | None
        ) -> None:
            previous = panel_anims.pop(panel, None)
            if previous is not None:
                previous.stop()
            if anim is not None:
                panel_anims[panel] = anim

        def _animate_panel(
            panel: QWidget, start: int, end: int, *, hide_on_done: bool
        ) -> None:
            anim = QPropertyAnimation(panel, b"maximumHeight")
            anim.setDuration(ACCORDION_ANIMATION_DURATION_MS)
            anim.setStartValue(start)
            anim.setEndValue(end)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            _replace_panel_anim(panel, anim)

            def _on_done() -> None:
                # Stale callback — this animation was replaced (stop() also
                # emits finished).
                if panel_anims.get(panel) is not anim:
                    return
                if hide_on_done:
                    panel.hide()
                panel.setMaximumHeight(QWIDGETSIZE_MAX)
                _replace_panel_anim(panel, None)

            anim.finished.connect(_on_done)
            anim.start()

        def _activate_panel(panel_index: int) -> None:
            # Clicking the already-open panel collapses it (all closed).
            already_open = panels[panel_index].isVisible()
            target = -1 if already_open else panel_index
            available_height = max(
                container.height() - sum(b.height() for b in btns), 50
            )

            for j, (button, panel) in enumerate(zip(btns, panels)):
                active = j == target
                button.blockSignals(True)
                button.setChecked(active)
                button.blockSignals(False)

                if active and not panel.isVisible():
                    panel.setMaximumHeight(0)
                    panel.show()
                    _animate_panel(panel, 0, available_height, hide_on_done=False)
                elif not active and panel.isVisible():
                    _animate_panel(panel, panel.height(), 0, hide_on_done=True)

        for i, btn in enumerate(btns):
            btn.clicked.connect(lambda _checked, i=i: _activate_panel(i))

        # Store for icon re-tinting on theme change and tour access.
        self._accordion_btns = list(zip(btns, [e[1] for e in tab_entries]))
        self._accordion_panels = dict(zip([e[0] for e in tab_entries], panels))
        # Exposed so the guided tour can follow in-flight panel animations.
        self._accordion_anims = panel_anims

        return container
