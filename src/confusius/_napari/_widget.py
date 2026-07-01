"""Main ConfUSIus napari plugin widget."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QByteArray, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer as _QSvgRenderer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._theme import make_lucide_icon
from confusius._napari._time_overlay import _TimeOverlay

if TYPE_CHECKING:
    import napari

    from confusius._napari._tour import GuidedTour

_ASSETS_DIR = Path(__file__).parent / "assets"


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
#confusius_title   {{
    color: {accent};
    font-size: 14px;
    font-weight: bold;
    background: transparent;
    padding-left: 0px;
}}
#confusius_subtitle {{
    color: {subtitle_fg};
    font-size: 11px;
    background: transparent;
    padding-left: 0px;
    padding-top: 0px;
}}
#confusius_version  {{ color: {version_fg};  font-size: 10px; background: transparent; }}

/* ---- Section selector ---- */
QComboBox#section_selector {{
    background: {tab_selected_bg};
    color: {accent};
    border: none;
    border-left: 3px solid {accent};
    border-radius: 0;
    padding: 8px 12px 8px 9px;
    font-weight: bold;
    font-size: 12px;
}}
QComboBox#section_selector:hover {{
    background: {tab_hover_bg};
}}
QComboBox#section_selector::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox#section_selector QAbstractItemView {{
    background: {tab_bg};
    color: {tab_fg};
    border: 1px solid {input_border};
    selection-background-color: {tab_selected_bg};
    selection-color: {accent};
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
    font-size: 10px;
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
        self._refresh_section_icons()

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
        layout.setContentsMargins(4, 6, 12, 14)
        layout.setSpacing(2)

        tour_btn = QPushButton("Take a Tour")
        tour_btn.setObjectName("tour_btn")
        tour_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        tour_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        tour_btn.clicked.connect(self._start_tour)
        tour_btn.adjustSize()

        logo_widget = self._load_logo()
        logo_row = QHBoxLayout()
        logo_row.setContentsMargins(0, 0, 0, 6)
        logo_row.setSpacing(10)
        if logo_widget is not None:
            logo_row.addWidget(logo_widget)

        title = QLabel("ConfUSIus")
        title.setObjectName("confusius_title")
        title.setIndent(0)

        subtitle = QLabel("Functional Ultrasound Imaging Analysis")
        subtitle.setObjectName("confusius_subtitle")
        subtitle.setIndent(0)

        tour_btn_title_and_subtitle = QWidget()
        tour_btn_title_and_subtitle_layout = QVBoxLayout(tour_btn_title_and_subtitle)
        tour_btn_title_and_subtitle_layout.setContentsMargins(0, 0, 0, 0)
        tour_btn_title_and_subtitle_layout.setSpacing(0)

        tour_btn_title_and_subtitle_layout.addStretch()
        tour_btn_title_and_subtitle_layout.addWidget(
            tour_btn, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
        )
        tour_btn_title_and_subtitle_layout.addWidget(
            title, alignment=Qt.AlignmentFlag.AlignLeft
        )
        tour_btn_title_and_subtitle_layout.addWidget(
            subtitle, alignment=Qt.AlignmentFlag.AlignLeft
        )

        logo_row.addWidget(tour_btn_title_and_subtitle)
        layout.addLayout(logo_row)

        return header

    def _load_logo(self) -> QWidget | None:
        """Build the header logo as a fixed-size QLabel.

        Returns
        -------
        QWidget or None
            A QLabel displaying the logo, or None if the SVG asset is missing or
            the renderer fails to load it.
        """
        import re

        svg_path = _ASSETS_DIR / "confusius-logo.svg"
        if not svg_path.exists():
            return None

        svg_bytes = svg_path.read_bytes()

        # QtSvg clips all drawing to the SVG viewBox. This logo's outline stroke is
        # centred on a path that runs flush with the viewBox edges, so the outer
        # half of the stroke falls outside the viewBox and is clipped — shaving the
        # right and bottom contour off the rendered logo. Enlarging the viewBox
        # pulls the whole stroke back inside and leaves a thin transparent margin,
        # which also keeps the artwork off the rasterised image edge (where it would
        # otherwise be shaved on sub-pixel boundaries).
        #
        # The pad is a fraction of the viewBox, not an absolute number of units, so
        # the breathing room scales with the artwork and never needs retuning when
        # `target_height` changes.
        def _expand_viewbox(m: re.Match) -> bytes:
            x, y, w, h = (float(v) for v in m.group(1).split())
            pad = 0.05 * max(w, h)  # 5% of the viewBox, in SVG user units
            return (
                f'viewBox="{x - pad:.4f} {y - pad:.4f} '
                f'{w + 2 * pad:.4f} {h + 2 * pad:.4f}"'
            ).encode()

        svg_bytes = re.sub(rb'viewBox="([^"]+)"', _expand_viewbox, svg_bytes, count=1)

        renderer = _QSvgRenderer()
        renderer.load(QByteArray(svg_bytes))
        if not renderer.isValid():
            return None

        target_height = 50
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

    def _refresh_section_icons(self) -> None:
        """Retint the section selector icons for the active napari theme."""
        selector = getattr(self, "_section_selector", None)
        entries = getattr(self, "_section_entries", [])
        if not isinstance(selector, QComboBox):
            return
        accent = "#e94b5f" if self._is_dark() else "#d93a54"
        for i, (_title, icon_name) in enumerate(entries):
            selector.setItemIcon(i, make_lucide_icon(icon_name, accent))

    def _set_active_section(self, label: str) -> None:
        """Show the requested top-level section."""
        labels = [title for title, _icon_name in getattr(self, "_section_entries", [])]
        try:
            index = labels.index(label)
        except ValueError:
            return

        selector = getattr(self, "_section_selector", None)
        stack = getattr(self, "_section_stack", None)
        if not isinstance(selector, QComboBox) or not isinstance(stack, QStackedWidget):
            return

        if selector.currentIndex() != index:
            selector.blockSignals(True)
            selector.setCurrentIndex(index)
            selector.blockSignals(False)
        stack.setCurrentIndex(index)

    def _make_accordion(self) -> QWidget:
        """Build a top dropdown plus one visible section panel."""
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

        selector = QComboBox()
        selector.setObjectName("section_selector")
        selector.setIconSize(QSize(16, 16))
        selector.setMinimumHeight(36)
        selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for title, _icon_name in tab_entries:
            selector.addItem(title)
        layout.addWidget(selector)

        stack = QStackedWidget()
        stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        for panel in panels:
            panel.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            stack.addWidget(panel)
        layout.addWidget(stack, stretch=1)

        self._section_entries = tab_entries
        self._section_selector = selector
        self._section_stack = stack
        self._accordion_btns = []
        self._accordion_panels = dict(zip([e[0] for e in tab_entries], panels))
        self._accordion_anims = {}

        selector.currentTextChanged.connect(self._set_active_section)
        self._refresh_section_icons()
        self._set_active_section(tab_entries[0][0])

        return container
