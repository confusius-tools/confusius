"""Main ConfUSIus napari plugin widget."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QFont, QIcon, QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

_ASSETS_DIR = Path(__file__).parent / "assets"

try:
    from qtpy.QtSvg import QSvgRenderer as _QSvgRenderer

    _HAS_SVG = True
except ImportError:
    _HAS_SVG = False


def _make_lucide_icon(name: str, color: str, size: int = 16) -> QIcon:
    """Render a Lucide SVG icon at *size* px, tinted with *color*.

    Lucide icons use `stroke="currentColor"`. We replace that token with the desired hex
    color before passing the SVG bytes to `QSvgRenderer`.

    Parameters
    ----------
    name : str
        Stem of the SVG file inside the assets directory (e.g. `"folder-open"`).
    color : str
        CSS hex color string (e.g. `"#ffd33d"`).
    size : int, default: 16
        Target icon size in logical pixels.

    Returns
    -------
    QIcon
        A `QIcon` containing the rendered pixmap, or an empty `QIcon` when the SVG
        module is unavailable or the file does not exist.
    """
    if not _HAS_SVG:
        return QIcon()

    svg_path = _ASSETS_DIR / f"{name}.svg"
    if not svg_path.exists():
        return QIcon()

    svg_bytes = svg_path.read_bytes().replace(b"currentColor", color.encode())

    dpr = QApplication.instance().devicePixelRatio()  # type: ignore[union-attr]
    px = round(size * dpr)

    renderer = _QSvgRenderer(svg_bytes)
    image = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()

    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(dpr)
    return QIcon(pixmap)


def _build_stylesheet(is_dark: bool, napari_bg: str | None = None) -> str:  # noqa: C901
    """Return a full QSS stylesheet parametrised by theme brightness."""
    group_title_bg = napari_bg or ("#1c1c27" if is_dark else "#f0f0e8")
    if is_dark:
        header_bg = napari_bg or "#1c1c27"
        header_border = "#ffd33d"
        accent = "#ffd33d"
        accent_hover = "#ffe680"
        accent_fg = "#1c1c27"
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
        header_border = "#c49a0a"
        accent = "#c49a0a"
        accent_hover = "#d4aa1a"
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
        self._apply_theme()
        self._setup_ui()
        self.viewer.events.theme.connect(self._on_theme_changed)
        # Defer the title update so napari has time to fully configure the dock widget
        # (including installing its custom title bar).
        QTimer.singleShot(500, self._fix_dock_title)

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
                if tb is not None and hasattr(tb, "title"):
                    tb.title.setText(title)
                return
        except Exception:  # noqa: BLE001
            pass

    def _on_theme_changed(self) -> None:
        self._apply_theme()
        accent = "#ffd33d" if self._is_dark() else "#c49a0a"
        for btn, icon_name in getattr(self, "_accordion_btns", []):
            btn.setIcon(_make_lucide_icon(icon_name, accent))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_header())
        root.addWidget(self._make_accordion(), stretch=1)

    def _make_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("confusius_header")

        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(2)

        logo_widget = self._load_logo()
        if logo_widget is not None:
            logo_row = QHBoxLayout()
            logo_row.setContentsMargins(0, 0, 0, 6)
            logo_row.addWidget(logo_widget)
            logo_row.addStretch()
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

        Rendered at device pixel ratio for crisp display on HiDPI screens.
        """
        if not _HAS_SVG:
            return None

        svg_path = _ASSETS_DIR / "confusius-logo.svg"
        if not svg_path.exists():
            return None

        target_height = 80
        renderer = _QSvgRenderer(str(svg_path))
        default_size: QSize = renderer.defaultSize()
        aspect = default_size.width() / max(default_size.height(), 1)
        target_width = round(target_height * aspect)

        dpr = QApplication.instance().devicePixelRatio()  # type: ignore[union-attr]
        px_w = round(target_width * dpr)
        px_h = round(target_height * dpr)

        image = QImage(px_w, px_h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
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
        """Build a stacked accordion where the open section fills all space.

        The active section's content widget is visible (stretch=1) while all others are
        hidden. Because Qt excludes hidden widgets from layout calculations, the single
        visible content area expands to fill the full height between all the section
        headers.

        Opening and closing sections is animated via `QPropertyAnimation` on
        `maximumHeight`: the expanding panel slides in from 0 to unconstrained height;
        the collapsing panel slides out before being hidden.
        """
        from confusius._napari._data_panel import DataPanel
        from confusius._napari._qc_panel import QCPanel
        from confusius._napari._time_series_panel import TimeSeriesPanel

        container = QWidget()
        # Keep animation objects alive for their duration.
        container._anims: list[QPropertyAnimation] = []
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        accent = "#ffd33d" if self._is_dark() else "#c49a0a"
        tab_entries = [
            ("Data Loading", "file-input"),
            ("Time Series", "chart-line"),
            ("Quality Control", "clipboard-check"),
        ]
        panels = [
            DataPanel(self.viewer),
            TimeSeriesPanel(self.viewer),
            QCPanel(self.viewer),
        ]
        btns: list[QPushButton] = []

        for i, ((title, icon_name), panel) in enumerate(zip(tab_entries, panels)):
            btn = QPushButton(title)
            btn.setIcon(_make_lucide_icon(icon_name, accent))
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
            panel.setVisible(i == 0)
            btns.append(btn)

        # Trailing spacer: takes all space when every panel is hidden so the header
        # buttons stack at the top of the accordion area.
        layout.addStretch(1)

        _DURATION = 200  # ms

        def _drop_anim(a: QPropertyAnimation) -> None:
            try:
                container._anims.remove(a)
            except ValueError:
                pass

        def _activate(idx: int) -> None:
            # Clicking the already-open panel collapses it (all closed).
            already_open = panels[idx].isVisible()
            target = -1 if already_open else idx  # -1 means all collapsed

            # Available height shared between panels during animation.
            available_h = max(container.height() - sum(b.height() for b in btns), 50)

            for j, (b, p) in enumerate(zip(btns, panels)):
                active = j == target
                b.blockSignals(True)
                b.setChecked(active)
                b.blockSignals(False)

                if active and not p.isVisible():
                    # Expand: 0 → available_h
                    p.setMaximumHeight(0)
                    p.show()
                    a = QPropertyAnimation(p, b"maximumHeight")
                    a.setDuration(_DURATION)
                    a.setStartValue(0)
                    a.setEndValue(available_h)
                    a.setEasingCurve(QEasingCurve.Type.InOutCubic)

                    def _on_expand_done(
                        panel: QWidget = p, anim: QPropertyAnimation = a
                    ) -> None:
                        panel.setMaximumHeight(16777215)
                        _drop_anim(anim)

                    a.finished.connect(_on_expand_done)
                    container._anims.append(a)
                    a.start()

                elif not active and p.isVisible():
                    # Collapse: current height → 0; when done, if all panels are hidden
                    # constrain the container to button heights so Qt stops distributing
                    # extra space between the buttons.
                    a = QPropertyAnimation(p, b"maximumHeight")
                    a.setDuration(_DURATION)
                    a.setStartValue(p.height())
                    a.setEndValue(0)
                    a.setEasingCurve(QEasingCurve.Type.InOutCubic)

                    def _on_collapse_done(
                        panel: QWidget = p, anim: QPropertyAnimation = a
                    ) -> None:
                        panel.hide()
                        panel.setMaximumHeight(16777215)
                        _drop_anim(anim)

                    a.finished.connect(_on_collapse_done)
                    container._anims.append(a)
                    a.start()

        for i, btn in enumerate(btns):
            btn.clicked.connect(lambda _checked, i=i: _activate(i))

        # Store for icon re-tinting on theme change.
        self._accordion_btns = list(zip(btns, [e[1] for e in tab_entries]))

        return container
