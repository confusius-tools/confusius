"""Floating manager window for imported signals."""

from __future__ import annotations

from pathlib import Path

from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._signals._store import LiveSignal, SignalStore
from confusius._napari._theme import get_napari_colors

_SOURCE_LABELS = {
    "mouse": "Mouse",
    "point": "Points layer",
    "label": "Labels layer",
}


def _colored_button_style(r: int, g: int, b: int) -> str:
    """Generate a stylesheet for a colored push button.

    Parameters
    ----------
    r : int
        Red component (0-255).
    g : int
        Green component (0-255).
    b : int
        Blue component (0-255).

    Returns
    -------
    str
        Stylesheet string with normal and hover states.
    """
    base_color = f"rgba({r}, {g}, {b}, 0.25)"
    hover_color = f"rgba({r}, {g}, {b}, 0.45)"
    return (
        f"QPushButton {{ background-color: {base_color}; }}"
        f"QPushButton:hover {{ background-color: {hover_color}; }}"
    )


class SignalsManagerDialog(QDialog):
    """Modeless dialog for managing live and imported signals.

    Live signals (from mouse, points, or labels layers) appear at the top of the table.
    They can be renamed, recolored, and hidden but not removed. Imported signals appear
    below and support the full set of operations.

    Parameters
    ----------
    store : SignalStore
        Shared imported-signals store.
    parent : QWidget | None, optional
        Optional parent widget.
    """

    def __init__(self, store: SignalStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._updating_table = False
        self.setWindowTitle("Signal Manager")
        self.setModal(False)
        self.resize(720, 360)
        self._setup_ui()
        self._store.changed.connect(self._refresh_table)
        self._refresh_table()

    def _setup_ui(self) -> None:
        """Build the manager dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        button_row = QHBoxLayout()
        self._import_btn = QPushButton("Import")
        self._import_btn.setToolTip("Import a CSV or TSV file")
        self._import_btn.setStyleSheet(_colored_button_style(80, 160, 80))
        self._import_btn.clicked.connect(self._import_file)
        button_row.addWidget(self._import_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setToolTip("Remove selected imported signal")
        self._remove_btn.setStyleSheet(_colored_button_style(200, 80, 80))
        self._remove_btn.clicked.connect(self._remove_selected)
        button_row.addWidget(self._remove_btn)

        self._clear_btn = QPushButton("Clear All Imported")
        self._clear_btn.setToolTip("Remove all imported signals")
        self._clear_btn.setStyleSheet(_colored_button_style(200, 80, 80))
        self._clear_btn.clicked.connect(self._store.clear)
        button_row.addWidget(self._clear_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Visible", "Name", "Color", "Source"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._table.setShowGrid(True)
        self._table.setAlternatingRowColors(True)
        vertical_header = self._table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
            vertical_header.setDefaultSectionSize(26)
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            header.setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table)

    # -- Table refresh --------------------------------------------------------

    def _refresh_table(self) -> None:
        """Refresh the table from the shared store."""
        self._updating_table = True
        try:
            live_list = self._store.live_signals()
            imported_list = self._store.imported_signals()
            total = len(live_list) + len(imported_list)
            self._table.setRowCount(total)

            row = 0
            for signal in live_list:
                self._set_live_row(row, signal)
                row += 1
            for signal in imported_list:
                self._set_imported_row(row, signal)
                row += 1
        finally:
            self._updating_table = False

    def _set_live_row(self, row: int, signal: LiveSignal) -> None:
        """Populate one table row from a live signal."""
        visible_item = QTableWidgetItem()
        visible_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
        )
        visible_item.setCheckState(
            Qt.CheckState.Checked if signal.visible else Qt.CheckState.Unchecked
        )
        visible_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        self._table.setItem(row, 0, visible_item)

        name_item = QTableWidgetItem(signal.name)
        name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
        name_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        self._table.setItem(row, 1, name_item)

        color_btn = QPushButton()
        color_btn.setToolTip("Click to change color")
        color_btn.setStyleSheet(
            f"background-color: {signal.color};"
            "border: 1px solid rgba(128, 128, 128, 0.6);"
            "border-radius: 3px;"
            "margin: 3px 6px;"
        )
        color_btn.clicked.connect(
            lambda _checked=False, sid=signal.id: self._choose_color(sid)
        )
        self._table.setCellWidget(row, 2, color_btn)

        source_label = _SOURCE_LABELS.get(signal.source_type, signal.source_type)
        source_item = QTableWidgetItem(source_label)
        source_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        source_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        italic_font = QFont()
        italic_font.setItalic(True)
        source_item.setFont(italic_font)
        self._table.setItem(row, 3, source_item)

    def _set_imported_row(self, row: int, signal) -> None:
        """Populate one table row from an imported signal."""
        visible_item = QTableWidgetItem()
        visible_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        visible_item.setCheckState(
            Qt.CheckState.Checked if signal.visible else Qt.CheckState.Unchecked
        )
        visible_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        self._table.setItem(row, 0, visible_item)

        name_item = QTableWidgetItem(signal.name)
        name_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )
        name_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        self._table.setItem(row, 1, name_item)

        color_btn = QPushButton()
        color_btn.setToolTip("Click to change color")
        color_btn.setStyleSheet(
            f"background-color: {signal.color};"
            "border: 1px solid rgba(128, 128, 128, 0.6);"
            "border-radius: 3px;"
            "margin: 3px 6px;"
        )
        color_btn.clicked.connect(
            lambda _checked=False, sid=signal.id: self._choose_color(sid)
        )
        self._table.setCellWidget(row, 2, color_btn)

        source_item = QTableWidgetItem(signal.source_label)
        source_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        source_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        self._table.setItem(row, 3, source_item)

    # -- Item edit callbacks --------------------------------------------------

    def _is_live_id(self, signal_id: str) -> bool:
        """Return whether a signal ID belongs to a live signal."""
        return signal_id.startswith(("mouse-", "point-", "label-"))

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Apply visibility or name edits back to the store."""
        if self._updating_table:
            return

        signal_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(signal_id, str):
            return

        is_live = self._is_live_id(signal_id)

        try:
            if item.column() == 0:
                visible = item.checkState() == Qt.CheckState.Checked
                if is_live:
                    self._store.set_live_signal_visible(signal_id, visible)
                else:
                    self._store.set_signal_visible(signal_id, visible)
            elif item.column() == 1:
                if is_live:
                    self._store.rename_live_signal(signal_id, item.text())
                else:
                    self._store.rename_signal(signal_id, item.text())
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
            self._refresh_table()

    def _choose_color(self, signal_id: str) -> None:
        """Open a color picker for a signal."""
        is_live = self._is_live_id(signal_id)

        if is_live:
            live = self._store.get_live_signal(signal_id)
            current = live.color if live is not None else "#ffffff"
        else:
            current = next(
                (s.color for s in self._store.imported_signals() if s.id == signal_id),
                "#ffffff",
            )

        color = QColorDialog.getColor(QColor(current), self, "Choose Signal Color")
        if not color.isValid():
            return

        try:
            if is_live:
                self._store.set_live_signal_color(signal_id, color.name())
            else:
                self._store.set_signal_color(signal_id, color.name())
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))

    def _remove_selected(self) -> None:
        """Remove the currently selected imported signal (live signals are skipped)."""
        signal_ids = []
        for index in self._table.selectionModel().selectedRows():
            item = self._table.item(index.row(), 1)
            if item is None:
                continue
            signal_id = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(signal_id, str) and not self._is_live_id(signal_id):
                signal_ids.append(signal_id)
        self._store.remove_signals(signal_ids)

    def _import_file(self) -> None:
        """Import one CSV or TSV file into the shared store."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Import Signal",
            "",
            "Delimited text (*.tsv *.csv);;TSV (*.tsv);;CSV (*.csv);;All files (*)",
        )
        if not path_str:
            return

        try:
            imported = self._store.import_file(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
            return

        count = len(imported)
        noun = "signal" if count != 1 else "signals"
        show_info(f"Imported {count} {noun} from {Path(path_str).name}")

    def apply_theme(self, theme_name: str) -> None:
        """Apply napari theme to the table for proper alternating row colors.

        Parameters
        ----------
        theme_name : str
            Name of the napari theme (e.g. `"dark"` or `"light"`).
        """
        colors = get_napari_colors(theme_name)
        bg = colors["bg"]
        fg = colors["fg"]

        is_dark = colors["is_dark"]
        if is_dark:
            alt_bg = "#38384a"
        else:
            alt_bg = "#e8e8f0"

        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {bg};
                color: {fg};
                alternate-background-color: {alt_bg};
                border: none;
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QHeaderView::section {{
                background-color: {bg};
                color: {fg};
                padding: 5px;
                border: 1px solid {alt_bg};
            }}
        """)
