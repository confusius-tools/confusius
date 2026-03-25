"""Floating manager window for imported time series."""

from __future__ import annotations

from pathlib import Path

from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QDialog,
    QAbstractItemView,
    QHeaderView,
)

from confusius._napari._theme import get_napari_colors
from confusius._napari._time_series._store import ImportedSeries, TimeSeriesStore


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


class TimeSeriesManagerDialog(QDialog):
    """Modeless dialog for managing imported time series.

    Parameters
    ----------
    store : TimeSeriesStore
        Shared imported-series store.
    parent : QWidget | None, optional
        Optional parent widget.
    """

    def __init__(self, store: TimeSeriesStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._updating_table = False
        self.setWindowTitle("Time Series Manager")
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
        self._import_btn = QPushButton("Import...")
        self._import_btn.setToolTip("Import a CSV or TSV file")
        self._import_btn.setStyleSheet(_colored_button_style(80, 160, 80))
        self._import_btn.clicked.connect(self._import_file)
        button_row.addWidget(self._import_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setToolTip("Remove selected series")
        self._remove_btn.setStyleSheet(_colored_button_style(200, 80, 80))
        self._remove_btn.clicked.connect(self._remove_selected)
        button_row.addWidget(self._remove_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setToolTip("Remove all imported series")
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

    def _refresh_table(self) -> None:
        """Refresh the table from the shared store."""
        self._updating_table = True
        try:
            series_list = self._store.imported_series()
            self._table.setRowCount(len(series_list))
            for row, series in enumerate(series_list):
                self._set_row(row, series)
        finally:
            self._updating_table = False

    def _set_row(self, row: int, series: ImportedSeries) -> None:
        """Populate one table row from one imported series."""
        visible_item = QTableWidgetItem()
        visible_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        visible_item.setCheckState(
            Qt.CheckState.Checked if series.visible else Qt.CheckState.Unchecked
        )
        visible_item.setData(Qt.ItemDataRole.UserRole, series.id)
        self._table.setItem(row, 0, visible_item)

        name_item = QTableWidgetItem(series.name)
        name_item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
        )
        name_item.setData(Qt.ItemDataRole.UserRole, series.id)
        self._table.setItem(row, 1, name_item)

        color_btn = QPushButton()
        color_btn.setToolTip("Click to change color")
        color_btn.setStyleSheet(
            f"background-color: {series.color};"
            "border: 1px solid rgba(128, 128, 128, 0.6);"
            "border-radius: 3px;"
            "margin: 3px 6px;"
        )
        color_btn.clicked.connect(
            lambda _checked=False, series_id=series.id: self._choose_color(series_id)
        )
        self._table.setCellWidget(row, 2, color_btn)

        source_item = QTableWidgetItem(series.source_label)
        source_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        source_item.setData(Qt.ItemDataRole.UserRole, series.id)
        self._table.setItem(row, 3, source_item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Apply visibility or name edits back to the store."""
        if self._updating_table:
            return

        series_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(series_id, str):
            return

        try:
            if item.column() == 0:
                self._store.set_series_visible(
                    series_id,
                    item.checkState() == Qt.CheckState.Checked,
                )
            elif item.column() == 1:
                self._store.rename_series(series_id, item.text())
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
            self._refresh_table()

    def _choose_color(self, series_id: str) -> None:
        """Open a color picker for one imported series."""
        current = next(
            (
                series.color
                for series in self._store.imported_series()
                if series.id == series_id
            ),
            "#ffffff",
        )
        color = QColorDialog.getColor(QColor(current), self, "Choose Series Color")
        if not color.isValid():
            return

        try:
            self._store.set_series_color(series_id, color.name())
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))

    def _remove_selected(self) -> None:
        """Remove the currently selected imported series."""
        series_ids = []
        for index in self._table.selectionModel().selectedRows():
            item = self._table.item(index.row(), 1)
            if item is None:
                continue
            series_id = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(series_id, str):
                series_ids.append(series_id)
        self._store.remove_series(series_ids)

    def _import_file(self) -> None:
        """Import one CSV or TSV file into the shared store."""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Import Time Series",
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
        noun = "series" if count != 1 else "time series"
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
