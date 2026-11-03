"""Floating manager window for stored signals."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._export import (
    ExportSignal,
    prompt_delimited_export_path,
    write_delimited_signals,
)
from confusius._napari._signals._store import LiveSignal, SignalStore
from confusius._napari._theme import get_napari_colors

if TYPE_CHECKING:
    from confusius._napari._signals._live import LiveSources

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
    """Modeless dialog for managing live and stored signals.

    Live signals (from mouse, points, or labels layers) appear at the top of the table.
    They can be renamed, recolored, and hidden but not removed. Stored signals — either
    imported from a file or pinned from a live signal — appear below and support the
    full set of operations.

    Parameters
    ----------
    store : SignalStore
        Shared stored-signals store.
    parent : QWidget | None, optional
        Optional parent widget.
    live_sources : LiveSources, optional
        Shared live controller for per-ROI layer enablement.
    """

    def __init__(
        self,
        store: SignalStore,
        parent: QWidget | None = None,
        live_sources: LiveSources | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._live_sources = live_sources
        self._updating_table = False
        self.setWindowTitle("Signal Manager")
        self.setModal(False)
        self.resize(720, 360)
        self._setup_ui()
        self._store.changed.connect(self._refresh_table)
        if live_sources is not None:
            live_sources.sources_changed.connect(self._refresh_sources)
            self._refresh_sources()
        self._refresh_table()

    def _setup_ui(self) -> None:
        """Build the manager dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if self._live_sources is not None:
            self._sources_label = QLabel("ROI layers")
            layout.addWidget(self._sources_label)
            self._sources_list = QListWidget()
            self._sources_list.setToolTip(
                "Enable individual ROI layers. The Signals tab toggles all layers of a type."
            )
            self._sources_list.setMaximumHeight(110)
            self._sources_list.itemChanged.connect(self._on_source_enabled)
            layout.addWidget(self._sources_list)

        button_row = QHBoxLayout()
        self._import_btn = QPushButton("Import")
        self._import_btn.setToolTip("Import a CSV or TSV file")
        self._import_btn.setStyleSheet(_colored_button_style(80, 160, 80))
        self._import_btn.clicked.connect(self._import_file)
        button_row.addWidget(self._import_btn)

        self._export_btn = QPushButton("Export Selected")
        self._export_btn.setToolTip(
            "Export selected stored signals to a CSV or TSV file"
        )
        self._export_btn.setStyleSheet(_colored_button_style(80, 120, 200))
        self._export_btn.clicked.connect(self._export_selected)
        button_row.addWidget(self._export_btn)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setToolTip("Remove selected stored signal")
        self._remove_btn.setStyleSheet(_colored_button_style(200, 80, 80))
        self._remove_btn.clicked.connect(self._remove_selected)
        button_row.addWidget(self._remove_btn)

        button_row.addStretch()
        layout.addLayout(button_row)

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Visible", "Name", "Color", "Source", "Sync color", "Line style"]
        )
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
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table)

    def _refresh_sources(self) -> None:
        """Refresh per-ROI layer enablement controls."""
        controller = self._live_sources
        if controller is None:
            return
        self._sources_list.blockSignals(True)
        self._sources_list.clear()
        for source in controller.sources():
            if source.layer is None:
                continue
            item = QListWidgetItem(source.layer.name)
            item.setData(Qt.ItemDataRole.UserRole, source.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if source.enabled else Qt.CheckState.Unchecked
            )
            self._sources_list.addItem(item)
        self._sources_list.blockSignals(False)
        has_rois = self._sources_list.count() > 0
        self._sources_list.setVisible(has_rois)
        self._sources_label.setVisible(has_rois)

    def _on_source_enabled(self, item: QListWidgetItem) -> None:
        """Apply source enablement without deleting its traces.

        Parameters
        ----------
        item : qtpy.QtWidgets.QListWidgetItem
            Changed source row.
        """
        if self._live_sources is not None:
            self._live_sources.set_enabled(
                item.data(Qt.ItemDataRole.UserRole),
                item.checkState() == Qt.CheckState.Checked,
            )

    # -- Table refresh --------------------------------------------------------

    def _refresh_table(self) -> None:
        """Refresh the table from the shared store."""
        self._updating_table = True
        try:
            live_list = self._store.live_signals()
            stored_list = self._store.stored_signals()
            total = len(live_list) + len(stored_list)
            self._table.setRowCount(total)

            row = 0
            for signal in live_list:
                self._set_live_row(row, signal)
                row += 1
            for signal in stored_list:
                self._set_stored_row(row, signal)
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

        source_label = signal.default_name or _SOURCE_LABELS.get(
            signal.source_type, signal.source_type
        )
        name_item.setToolTip(signal.display_name)
        if not signal.enabled or not signal.available:
            source_label += " (unavailable)"
        source_item = QTableWidgetItem(source_label)
        source_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        source_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        italic_font = QFont()
        italic_font.setItalic(True)
        source_item.setFont(italic_font)
        self._table.setItem(row, 3, source_item)
        sync_item = QTableWidgetItem()
        sync_item.setData(Qt.ItemDataRole.UserRole, signal.id)
        if signal.backing_layer is not None:
            sync_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
            )
            sync_item.setCheckState(
                Qt.CheckState.Unchecked
                if signal.color_overridden
                else Qt.CheckState.Checked
            )
            sync_item.setToolTip(
                "Follow the point/label colour. Choosing a custom colour disables synchronization."
            )
        else:
            sync_item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._table.setItem(row, 4, sync_item)
        self._set_style_control(row, signal.id, signal.line_style)

    def _set_stored_row(self, row: int, signal) -> None:
        """Populate one table row from a stored signal."""
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
        sync_item = QTableWidgetItem()
        sync_item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._table.setItem(row, 4, sync_item)
        self._set_style_control(row, signal.id, signal.line_style)

    def _set_style_control(self, row: int, signal_id: str, style: str) -> None:
        """Populate the line-style selector for a live or stored trace.

        Parameters
        ----------
        row : int
            Table row.
        signal_id : str
            Trace identifier.
        style : str
            Current matplotlib line style.
        """
        combo = QComboBox()
        for name, value in (
            ("Solid", "-"),
            ("Dashed", "--"),
            ("Dash-dot", "-."),
            ("Dotted", ":"),
        ):
            combo.addItem(name, value)
        combo.setCurrentIndex(combo.findData(style))
        combo.currentIndexChanged.connect(
            lambda: self._store.set_signal_line_style(signal_id, combo.currentData())
        )
        self._table.setCellWidget(row, 5, combo)

    # -- Item edit callbacks --------------------------------------------------

    def _is_live_id(self, signal_id: str) -> bool:
        """Return whether a signal ID belongs to a live signal."""
        return self._store.get_live_signal(signal_id) is not None

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
            elif item.column() == 4 and is_live:
                self._store.set_live_signal_color_sync(
                    signal_id, item.checkState() == Qt.CheckState.Checked
                )
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
                (s.color for s in self._store.stored_signals() if s.id == signal_id),
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

    def _selected_stored_ids(self) -> list[str]:
        """Return ids of the currently selected stored (non-live) rows.

        Live rows are skipped: the manager only tracks their presentation
        metadata, not the underlying data, so they can't be removed or exported
        from here.
        """
        signal_ids = []
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return signal_ids
        rows = [index.row() for index in selection_model.selectedRows()]
        if not rows and self._table.currentRow() >= 0:
            rows = [self._table.currentRow()]
        for row in rows:
            item = self._table.item(row, 1)
            if item is None:
                continue
            signal_id = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(signal_id, str) and not self._is_live_id(signal_id):
                signal_ids.append(signal_id)
        return signal_ids

    def _remove_selected(self) -> None:
        """Remove the currently selected stored signal (live signals are skipped)."""
        self._store.remove_signals(self._selected_stored_ids())

    def _export_selected(self) -> None:
        """Export the currently selected stored signals to a CSV or TSV file."""
        signal_ids = self._selected_stored_ids()
        stored_by_id = {s.id: s for s in self._store.stored_signals()}
        export_signals = [
            ExportSignal(
                stored_by_id[sid].name, stored_by_id[sid].x, stored_by_id[sid].y
            )
            for sid in signal_ids
            if sid in stored_by_id
        ]
        if not export_signals:
            show_error("Select one or more stored signals to export.")
            return

        export_selection = prompt_delimited_export_path(
            self, "Export Signals", str(Path.home() / "signals.tsv")
        )
        if export_selection is None:
            return
        path, delimiter = export_selection

        try:
            write_delimited_signals(path, export_signals, delimiter=delimiter)
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
            return

        show_info(f"Exported signals to {path}")

    def _import_file(self) -> None:
        """Import one or more CSV or TSV files into the shared store."""
        path_strs, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Signal",
            "",
            "Delimited text (*.tsv *.tsv.gz *.csv);;TSV (*.tsv *.tsv.gz);;CSV (*.csv);;All files (*)",
        )
        if not path_strs:
            return

        imported_count = 0
        file_count = 0
        for path_str in path_strs:
            try:
                imported = self._store.import_file(Path(path_str))
            except Exception as exc:  # noqa: BLE001
                show_error(str(exc))
                return
            imported_count += len(imported)
            file_count += 1

        signal_noun = "signal" if imported_count == 1 else "signals"
        file_noun = "file" if file_count == 1 else "files"
        show_info(
            f"Imported {imported_count} {signal_noun} from {file_count} {file_noun}."
        )

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
