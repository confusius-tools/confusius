"""Temporal event annotation panel for the ConfUSIus sidebar."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from napari.utils.key_bindings import coerce_keybinding
from napari.utils.notifications import show_error, show_info
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._events._store import EventStore
from confusius._napari._timeaxis import (
    read_current_time,
    read_time_units,
    resolve_reference_layer,
    time_is_sliced,
)

if TYPE_CHECKING:
    import napari


class EventPanel(QWidget):
    """Sidebar panel for loading, annotating, and saving BIDS temporal events.

    The panel drives a shared `EventStore`. New events are created by entering a
    "temporal annotation" mode: the user presses Start to capture the current time
    as the onset, scrubs the napari time slider, then presses End to capture the
    offset (onset + duration). The events can be loaded from and saved to BIDS
    events `.tsv` files.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    event_store : EventStore
        Shared store of temporal events.
    """

    def __init__(self, viewer: napari.Viewer, event_store: EventStore) -> None:
        super().__init__()
        self._viewer = viewer
        self._store = event_store
        self._pending_onset: float | None = None
        self._pending_name: str = ""
        self._setup_ui()
        self._store.changed.connect(self._refresh_list)
        self._refresh_list()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self._make_annotate_group())
        layout.addWidget(self._make_list_group())
        layout.addWidget(self._make_file_group())
        layout.addWidget(self._make_display_group())
        layout.addStretch()

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def showEvent(self, a0) -> None:
        """Bind the annotation shortcut keys when the panel becomes visible.

        Parameters
        ----------
        a0 : QShowEvent
            The Qt show event.
        """
        super().showEvent(a0)
        self._bind_keys()

    def hideEvent(self, a0) -> None:
        """Unbind the annotation shortcut keys when the panel is hidden.

        Parameters
        ----------
        a0 : QHideEvent
            The Qt hide event.
        """
        super().hideEvent(a0)
        self._unbind_keys()

    _KEY_BINDINGS = ("S", "E", "Escape")
    """Viewer keys bound to Start, End, and Cancel while the panel is visible."""

    def _bind_keys(self) -> None:
        """Bind S/E/Escape in the napari viewer keymap to Start/End/Cancel.

        Uses napari key bindings rather than QShortcuts so the letters are not
        intercepted while the user is typing in a text field. Escape (rather than
        a letter) cancels, avoiding a clash with napari's own single-letter
        shortcuts.
        """
        handlers = (self._on_start, self._on_end, self._on_cancel)
        for key, handler in zip(self._KEY_BINDINGS, handlers):
            self._viewer.bind_key(
                key, lambda _viewer, handler=handler: handler(), overwrite=True
            )

    def _unbind_keys(self) -> None:
        """Remove the S/E/Escape bindings from the napari viewer keymap."""
        for key in self._KEY_BINDINGS:
            # The keymap is keyed by normalized KeyBinding objects, not strings,
            # so the key must be coerced before removal.
            self._viewer.keymap.pop(coerce_keybinding(key), None)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _make_annotate_group(self) -> QGroupBox:
        group = QGroupBox("Annotate")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Event:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("event")
        self._name_edit.setToolTip(
            "Trial type (name) for the next annotated event.\n"
            "Defaults to 'event' when left blank."
        )
        name_row.addWidget(self._name_edit, stretch=1)
        group_layout.addLayout(name_row)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start (S)")
        self._start_btn.setObjectName("primary_btn")
        self._start_btn.setToolTip(
            "Mark the onset of an event at the current time step.\n"
            "Move the time slider so a time axis is active first."
        )
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._end_btn = QPushButton("End (E)")
        self._end_btn.setToolTip("Mark the offset of the event being annotated.")
        self._end_btn.clicked.connect(self._on_end)
        self._end_btn.setEnabled(False)
        btn_row.addWidget(self._end_btn)

        self._cancel_btn = QPushButton("Cancel (Esc)")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        btn_row.addWidget(self._cancel_btn)
        group_layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        group_layout.addWidget(self._status_label)

        return group

    def _make_list_group(self) -> QGroupBox:
        group = QGroupBox("Events")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Trial type", "Onset", "End", "Duration"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        vertical_header = self._table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(120)
        group_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.clicked.connect(self._on_remove_selected)
        btn_row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)
        group_layout.addLayout(btn_row)

        return group

    def _make_file_group(self) -> QGroupBox:
        group = QGroupBox("Events File")
        group_layout = QHBoxLayout(group)
        group_layout.setSpacing(4)

        load_btn = QPushButton("Load…")
        load_btn.setToolTip("Load events from an events .tsv file.")
        load_btn.clicked.connect(self._on_load)
        group_layout.addWidget(load_btn)

        save_btn = QPushButton("Save…")
        save_btn.setToolTip("Save events to an events .tsv file.")
        save_btn.clicked.connect(self._on_save)
        group_layout.addWidget(save_btn)

        return group

    def _make_display_group(self) -> QGroupBox:
        group = QGroupBox("Display Options")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(4)

        self._shade_check = QCheckBox("Shade events on signal plot")
        self._shade_check.setChecked(self._store.shade_signals)
        self._shade_check.toggled.connect(self._store.set_shade_signals)
        group_layout.addWidget(self._shade_check)

        self._overlay_check = QCheckBox("Show active events in time overlay")
        self._overlay_check.setChecked(self._store.show_in_overlay)
        self._overlay_check.toggled.connect(self._store.set_show_in_overlay)
        group_layout.addWidget(self._overlay_check)

        return group

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _current_time(self) -> tuple[float | None, str]:
        """Return the current time value and units, falling back to seconds.

        Returns
        -------
        value : float | None
            The current time, or `None` when no time axis is sliced.
        units : str
            The time units, defaulting to ``"s"``.
        """
        if not time_is_sliced(self._viewer):
            return None, "s"
        value, units = read_current_time(self._viewer)
        return value, units or "s"

    def _on_start(self) -> None:
        value, units = self._current_time()
        if value is None:
            show_error(
                "No time axis is active. Load 4-D data and move the time slider "
                "before annotating."
            )
            return
        self._pending_onset = value
        self._pending_name = self._name_edit.text().strip() or "event"
        self._start_btn.setEnabled(False)
        self._end_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        self._name_edit.setEnabled(False)
        self._status_label.setText(
            f"Recording '{self._pending_name}' from {value:.2f} {units}…"
        )

    def _on_end(self) -> None:
        if self._pending_onset is None:
            return
        value, units = self._current_time()
        if value is None:
            show_error("No time axis is active.")
            return
        if value <= self._pending_onset:
            show_error(
                f"End time ({value:.2f} {units}) must be strictly after the start "
                f"time ({self._pending_onset:.2f} {units}). Move forward in time, "
                "then press End again."
            )
            return
        duration = value - self._pending_onset
        try:
            self._store.add_event(self._pending_onset, duration, self._pending_name)
        except ValueError as exc:
            show_error(str(exc))
            return
        self._status_label.setText(
            f"Added '{self._pending_name}' "
            f"({self._pending_onset:.2f}–{value:.2f} {units})."
        )
        self._reset_annotation()

    def _on_cancel(self) -> None:
        self._reset_annotation()
        self._status_label.setText("")

    def _reset_annotation(self) -> None:
        """Clear the in-progress annotation state and re-enable the Start button."""
        self._pending_onset = None
        self._pending_name = ""
        self._start_btn.setEnabled(True)
        self._end_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._name_edit.setEnabled(True)

    # ------------------------------------------------------------------
    # Event list
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        """Repopulate the events table from the store.

        Rows map 1:1 to `EventStore.events` order, so a row index equals its
        event index.
        """
        self._table.setRowCount(0)
        units = read_time_units(resolve_reference_layer(self._viewer)) or "s"
        for row, event in enumerate(self._store.events()):
            end = event.onset + event.duration
            self._table.insertRow(row)
            name_item = QTableWidgetItem(event.trial_type)
            name_item.setForeground(QColor(self._store.color_for(event.trial_type)))
            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, QTableWidgetItem(f"{event.onset:.2f} {units}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{end:.2f} {units}"))
            self._table.setItem(
                row, 3, QTableWidgetItem(f"{event.duration:.2f} {units}")
            )

    def _on_remove_selected(self) -> None:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        rows = {idx.row() for idx in selection_model.selectedRows()}
        if rows:
            self._store.remove_events(sorted(rows))

    def _on_clear(self) -> None:
        self._store.clear()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load Events",
            str(Path.home()),
            "Events (*.tsv);;All files (*)",
        )
        if not path_str:
            return
        try:
            loaded = self._store.load_file(Path(path_str))
        except (ValueError, OSError) as exc:
            show_error(str(exc))
            return
        show_info(f"Loaded {len(loaded)} event(s) from {Path(path_str).name}.")

    def _on_save(self) -> None:
        if not self._store.events():
            show_error("There are no events to save.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Events",
            str(Path.home() / "events.tsv"),
            "Events (*.tsv);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        if not path.suffix:
            path = path.with_suffix(".tsv")
        try:
            self._store.save_file(path)
        except OSError as exc:
            show_error(str(exc))
            return
        show_info(f"Saved {len(self._store.events())} event(s) to {path.name}.")
