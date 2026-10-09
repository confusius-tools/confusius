"""Atlas panel for the ConfUSIus napari plugin."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import xarray as xr
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._atlas._tree import StructureTreeWidget
from confusius._napari._qt import install_no_scroll_wheel_filter
from confusius.datasets import (
    fetch_brainglobe_atlas,
    fetch_template_huang_2025,
    fetch_template_pepe_mariani_2026,
)
from confusius.plotting.napari import plot_napari

if TYPE_CHECKING:
    import napari
    from qtpy.QtGui import QShowEvent
    from qtpy.QtWidgets import QDockWidget

TEMPLATES: dict[str, Callable[..., xr.DataArray]] = {
    "Huang 2025 mouse vascular": fetch_template_huang_2025,
    "Pepe-Mariani 2026 mouse vascular": fetch_template_pepe_mariani_2026,
}
"""fUSI templates offered in the template dropdown, display label to fetcher."""

MASK_SIDES = ("both", "left", "right")
"""Hemisphere choices offered for region masks, in dropdown order."""

SEARCH_FIELDS = ("all", "acronym", "name")
"""Search field choices, matching `AtlasAccessor.search`'s `field` argument."""


@thread_worker
def _list_atlases() -> tuple[list[str], list[str] | None]:
    """List BrainGlobe atlases in a background thread.

    Returns
    -------
    downloaded : list[str]
        Atlas names already present in the BrainGlobe cache.
    available : list[str] or None
        Every atlas name BrainGlobe knows about, sorted, or `None` when the list
        could not be fetched and BrainGlobe has no cached copy of it.
    """
    from brainglobe_atlasapi.list_atlases import (
        get_all_atlases_lastversions,
        get_downloaded_atlases,
    )

    downloaded = get_downloaded_atlases()
    try:
        available = sorted(get_all_atlases_lastversions())
    except Exception:  # noqa: BLE001
        # Offline with no cached `last_versions.conf`: BrainGlobe raises instead of
        # returning an empty mapping.
        available = None
    return downloaded, available


@thread_worker
def _fetch_atlas(atlas_name: str) -> xr.Dataset:
    """Fetch a BrainGlobe atlas and load it into memory in a background thread.

    Parameters
    ----------
    atlas_name : str
        BrainGlobe atlas name.

    Returns
    -------
    xarray.Dataset
        In-memory atlas Dataset.
    """
    # The fetched Dataset is lazy zarr; the three layers and `get_masks` all need the
    # full arrays, so compute once here rather than on each GUI-thread access.
    # `fetch_brainglobe_atlas` accepts any atlas name string despite its `AtlasName`
    # Literal annotation, so new BrainGlobe atlases work without a stub bump.
    return fetch_brainglobe_atlas(atlas_name).compute()  # ty: ignore[invalid-argument-type]


@thread_worker
def _fetch_template(fetch: Callable[..., xr.DataArray]) -> xr.DataArray:
    """Fetch a fUSI template in a background thread.

    Parameters
    ----------
    fetch : callable
        One of the `confusius.datasets.fetch_template_*` functions.

    Returns
    -------
    xarray.DataArray
        Template VoxelData array.
    """
    return fetch(print_citation=False)


@thread_worker
def _compute_masks(atlas: xr.Dataset, region_ids: list[int], side: str) -> xr.DataArray:
    """Compute region masks in a background thread.

    Parameters
    ----------
    atlas : xarray.Dataset
        Loaded atlas Dataset.
    region_ids : list[int]
        Structure ids to mask.
    side : {"both", "left", "right"}
        Hemisphere applied to every region.

    Returns
    -------
    xarray.DataArray
        Masks stacked along a `mask` dimension, as returned by
        [`get_masks`][confusius.atlas.AtlasAccessor.get_masks].
    """
    return atlas.atlas.get_masks(region_ids, sides=side)  # type: ignore[arg-type]


def _make_progress_bar() -> QProgressBar:
    """Build the thin indeterminate progress bar used across the plugin.

    Returns
    -------
    qtpy.QtWidgets.QProgressBar
        Hidden, indeterminate, 4px-high progress bar.
    """
    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setMaximumHeight(4)
    progress.hide()
    return progress


def _make_region_table() -> QTableWidget:
    """Build a read-only `id` / `acronym` / `name` region table.

    Returns
    -------
    qtpy.QtWidgets.QTableWidget
        Empty table with row selection enabled.
    """
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["id", "acronym", "name"])
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)
    header = table.horizontalHeader()
    if header is not None:
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    table.setMinimumHeight(140)
    return table


def _set_table_rows(table: QTableWidget, rows: Iterable[tuple[int, str, str]]) -> None:
    """Replace every row of a region table.

    Parameters
    ----------
    table : qtpy.QtWidgets.QTableWidget
        Table built by `_make_region_table`.
    rows : iterable of (int, str, str)
        `(id, acronym, name)` triples, one per row.
    """
    table.setRowCount(0)
    for region_id, acronym, name in rows:
        _append_table_row(table, region_id, acronym, name)


def _append_table_row(
    table: QTableWidget, region_id: int, acronym: str, name: str
) -> None:
    """Append one region row to a region table.

    Parameters
    ----------
    table : qtpy.QtWidgets.QTableWidget
        Table built by `_make_region_table`.
    region_id : int
        Structure id, stored both as text and as the row's user data.
    acronym : str
        Structure acronym.
    name : str
        Structure name.
    """
    row = table.rowCount()
    table.insertRow(row)
    id_item = QTableWidgetItem(str(region_id))
    id_item.setData(Qt.ItemDataRole.UserRole, int(region_id))
    table.setItem(row, 0, id_item)
    table.setItem(row, 1, QTableWidgetItem(acronym))
    table.setItem(row, 2, QTableWidgetItem(name))


def _table_rows(table: QTableWidget) -> list[tuple[int, str, str]]:
    """Read every row of a region table back as `(id, acronym, name)` triples.

    Parameters
    ----------
    table : qtpy.QtWidgets.QTableWidget
        Table built by `_make_region_table`.

    Returns
    -------
    list[tuple[int, str, str]]
        One triple per row, in table order.
    """
    rows = []
    for row in range(table.rowCount()):
        id_item = table.item(row, 0)
        acronym_item = table.item(row, 1)
        name_item = table.item(row, 2)
        if id_item is None or acronym_item is None or name_item is None:
            continue
        rows.append(
            (
                int(id_item.data(Qt.ItemDataRole.UserRole)),
                acronym_item.text(),
                name_item.text(),
            )
        )
    return rows


def _selected_row_indices(table: QTableWidget) -> list[int]:
    """Return the indices of the currently selected rows, ascending.

    Parameters
    ----------
    table : qtpy.QtWidgets.QTableWidget
        Table built by `_make_region_table`.

    Returns
    -------
    list[int]
        Selected row indices.
    """
    selection = table.selectionModel()
    if selection is None:
        return []
    return sorted({index.row() for index in selection.selectedRows()})


class AtlasPanel(QWidget):
    """Panel for loading BrainGlobe atlases, region masks and fUSI templates.

    Three groups:

    - **BrainGlobe atlas**: pick an atlas (downloaded ones listed first), load its
      `reference`, `annotation` and `hemispheres` as three layers, then open the
      structure tree in a right dock or reveal the region-mask group.
    - **Region masks**: search the loaded atlas's structures, build a list of regions,
      pick a hemisphere, and load one Labels layer per region mask.
    - **fUSI template**: pick one of the templates shipped in
      [`confusius.datasets`][confusius.datasets] and load it as an Image layer.

    Every download runs in a background thread behind the same thin indeterminate
    progress bar the Data I/O panel uses. The atlas list is fetched the first time the
    panel is shown, so opening the plugin never waits on the network.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._atlas: xr.Dataset | None = None
        self._atlas_name: str | None = None
        self._template_label: str | None = None
        self._tree: StructureTreeWidget | None = None
        self._tree_dock: QDockWidget | None = None
        self._atlases_listed = False
        self._setup_ui()
        install_no_scroll_wheel_filter(self)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._make_atlas_group())
        layout.addWidget(self._make_masks_group())
        layout.addWidget(self._make_template_group())
        layout.addStretch()

    def _make_atlas_group(self) -> QGroupBox:
        group = QGroupBox("BrainGlobe atlas")
        self._atlas_group = group
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)

        self._atlas_combo = QComboBox()
        self._atlas_combo.setMaxVisibleItems(20)
        self._atlas_combo.addItem("Fetching atlas list…")
        self._atlas_combo.setEnabled(False)
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Atlas"))
        combo_row.addWidget(self._atlas_combo, stretch=1)
        group_layout.addLayout(combo_row)

        self._atlas_status = QLabel()
        self._atlas_status.setObjectName("status_err")
        self._atlas_status.setWordWrap(True)
        self._atlas_status.hide()
        group_layout.addWidget(self._atlas_status)

        self._atlas_progress = _make_progress_bar()
        group_layout.addWidget(self._atlas_progress)

        self._load_atlas_btn = QPushButton("Load atlas")
        self._load_atlas_btn.setObjectName("primary_btn")
        self._load_atlas_btn.setEnabled(False)
        self._load_atlas_btn.clicked.connect(self._load_atlas)
        group_layout.addWidget(self._load_atlas_btn)

        self._tree_btn = QPushButton("Structure tree")
        self._tree_btn.setEnabled(False)
        self._tree_btn.clicked.connect(self._show_tree)
        self._mask_btn = QPushButton("Get mask")
        self._mask_btn.setCheckable(True)
        self._mask_btn.setEnabled(False)
        self._mask_btn.toggled.connect(self._on_mask_toggled)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._tree_btn)
        btn_row.addWidget(self._mask_btn)
        group_layout.addLayout(btn_row)
        return group

    def _make_masks_group(self) -> QGroupBox:
        group = QGroupBox("Region masks")
        self._masks_group = group
        group.hide()
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Acronym or name (regex)")
        self._search_edit.textChanged.connect(self._refresh_search)
        self._field_combo = QComboBox()
        self._field_combo.addItems(SEARCH_FIELDS)
        self._field_combo.currentTextChanged.connect(self._refresh_search)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search"))
        search_row.addWidget(self._search_edit, stretch=1)
        search_row.addWidget(QLabel("Field"))
        search_row.addWidget(self._field_combo)
        group_layout.addLayout(search_row)

        self._results_table = _make_region_table()
        self._results_table.doubleClicked.connect(self._add_regions)
        group_layout.addWidget(self._results_table)

        self._add_btn = QPushButton("▼ Add")
        self._add_btn.clicked.connect(self._add_regions)
        self._remove_btn = QPushButton("▲ Remove")
        self._remove_btn.clicked.connect(self._remove_regions)
        move_row = QHBoxLayout()
        move_row.addStretch()
        move_row.addWidget(self._add_btn)
        move_row.addWidget(self._remove_btn)
        move_row.addStretch()
        group_layout.addLayout(move_row)

        self._selected_table = _make_region_table()
        self._selected_table.doubleClicked.connect(self._remove_regions)
        group_layout.addWidget(self._selected_table)

        self._side_combo = QComboBox()
        self._side_combo.addItems(MASK_SIDES)
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Side"))
        side_row.addWidget(self._side_combo)
        side_row.addStretch()
        group_layout.addLayout(side_row)

        self._masks_progress = _make_progress_bar()
        group_layout.addWidget(self._masks_progress)

        self._load_masks_btn = QPushButton("Load masks")
        self._load_masks_btn.setObjectName("primary_btn")
        self._load_masks_btn.setEnabled(False)
        self._load_masks_btn.clicked.connect(self._load_masks)
        group_layout.addWidget(self._load_masks_btn)
        return group

    def _make_template_group(self) -> QGroupBox:
        group = QGroupBox("fUSI template")
        self._template_group = group
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)

        self._template_combo = QComboBox()
        self._template_combo.addItems(list(TEMPLATES))
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Template"))
        combo_row.addWidget(self._template_combo, stretch=1)
        group_layout.addLayout(combo_row)

        self._template_progress = _make_progress_bar()
        group_layout.addWidget(self._template_progress)

        self._load_template_btn = QPushButton("Load template")
        self._load_template_btn.setObjectName("primary_btn")
        self._load_template_btn.clicked.connect(self._load_template)
        group_layout.addWidget(self._load_template_btn)
        return group

    # ------------------------------------------------------------------
    # Atlas list
    # ------------------------------------------------------------------

    def showEvent(self, event: QShowEvent | None) -> None:  # ty: ignore[invalid-method-override]
        """Fetch the BrainGlobe atlas list the first time the panel becomes visible.

        Parameters
        ----------
        event : qtpy.QtGui.QShowEvent
            Qt show event.
        """
        super().showEvent(event)
        if not self._atlases_listed:
            self._atlases_listed = True
            worker = _list_atlases()
            worker.returned.connect(self._on_atlases_listed)
            worker.errored.connect(self._on_atlases_error)
            worker.start()

    def _on_atlases_listed(self, result: tuple[list[str], list[str] | None]) -> None:
        downloaded, available = result
        combo = self._atlas_combo
        combo.clear()
        for name in downloaded:
            combo.addItem(f"{name} (downloaded)", name)
        if available is None:
            self._set_atlas_status("Could not fetch the BrainGlobe atlas list.")
        else:
            self._atlas_status.hide()
            others = [name for name in available if name not in set(downloaded)]
            if downloaded and others:
                combo.insertSeparator(combo.count())
            for name in others:
                combo.addItem(name, name)
        has_atlases = combo.count() > 0
        combo.setEnabled(has_atlases)
        self._load_atlas_btn.setEnabled(has_atlases)

    def _on_atlases_error(self, exc: Exception) -> None:
        self._atlas_combo.clear()
        self._set_atlas_status(f"Could not list BrainGlobe atlases: {exc}")

    def _set_atlas_status(self, message: str) -> None:
        self._atlas_status.setText(message)
        self._atlas_status.show()

    # ------------------------------------------------------------------
    # Atlas loading
    # ------------------------------------------------------------------

    def _begin_atlas_work(self) -> None:
        self._atlas_combo.setEnabled(False)
        self._load_atlas_btn.setEnabled(False)
        self._load_atlas_btn.setText("Loading…")
        self._tree_btn.setEnabled(False)
        self._mask_btn.setEnabled(False)
        self._atlas_status.hide()
        self._atlas_progress.show()
        QApplication.processEvents()

    def _end_atlas_work(self) -> None:
        has_atlas = self._atlas is not None
        self._atlas_combo.setEnabled(self._atlas_combo.count() > 0)
        self._load_atlas_btn.setEnabled(self._atlas_combo.count() > 0)
        self._load_atlas_btn.setText("Load atlas")
        self._tree_btn.setEnabled(has_atlas)
        self._mask_btn.setEnabled(has_atlas)
        self._atlas_progress.hide()

    def _load_atlas(self) -> None:
        name = self._atlas_combo.currentData()
        if not name:
            show_error("Select an atlas.")
            return
        self._atlas_name = str(name)
        self._begin_atlas_work()
        worker = _fetch_atlas(self._atlas_name)
        worker.returned.connect(self._on_atlas_returned)
        worker.errored.connect(self._on_atlas_error)
        worker.start()

    def _on_atlas_returned(self, ds: xr.Dataset) -> None:
        name = self._atlas_name or str(ds.attrs.get("name", "atlas"))
        try:
            # Capture warnings from plot_napari (e.g. non-uniform spacing) and re-emit
            # them as napari notifications so they appear in the UI.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plot_napari(
                    ds["reference"], viewer=self.viewer, name=f"{name} reference"
                )
                plot_napari(
                    ds["annotation"],
                    viewer=self.viewer,
                    layer_type="labels",
                    name=f"{name} annotation",
                )
                plot_napari(
                    ds["hemispheres"],
                    viewer=self.viewer,
                    layer_type="labels",
                    name=f"{name} hemispheres",
                )
            for w in caught:
                if issubclass(w.category, UserWarning):
                    show_warning(str(w.message))
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
            self._end_atlas_work()
            return

        self._atlas = ds
        self._selected_table.setRowCount(0)
        self._load_masks_btn.setEnabled(False)
        self._refresh_search()
        if self._tree is not None:
            self._tree.set_atlas(ds)
        self._end_atlas_work()

    def _on_atlas_error(self, exc: Exception) -> None:
        self._end_atlas_work()
        show_error(str(exc))

    # ------------------------------------------------------------------
    # Structure tree
    # ------------------------------------------------------------------

    def _show_tree(self) -> None:
        """Open the structure tree in a right dock, creating it on first use.

        The widget outlives its dock: closing the dock orphans the tree (its parent
        becomes `None`), in which case it is docked again with its content intact.
        """
        if self._atlas is None:
            return
        if self._tree is not None:
            try:
                self._tree.isVisible()  # Raises RuntimeError if Qt deleted it.
            except RuntimeError:
                self._tree = None
                self._tree_dock = None
        if self._tree is None:
            self._tree = StructureTreeWidget()
            self._tree.set_atlas(self._atlas)
        if self._tree.parent() is None:
            self._tree_dock = self.viewer.window.add_dock_widget(
                self._tree, name="Atlas structures", area="right"
            )
        elif self._tree_dock is not None:
            self._tree_dock.show()
            self._tree_dock.raise_()

    # ------------------------------------------------------------------
    # Region masks
    # ------------------------------------------------------------------

    def _on_mask_toggled(self, checked: bool) -> None:
        self._masks_group.setVisible(checked)

    def _refresh_search(self) -> None:
        """Filter the results table through `AtlasAccessor.search`."""
        if self._atlas is None:
            self._results_table.setRowCount(0)
            return
        lookup = self._atlas.atlas.lookup
        pattern = self._search_edit.text().strip()
        if not pattern:
            df = lookup
        else:
            try:
                # Validate here: a half-typed regex such as "VIS(" otherwise surfaces
                # as a backend-specific error (`re.error` or pyarrow's) from pandas.
                re.compile(pattern)
            except re.error:
                df = lookup.iloc[0:0]
            else:
                df = self._atlas.atlas.search(
                    pattern,
                    field=self._field_combo.currentText(),  # type: ignore[arg-type]
                )
        _set_table_rows(
            self._results_table,
            zip((int(i) for i in df.index), df["acronym"], df["name"]),
        )

    def _add_regions(self) -> None:
        """Move the rows selected in the results table to the selected table."""
        existing = {region_id for region_id, _, _ in _table_rows(self._selected_table)}
        results = _table_rows(self._results_table)
        for row in _selected_row_indices(self._results_table):
            region_id, acronym, name = results[row]
            if region_id in existing:
                continue
            existing.add(region_id)
            _append_table_row(self._selected_table, region_id, acronym, name)
        self._load_masks_btn.setEnabled(self._selected_table.rowCount() > 0)

    def _remove_regions(self) -> None:
        """Drop the rows selected in the selected table."""
        for row in reversed(_selected_row_indices(self._selected_table)):
            self._selected_table.removeRow(row)
        self._load_masks_btn.setEnabled(self._selected_table.rowCount() > 0)

    def _selected_region_ids(self) -> list[int]:
        """Return the structure ids listed in the selected table, in order.

        Returns
        -------
        list[int]
            Structure ids to mask.
        """
        return [region_id for region_id, _, _ in _table_rows(self._selected_table)]

    def _begin_masks_work(self) -> None:
        self._load_masks_btn.setEnabled(False)
        self._load_masks_btn.setText("Loading…")
        self._masks_progress.show()
        QApplication.processEvents()

    def _end_masks_work(self) -> None:
        self._load_masks_btn.setEnabled(self._selected_table.rowCount() > 0)
        self._load_masks_btn.setText("Load masks")
        self._masks_progress.hide()

    def _load_masks(self) -> None:
        region_ids = self._selected_region_ids()
        if self._atlas is None or not region_ids:
            show_error("Add at least one region.")
            return
        self._begin_masks_work()
        worker = _compute_masks(self._atlas, region_ids, self._side_combo.currentText())
        worker.returned.connect(self._on_masks_returned)
        worker.errored.connect(self._on_masks_error)
        worker.start()

    def _on_masks_returned(self, masks: xr.DataArray) -> None:
        try:
            for mask_name in masks.coords["mask"].values:
                plot_napari(
                    masks.sel(mask=mask_name).drop_vars("mask"),
                    viewer=self.viewer,
                    layer_type="labels",
                    name=str(mask_name),
                )
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
        self._end_masks_work()

    def _on_masks_error(self, exc: Exception) -> None:
        self._end_masks_work()
        show_error(str(exc))

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def _begin_template_work(self) -> None:
        self._template_combo.setEnabled(False)
        self._load_template_btn.setEnabled(False)
        self._load_template_btn.setText("Loading…")
        self._template_progress.show()
        QApplication.processEvents()

    def _end_template_work(self) -> None:
        self._template_combo.setEnabled(True)
        self._load_template_btn.setEnabled(True)
        self._load_template_btn.setText("Load template")
        self._template_progress.hide()

    def _load_template(self) -> None:
        label = self._template_combo.currentText()
        self._template_label = label
        self._begin_template_work()
        worker = _fetch_template(TEMPLATES[label])
        worker.returned.connect(self._on_template_returned)
        worker.errored.connect(self._on_template_error)
        worker.start()

    def _on_template_returned(self, da: xr.DataArray) -> None:
        name = self._template_label or "template"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plot_napari(da, viewer=self.viewer, name=name)
            for w in caught:
                if issubclass(w.category, UserWarning):
                    show_warning(str(w.message))
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
        self._end_template_work()

    def _on_template_error(self, exc: Exception) -> None:
        self._end_template_work()
        show_error(str(exc))
