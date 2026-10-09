"""Atlas panel for the ConfUSIus napari plugin."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, TypedDict

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
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._atlas._tree import StructureTreeWidget
from confusius._napari._qt import (
    install_no_scroll_wheel_filter,
    make_segmented_buttons,
)
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


class TemplateSpec(TypedDict):
    """One entry of the fUSI template dropdown."""

    fetch: Callable[..., xr.DataArray]
    """`confusius.datasets.fetch_template_*` function returning the template."""

    reference_atlas: str
    """Human-readable name of the atlas space the template is aligned to."""


TEMPLATES: dict[str, TemplateSpec] = {
    "Huang 2025 mouse vascular": {
        "fetch": fetch_template_huang_2025,
        "reference_atlas": "Allen Mouse Brain Atlas (CCFv3)",
    },
    "Pepe-Mariani 2026 mouse vascular": {
        "fetch": fetch_template_pepe_mariani_2026,
        "reference_atlas": "Allen Mouse Brain Atlas (CCFv3)",
    },
}
"""fUSI templates offered in the template dropdown, keyed by display label."""

REGION_TABLE_ROW_HEIGHT_PX = 24
"""Fixed row height shared by the results and selected region tables."""

REGION_TABLE_COLUMN_WIDTHS_PX = (80, 90)
"""Fixed widths of the `id` and `acronym` columns; `name` takes the rest."""

MASK_SIDES = ("both", "left", "right")
"""Hemisphere choices offered for region masks, in dropdown order."""

RESULTS_TABLE_MIN_HEIGHT_PX = 140
"""Minimum height of the search-results table."""

SELECTED_TABLE_MIN_HEIGHT_PX = 93
"""Minimum height of the selected-regions table, two thirds of the results one."""


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
    # Keep the bar's slot in the layout while hidden: otherwise showing it grows
    # the panel by a few pixels and pops a sidebar scrollbar for the duration of
    # the work.
    policy = progress.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    progress.setSizePolicy(policy)
    progress.hide()
    return progress


def _make_region_table(min_height: int) -> QTableWidget:
    """Build a read-only `id` / `acronym` / `name` region table.

    Parameters
    ----------
    min_height : int
        Minimum table height in pixels.

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
    # Fixed row height and column widths so the results and selected tables line up
    # whatever their content.
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vertical_header.setDefaultSectionSize(REGION_TABLE_ROW_HEIGHT_PX)
    header = table.horizontalHeader()
    if header is not None:
        for column, width in enumerate(REGION_TABLE_COLUMN_WIDTHS_PX):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(column, width)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    table.setMinimumHeight(min_height)
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


class _LoadedAtlasRow(QWidget):
    """One row of the "Loaded atlases" list: atlas name plus four action buttons.

    Parameters
    ----------
    name : str
        BrainGlobe atlas name shown in the row (also its tooltip, since the label
        may be clipped in a narrow sidebar).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        outer.addLayout(layout)

        label = QLabel(name)
        label.setToolTip(name)
        # Ignore the text's width so the row never forces the sidebar wider; the
        # label clips instead (tooltip carries the full name).
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(label, stretch=1)

        self.volumes_btn = QPushButton("Volumes")
        self.volumes_btn.setToolTip(
            "Add the reference, annotation and hemispheres volumes as layers"
        )
        self.tree_btn = QPushButton("Tree")
        self.tree_btn.setToolTip("Show or hide the structure tree")
        self.masks_btn = QPushButton("Masks")
        self.masks_btn.setToolTip("Build region masks")
        self.remove_btn = QPushButton("✕")
        self.remove_btn.setToolTip("Remove atlas and free its memory")
        for btn in (self.volumes_btn, self.tree_btn, self.masks_btn, self.remove_btn):
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            layout.addWidget(btn)

        self.progress = _make_progress_bar()
        outer.addWidget(self.progress)


class AtlasPanel(QWidget):
    """Panel for loading fUSI templates, BrainGlobe atlases and region masks.

    Three groups:

    - **fUSI template**: pick one of the templates shipped in
      [`confusius.datasets`][confusius.datasets], see which atlas space it is aligned
      to, and load it as an Image layer.
    - **BrainGlobe atlas**: pick an atlas (downloaded ones listed first) and load it
      into memory. Each loaded atlas gets a row with buttons to add its `reference`,
      `annotation` and `hemispheres` layers, open its structure tree in a right dock,
      reveal the region-mask group for it, or remove it to free memory. An atlas can
      be loaded only once.
    - **Region masks**: search one loaded atlas's structures, build a list of
      regions, pick a hemisphere, and load one Labels layer per region mask.

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
        self._atlases: dict[str, xr.Dataset] = {}
        self._atlas_rows: dict[str, _LoadedAtlasRow] = {}
        self._trees: dict[str, StructureTreeWidget] = {}
        self._tree_docks: dict[str, QDockWidget] = {}
        self._atlas_name: str | None = None
        self._masks_atlas_name: str | None = None
        self._template_label: str | None = None
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
        page_row, (self._atlases_page_btn, self._templates_page_btn) = (
            make_segmented_buttons(["Atlases", "fUSI templates"], self)
        )
        layout.addLayout(page_row)

        self._atlases_page = QWidget()
        atlases_layout = QVBoxLayout(self._atlases_page)
        atlases_layout.setContentsMargins(0, 0, 0, 0)
        atlases_layout.setSpacing(8)
        atlases_layout.addWidget(self._make_atlas_group())
        atlases_layout.addWidget(self._make_masks_group())
        layout.addWidget(self._atlases_page)

        self._templates_page = QWidget()
        templates_layout = QVBoxLayout(self._templates_page)
        templates_layout.setContentsMargins(0, 0, 0, 0)
        templates_layout.addWidget(self._make_template_group())
        self._templates_page.hide()
        layout.addWidget(self._templates_page)

        self._atlases_page_btn.toggled.connect(self._atlases_page.setVisible)
        self._templates_page_btn.toggled.connect(self._templates_page.setVisible)
        layout.addStretch()

    def _make_template_group(self) -> QGroupBox:
        group = QGroupBox("fUSI template")
        self._template_group = group
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)

        self._template_combo = QComboBox()
        # Every combo in this panel keeps a content-independent minimum width:
        # a combo's default minimum hint spans its widest item text, and the atlas
        # combo holds ~200 BrainGlobe names, which would force the sidebar to
        # overflow horizontally (issue #183 pattern).
        self._template_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._template_combo.addItems(list(TEMPLATES))
        self._template_combo.currentTextChanged.connect(self._update_template_reference)
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Template"))
        combo_row.addWidget(self._template_combo, stretch=1)
        group_layout.addLayout(combo_row)

        self._template_reference = QLabel()
        self._template_reference.setObjectName("confusius_subtitle")
        self._template_reference.setWordWrap(True)
        group_layout.addWidget(self._template_reference)
        self._update_template_reference(self._template_combo.currentText())

        self._template_progress = _make_progress_bar()
        group_layout.addWidget(self._template_progress)

        self._load_template_btn = QPushButton("Load template")
        self._load_template_btn.setObjectName("primary_btn")
        self._load_template_btn.clicked.connect(self._load_template)
        group_layout.addWidget(self._load_template_btn)
        return group

    def _make_atlas_group(self) -> QGroupBox:
        group = QGroupBox("BrainGlobe atlas")
        self._atlas_group = group
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)

        self._atlas_combo = QComboBox()
        self._atlas_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._atlas_combo.setMaxVisibleItems(20)
        self._atlas_combo.addItem("Fetching atlas list…")
        self._atlas_combo.setEnabled(False)
        self._atlas_combo.currentIndexChanged.connect(self._update_load_atlas_button)
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

        self._loaded_label = QLabel("Loaded atlases")
        self._loaded_label.hide()
        group_layout.addWidget(self._loaded_label)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        group_layout.addLayout(self._rows_layout)
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
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search"))
        search_row.addWidget(self._search_edit, stretch=1)
        group_layout.addLayout(search_row)

        self._results_table = _make_region_table(RESULTS_TABLE_MIN_HEIGHT_PX)
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

        self._selected_table = _make_region_table(SELECTED_TABLE_MIN_HEIGHT_PX)
        self._selected_table.doubleClicked.connect(self._remove_regions)
        group_layout.addWidget(self._selected_table)

        self._side_combo = QComboBox()
        self._side_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        # The content-independent adjust policy alone leaves the combo too narrow
        # to show the selected side.
        self._side_combo.setMinimumContentsLength(len(max(MASK_SIDES, key=len)) + 2)
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
        combo.setEnabled(combo.count() > 0)
        self._update_load_atlas_button()

    def _on_atlases_error(self, exc: Exception) -> None:
        self._atlas_combo.clear()
        self._set_atlas_status(f"Could not list BrainGlobe atlases: {exc}")

    def _set_atlas_status(self, message: str) -> None:
        self._atlas_status.setText(message)
        self._atlas_status.show()

    def _update_load_atlas_button(self) -> None:
        """Enable Load atlas only for a selectable atlas that is not loaded yet."""
        name = self._atlas_combo.currentData()
        self._load_atlas_btn.setEnabled(bool(name) and name not in self._atlases)

    # ------------------------------------------------------------------
    # Atlas loading
    # ------------------------------------------------------------------

    def _begin_atlas_work(self) -> None:
        self._atlas_combo.setEnabled(False)
        self._load_atlas_btn.setEnabled(False)
        self._load_atlas_btn.setText("Loading…")
        self._atlas_status.hide()
        self._atlas_progress.show()
        QApplication.processEvents()

    def _end_atlas_work(self) -> None:
        self._atlas_combo.setEnabled(self._atlas_combo.count() > 0)
        self._load_atlas_btn.setText("Load atlas")
        self._update_load_atlas_button()
        self._atlas_progress.hide()

    def _load_atlas(self) -> None:
        name = self._atlas_combo.currentData()
        if not name:
            show_error("Select an atlas.")
            return
        name = str(name)
        if name in self._atlases:
            show_error(f"{name} is already loaded.")
            return
        self._atlas_name = name
        self._begin_atlas_work()
        worker = _fetch_atlas(name)
        worker.returned.connect(self._on_atlas_returned)
        worker.errored.connect(self._on_atlas_error)
        worker.start()

    def _on_atlas_returned(self, ds: xr.Dataset) -> None:
        """Register a loaded atlas under its row; layers are added on demand."""
        name = self._atlas_name or str(ds.attrs.get("name", "atlas"))
        self._atlases[name] = ds
        if name not in self._atlas_rows:
            self._add_atlas_row(name)
        if name in self._trees:
            self._trees[name].set_atlas(ds)
        if self._masks_atlas_name == name:
            self._selected_table.setRowCount(0)
            self._load_masks_btn.setEnabled(False)
            self._refresh_search()
        self._end_atlas_work()

    def _on_atlas_error(self, exc: Exception) -> None:
        self._end_atlas_work()
        show_error(str(exc))

    def _add_atlas_row(self, name: str) -> None:
        row = _LoadedAtlasRow(name)
        row.volumes_btn.clicked.connect(lambda: self._add_atlas_layers(name))
        row.tree_btn.clicked.connect(lambda: self._toggle_tree(name))
        row.masks_btn.clicked.connect(lambda: self._toggle_masks(name))
        row.remove_btn.clicked.connect(lambda: self._remove_atlas(name))
        self._atlas_rows[name] = row
        self._rows_layout.addWidget(row)
        # A widget added to a live layout is only shown on the next layout pass;
        # show it now so a caller (the guided tour) can target it immediately.
        row.show()
        self._loaded_label.show()

    def _add_atlas_layers(self, name: str) -> None:
        """Add the reference, annotation and hemispheres layers of a loaded atlas.

        Parameters
        ----------
        name : str
            Loaded atlas name.
        """
        ds = self._atlases[name]
        row = self._atlas_rows.get(name)
        if row is not None:
            # Layer creation must run on the GUI thread, so the bar cannot animate;
            # painting it before the work still tells the user something is happening.
            row.volumes_btn.setEnabled(False)
            row.progress.show()
            QApplication.processEvents()
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
        finally:
            if row is not None:
                row.progress.hide()
                row.volumes_btn.setEnabled(True)

    def _remove_atlas(self, name: str) -> None:
        """Forget a loaded atlas: drop its data, row, tree dock and mask group.

        Parameters
        ----------
        name : str
            Loaded atlas name.
        """
        self._atlases.pop(name, None)
        row = self._atlas_rows.pop(name, None)
        if row is not None:
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._loaded_label.setVisible(bool(self._atlas_rows))

        tree = self._trees.pop(name, None)
        dock = self._tree_docks.pop(name, None)
        if dock is not None and tree is not None and tree.parent() is not None:
            self.viewer.window.remove_dock_widget(dock)
        if tree is not None:
            tree.deleteLater()

        if self._masks_atlas_name == name:
            self._masks_atlas_name = None
            self._masks_group.hide()
        self._update_load_atlas_button()

    # ------------------------------------------------------------------
    # Structure tree
    # ------------------------------------------------------------------

    def _toggle_tree(self, name: str) -> None:
        """Open the structure tree of a loaded atlas in a right dock, or hide it.

        A second click while the dock is visible hides it, mirroring the Masks
        button. The widget outlives its dock: closing the dock orphans the tree (its
        parent becomes `None`), in which case it is docked again with its content
        intact.

        Parameters
        ----------
        name : str
            Loaded atlas name.
        """
        ds = self._atlases.get(name)
        if ds is None:
            return
        tree = self._trees.get(name)
        if tree is not None:
            try:
                tree.isVisible()  # Raises RuntimeError if Qt deleted it.
            except RuntimeError:
                tree = None
                self._tree_docks.pop(name, None)
        if tree is None:
            tree = StructureTreeWidget()
            tree.set_atlas(ds)
            self._trees[name] = tree
        if tree.parent() is None:
            self._tree_docks[name] = self.viewer.window.add_dock_widget(
                tree, name=f"Atlas structures: {name}", area="right"
            )
        elif (dock := self._tree_docks.get(name)) is not None:
            # `isHidden` reads the dock's own flag; `isVisible` is also false while
            # the main window is hidden (headless tests) and would never toggle.
            if dock.isHidden():
                dock.show()
                dock.raise_()
            else:
                dock.hide()

    # ------------------------------------------------------------------
    # Region masks
    # ------------------------------------------------------------------

    def _toggle_masks(self, name: str) -> None:
        """Show the region-mask group for `name`, or hide it if already showing it.

        Parameters
        ----------
        name : str
            Loaded atlas name.
        """
        if self._masks_atlas_name == name and self._masks_group.isVisibleTo(self):
            self._masks_group.hide()
            return
        if self._masks_atlas_name != name:
            self._masks_atlas_name = name
            self._masks_group.setTitle(f"Region masks: {name}")
            self._selected_table.setRowCount(0)
            self._load_masks_btn.setEnabled(False)
            self._refresh_search()
        self._masks_group.show()

    @property
    def _masks_atlas(self) -> xr.Dataset | None:
        """Atlas the region-mask group currently operates on, if any."""
        if self._masks_atlas_name is None:
            return None
        return self._atlases.get(self._masks_atlas_name)

    def _refresh_search(self) -> None:
        """Filter the results table through `AtlasAccessor.search`."""
        atlas = self._masks_atlas
        if atlas is None:
            self._results_table.setRowCount(0)
            return
        lookup = atlas.atlas.lookup
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
                df = atlas.atlas.search(pattern)
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
        atlas = self._masks_atlas
        region_ids = self._selected_region_ids()
        if atlas is None or not region_ids:
            show_error("Add at least one region.")
            return
        self._begin_masks_work()
        worker = _compute_masks(atlas, region_ids, self._side_combo.currentText())
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

    def _update_template_reference(self, label: str) -> None:
        spec = TEMPLATES.get(label)
        self._template_reference.setText(
            f"Reference atlas: {spec['reference_atlas']}" if spec else ""
        )

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
        worker = _fetch_template(TEMPLATES[label]["fetch"])
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
