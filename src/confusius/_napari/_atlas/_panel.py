"""Atlas panel for the ConfUSIus napari plugin."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_warning
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
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
from confusius._napari._qt import (
    install_no_scroll_wheel_filter,
    make_segmented_buttons,
)
from confusius._utils.templates import (
    FUSI_TEMPLATES,
    fetch_template,
    is_template_cached,
)
from confusius.datasets import fetch_brainglobe_atlas, is_brainglobe_atlas_cached
from confusius.plotting.napari import plot_napari

if TYPE_CHECKING:
    from collections.abc import Generator

    import napari
    from qtpy.QtGui import QShowEvent
    from qtpy.QtWidgets import QDockWidget


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

AtlasVolume = Literal["template", "annotation", "hemispheres"]
"""Data variables of an atlas Dataset the panel can add as layers."""

ATLAS_VOLUME_LAYER_TYPES: dict[AtlasVolume, Literal["image", "labels"]] = {
    "template": "image",
    "annotation": "labels",
    "hemispheres": "labels",
}
"""napari layer type used for each atlas volume, in button order."""


AtlasListing = tuple[dict[str, str | None], dict[str, str] | None]
"""Result of `_list_atlases`: local versions of the downloaded atlases, by name, and
the latest version of every atlas BrainGlobe knows about, or `None` when that list
could not be fetched."""


@thread_worker
def _list_atlases() -> AtlasListing:
    """List BrainGlobe atlases in a background thread.

    Returns
    -------
    downloaded : dict[str, str or None]
        Local version (e.g. `"1.2"`) of every atlas in the BrainGlobe cache, by name.
    available : dict[str, str] or None
        Latest version of every atlas BrainGlobe knows about, by name, or `None` when
        the list could not be fetched and BrainGlobe has no cached copy of it.
    """
    from brainglobe_atlasapi.list_atlases import (
        get_all_atlases_lastversions,
        get_downloaded_atlases,
        get_local_atlas_version,
    )

    downloaded = {
        name: get_local_atlas_version(name) for name in get_downloaded_atlases()
    }
    try:
        available = {
            name: str(version)
            for name, version in get_all_atlases_lastversions().items()
        }
    except Exception:  # noqa: BLE001
        # Offline with no cached `last_versions.conf`: BrainGlobe raises instead of
        # returning an empty mapping.
        available = None
    return downloaded, available


@thread_worker
def _fetch_atlas(atlas_name: str) -> Generator[str, None, xr.Dataset]:
    """Fetch a BrainGlobe atlas in a background thread, leaving it lazy.

    Parameters
    ----------
    atlas_name : str
        BrainGlobe atlas name.

    Yields
    ------
    str
        `atlas_name`, once, before the fetch when it has to download the atlas (or
        part of it) into the BrainGlobe cache.

    Returns
    -------
    xarray.Dataset
        Atlas Dataset whose volumes are still Dask arrays over the cached Zarr
        stores.
    """
    # BrainGlobe downloads behind a terminal progress bar the panel cannot show, so
    # tell the panel up front that a (possibly long) download is starting.
    if not is_brainglobe_atlas_cached(atlas_name):
        yield atlas_name
    # Reading the volumes here cost several seconds and gigabytes for fine atlases
    # (e.g. 2.4 GB for whs_sd_rat_39um). The layers take the Dask arrays as is, and
    # napari reads only the displayed slices; masks read what they need on first use.
    # `fetch_brainglobe_atlas` accepts any atlas name string despite its `AtlasName`
    # Literal annotation, so new BrainGlobe atlases work without a stub bump.
    return fetch_brainglobe_atlas(atlas_name)  # ty: ignore[invalid-argument-type]


@thread_worker
def _fetch_template(name: str, lazy: bool) -> Generator[str, None, xr.DataArray]:
    """Fetch a fUSI template in a background thread.

    Parameters
    ----------
    name : str
        A key of `FUSI_TEMPLATES`, e.g. `"brunner_grillet_2021"`.
    lazy : bool
        Whether to leave the voxels as a Dask array over the cached file. Otherwise
        they are read into memory here, once; the world coordinates stay lazy either
        way (see `_compute_masks` for why not `DataArray.compute()`).

    Yields
    ------
    str
        `name`, once, before the fetch when it has to download the template.

    Returns
    -------
    xarray.DataArray
        Template VoxelData array.
    """
    if not is_template_cached(name):
        yield name
    da = fetch_template(name, print_citation=False)
    if not lazy:
        da = da.copy(data=np.asarray(da.data))
    return da


@thread_worker
def _read_volume(volume: xr.DataArray) -> xr.DataArray:
    """Read one atlas volume into memory in a background thread.

    Parameters
    ----------
    volume : xarray.DataArray
        Lazy atlas volume (`template`, `annotation` or `hemispheres`).

    Returns
    -------
    xarray.DataArray
        The same volume with its voxels in memory; the world coordinates stay
        index-derived (see `_compute_masks` for why not `DataArray.compute()`).
    """
    return volume.copy(data=np.asarray(volume.data))


@thread_worker
def _compute_masks(
    atlas: xr.Dataset, region_ids: list[int], side: str
) -> tuple[xr.DataArray, xr.Dataset]:
    """Compute region masks in a background thread.

    Parameters
    ----------
    atlas : xarray.Dataset
        Loaded atlas Dataset, possibly still lazy.
    region_ids : list[int]
        Structure ids to mask.
    side : {"both", "left", "right"}
        Hemisphere applied to every region.

    Returns
    -------
    masks : xarray.DataArray
        Masks stacked along a `mask` dimension, as returned by
        [`get_masks`][confusius.atlas.AtlasAccessor.get_masks].
    atlas : xarray.Dataset
        `atlas` with `annotation` and `hemispheres` in memory, for the panel to keep
        so later mask requests skip the read.
    """
    # `get_masks` needs only annotation and hemispheres (the latter to restrict a mask
    # to one side); read them once here, off the GUI thread. Something like
    # `DataArray.compute()` would also materialize the lazy z/y/x world coordinates,
    # dense float64 grids that can be quite memory heavy for some atlases.
    # `np.asarray` reads a Dask array and passes an already-loaded NumPy array through
    # without copying.
    atlas = atlas.assign(
        annotation=atlas["annotation"].copy(data=np.asarray(atlas["annotation"].data)),
        hemispheres=atlas["hemispheres"].copy(
            data=np.asarray(atlas["hemispheres"].data)
        ),
    )
    masks = atlas.atlas.get_masks(region_ids, sides=side)  # type: ignore[arg-type]
    return masks, atlas


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


class AtlasPanel(QWidget):
    """Panel for loading fUSI templates, one BrainGlobe atlas and its region masks.

    Two pages:

    - **Atlases**: pick a BrainGlobe atlas (downloaded ones listed first). A
      downloaded atlas loads as soon as it is selected, its volumes staying on disk
      until needed; any other greys the rest of the page out behind a **Download
      atlas** button. The panel holds the selected atlas only: picking another one
      drops it. While an atlas is loaded, **Template**, **Annotation** and
      **Hemispheres** each add that volume as a layer, **Structure tree** opens the
      region hierarchy in a right dock, and the always-present **Region masks**
      group lets you search its structures, build a list of regions, pick a
      hemisphere, and load one Labels layer per region mask.
    - **fUSI templates**: pick one of the templates shipped in
      [`confusius.datasets`][confusius.datasets], see which atlas space it is aligned
      to, and load it as an Image layer.

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
        self._tree: StructureTreeWidget | None = None
        self._tree_dock: QDockWidget | None = None
        self._atlas_loading = False
        # Whether the in-flight load is downloading the atlas into the BrainGlobe
        # cache, which can take minutes for fine atlases and deserves a notice.
        self._atlas_downloading = False
        # In-flight template loads, so several can load at once (e.g. from the CLI)
        # and the busy state only clears when the last one ends; those still
        # downloading are named in the notice under the dropdown.
        self._template_jobs = 0
        self._templates_downloading: set[str] = set()
        # Whether to add the three volume layers as soon as the atlas is loaded.
        self._pending_layers = False
        # Volume reads in flight (Make layers lazily unchecked); the atlas progress
        # bar stays up until the last one ends.
        self._volume_reads = 0
        self._atlases_listed = False
        # Last `_list_atlases` result, kept to rebuild the dropdown after a download.
        self._atlas_list: AtlasListing | None = None
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
        self._update_atlas_controls()

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
        self._template_combo.addItems(list(FUSI_TEMPLATES))
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

        self._template_lazy_check = QCheckBox("Load lazily")
        self._template_lazy_check.setChecked(False)
        self._template_lazy_check.setToolTip(
            "Keep the template as a lazy Dask array that napari reads slice by "
            "slice.\nTemplates are small gzip files, so reading them into memory "
            "once (unchecked) browses faster."
        )
        group_layout.addWidget(self._template_lazy_check)

        self._template_notice = QLabel()
        self._template_notice.setObjectName("confusius_subtitle")
        self._template_notice.setWordWrap(True)
        self._template_notice.hide()
        group_layout.addWidget(self._template_notice)

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
        self._atlas_combo.setPlaceholderText("Select an atlas")
        self._atlas_combo.addItem("Fetching atlas list…")
        self._atlas_combo.currentIndexChanged.connect(self._on_atlas_selected)
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Atlas"))
        combo_row.addWidget(self._atlas_combo, stretch=1)
        group_layout.addLayout(combo_row)

        self._atlas_status = QLabel()
        self._atlas_status.setObjectName("status_err")
        self._atlas_status.setWordWrap(True)
        self._atlas_status.hide()
        group_layout.addWidget(self._atlas_status)

        self._atlas_notice = QLabel()
        self._atlas_notice.setObjectName("confusius_subtitle")
        self._atlas_notice.setWordWrap(True)
        self._atlas_notice.hide()
        group_layout.addWidget(self._atlas_notice)

        # Only shown while the selected atlas is not loaded: a downloaded atlas loads
        # on selection, so this is the download step (or a retry after an error).
        self._download_atlas_btn = QPushButton("Download atlas")
        self._download_atlas_btn.setObjectName("primary_btn")
        self._download_atlas_btn.clicked.connect(self._download_atlas)
        self._download_atlas_btn.hide()
        group_layout.addWidget(self._download_atlas_btn)

        # One bar for the atlas fetch above it and the volume reads below it.
        self._atlas_progress = _make_progress_bar()
        group_layout.addWidget(self._atlas_progress)

        self._lazy_layers_check = QCheckBox("Make layers lazily")
        self._lazy_layers_check.setChecked(True)
        self._lazy_layers_check.setToolTip(
            "Checked, napari reads the displayed slices from disk.\n"
            "Unchecked, napari first reads that volume into memory."
        )
        group_layout.addWidget(self._lazy_layers_check)

        self._volume_btns: dict[AtlasVolume, QPushButton] = {}
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(4)
        for volume in ATLAS_VOLUME_LAYER_TYPES:
            btn = QPushButton(volume.capitalize())
            btn.setToolTip(f"Add the {volume} volume as a layer")
            btn.clicked.connect(lambda _=False, v=volume: self._add_layer(v))
            self._volume_btns[volume] = btn
            buttons_row.addWidget(btn)
        group_layout.addLayout(buttons_row)

        self._tree_btn = QPushButton("Structure tree")
        self._tree_btn.setToolTip("Show or hide the structure tree")
        self._tree_btn.clicked.connect(self._toggle_tree)
        group_layout.addWidget(self._tree_btn)
        return group

    def _make_masks_group(self) -> QGroupBox:
        group = QGroupBox("Region masks")
        self._masks_group = group
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Acronym or name")
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

        self._masks_progress = _make_progress_bar()
        group_layout.addWidget(self._masks_progress)

        self._load_masks_btn = QPushButton("Load masks")
        self._load_masks_btn.setObjectName("primary_btn")
        self._load_masks_btn.setEnabled(False)
        self._load_masks_btn.clicked.connect(self._load_masks)
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Side"))
        side_row.addWidget(self._side_combo)
        side_row.addWidget(self._load_masks_btn, stretch=1)
        group_layout.addLayout(side_row)
        return group

    def _update_atlas_controls(self) -> None:
        """Enable the atlas widgets for the current state: none, loading or loaded.

        The dropdown locks while an atlas loads. Everything that needs an atlas (its
        volume buttons, Structure tree and the Region masks group) is enabled only
        while one is loaded. Download atlas appears only while the selected atlas is
        not the loaded one: before a download, during it (disabled, as a status) or
        after a failed load.
        """
        loaded = self._atlas is not None
        selected = self._atlas_combo.currentData()
        if self._atlas_loading:
            show_download = self._atlas_downloading
            text = "Downloading…"
        elif selected and selected != self._atlas_name:
            show_download = True
            text = "Load atlas" if self._is_downloaded(selected) else "Download atlas"
        else:
            show_download = False
            text = "Download atlas"
        self._download_atlas_btn.setText(text)
        self._download_atlas_btn.setEnabled(not self._atlas_loading)
        self._download_atlas_btn.setVisible(show_download)
        self._atlas_combo.setEnabled(
            not self._atlas_loading and self._atlas_combo.count() > 0
        )
        for btn in self._volume_btns.values():
            btn.setEnabled(loaded)
        self._tree_btn.setEnabled(loaded)
        self._masks_group.setEnabled(loaded)

    def _is_downloaded(self, name: str) -> bool:
        """Whether the last listing found an atlas in the BrainGlobe cache.

        Parameters
        ----------
        name : str
            BrainGlobe atlas name.

        Returns
        -------
        bool
            Whether the atlas is in the listing's downloaded group. False before the
            listing has arrived.
        """
        return self._atlas_list is not None and name in self._atlas_list[0]

    # ------------------------------------------------------------------
    # Atlas list
    # ------------------------------------------------------------------

    def showEvent(self, a0: QShowEvent | None) -> None:
        """Fetch the BrainGlobe atlas list the first time the panel becomes visible.

        Parameters
        ----------
        a0 : qtpy.QtGui.QShowEvent
            Qt show event.
        """
        super().showEvent(a0)
        if not self._atlases_listed:
            self._atlases_listed = True
            worker = _list_atlases()
            worker.returned.connect(self._on_atlases_listed)
            worker.errored.connect(self._on_atlases_error)
            worker.start()

    def _on_atlases_listed(self, result: AtlasListing) -> None:
        self._atlas_list = result
        if result[1] is None:
            self._set_atlas_status("Could not fetch the BrainGlobe atlas list.")
        else:
            self._atlas_status.hide()
        self._populate_atlas_combo()

    def _populate_atlas_combo(self) -> None:
        """Rebuild the dropdown from the last listing, keeping the selection.

        Downloaded atlases come first, flagged when BrainGlobe has published a newer
        version than the cached one, then a separator and every other atlas, all
        sorted by name.
        """
        if self._atlas_list is None:
            return
        downloaded, available = self._atlas_list
        latest = available or {}
        combo = self._atlas_combo
        selected = self._atlas_name or combo.currentData()
        # Rebuilding the list is not a user selection: keep `_on_atlas_selected` out
        # of it, so the panel's atlas is untouched and nothing loads by itself.
        with QSignalBlocker(combo):
            combo.clear()
            for name in sorted(downloaded):
                local = downloaded[name]
                outdated = local is not None and latest.get(name, local) != local
                suffix = (
                    " (downloaded, update available)" if outdated else " (downloaded)"
                )
                combo.addItem(name + suffix, name)
            others = [name for name in sorted(latest) if name not in downloaded]
            if downloaded and others:
                combo.insertSeparator(combo.count())
            for name in others:
                combo.addItem(name, name)
            # Nothing selected shows the placeholder text instead of the first atlas.
            combo.setCurrentIndex(combo.findData(selected) if selected else -1)
        self._update_atlas_controls()

    def _mark_downloaded(self, name: str) -> None:
        """Move a freshly fetched atlas into the dropdown's downloaded group.

        A successful fetch leaves the atlas in the BrainGlobe cache, at the latest
        version when it was just downloaded. An atlas that was already cached keeps
        its recorded local version, and so its update flag. Before the listing has
        arrived there is nothing to do: the listing will find the atlas on disk.

        Parameters
        ----------
        name : str
            BrainGlobe atlas name that was just fetched.
        """
        if self._atlas_list is None:
            return
        downloaded, available = self._atlas_list
        if name not in downloaded:
            downloaded[name] = (available or {}).get(name)
            self._populate_atlas_combo()

    def _on_atlases_error(self, exc: Exception) -> None:
        self._atlas_combo.clear()
        self._set_atlas_status(f"Could not list BrainGlobe atlases: {exc}")
        self._update_atlas_controls()

    def _set_atlas_status(self, message: str) -> None:
        self._atlas_status.setText(message)
        self._atlas_status.show()

    def _select_atlas_in_combo(self, name: str) -> None:
        """Point the dropdown at the loaded atlas, when the list has it.

        Parameters
        ----------
        name : str
            Loaded atlas name.
        """
        index = self._atlas_combo.findData(name)
        if index >= 0:
            with QSignalBlocker(self._atlas_combo):
                self._atlas_combo.setCurrentIndex(index)
            self._update_atlas_controls()

    # ------------------------------------------------------------------
    # Atlas loading
    # ------------------------------------------------------------------

    def _update_atlas_progress(self) -> None:
        """Show the shared bar while the atlas fetch or any volume read is running."""
        self._atlas_progress.setVisible(self._atlas_loading or self._volume_reads > 0)

    def _begin_atlas_work(self) -> None:
        self._atlas_loading = True
        self._atlas_status.hide()
        self._update_atlas_progress()
        self._update_atlas_controls()
        QApplication.processEvents()

    def _end_atlas_work(self) -> None:
        self._atlas_loading = False
        self._atlas_downloading = False
        self._atlas_notice.hide()
        self._update_atlas_progress()
        self._update_atlas_controls()

    def _on_atlas_downloading(self, name: str) -> None:
        """Tell the user the in-flight load is downloading `name` first.

        Parameters
        ----------
        name : str
            BrainGlobe atlas name being downloaded.
        """
        self._atlas_downloading = True
        self._atlas_notice.setText(
            f"Downloading {name}… This can take a while for large atlases."
        )
        self._atlas_notice.show()
        self._update_atlas_controls()

    def _on_atlas_selected(self) -> None:
        """Make the selected atlas the panel's atlas: load it if downloaded, else
        drop the current one and offer Download atlas."""
        name = self._atlas_combo.currentData()
        if not name or name == self._atlas_name:
            return
        if self._atlas is not None:
            self._remove_atlas()
        if self._is_downloaded(str(name)):
            self.load_atlas(str(name))
        else:
            self._update_atlas_controls()

    def _download_atlas(self) -> None:
        name = self._atlas_combo.currentData()
        if name:
            self.load_atlas(str(name))

    def load_atlas(self, name: str, *, add_layers: bool = False) -> None:
        """Load a BrainGlobe atlas in the background as the panel's atlas.

        This is what selecting a downloaded atlas, or clicking Download atlas, does;
        the CLI's `--atlas` option calls it directly. The panel holds one atlas: an
        atlas already loaded under another name is removed first (its layers stay in
        the viewer). The call is ignored while this atlas is already loaded or
        another load is in progress.

        Parameters
        ----------
        name : str
            BrainGlobe atlas name, e.g. `"allen_mouse_25um"`. Downloaded into the
            BrainGlobe cache on first use.
        add_layers : bool, default: False
            Whether to add the atlas's template, annotation and hemispheres layers as
            soon as it is loaded, as the three volume buttons would.
        """
        if name == self._atlas_name or self._atlas_loading:
            return
        if self._atlas is not None:
            self._remove_atlas()
        self._pending_layers = add_layers
        self._begin_atlas_work()
        worker = _fetch_atlas(name)
        worker.yielded.connect(self._on_atlas_downloading)
        worker.returned.connect(lambda ds: self._on_atlas_returned(ds, name))
        worker.errored.connect(self._on_atlas_error)
        worker.start()

    def _on_atlas_returned(self, ds: xr.Dataset, name: str) -> None:
        """Install a loaded atlas as the panel's atlas; layers are added on demand.

        Parameters
        ----------
        ds : xarray.Dataset
            Loaded atlas Dataset.
        name : str
            BrainGlobe atlas name it was requested under.
        """
        if self._atlas is not None and name != self._atlas_name:
            self._remove_atlas()
        self._atlas = ds
        self._atlas_name = name
        if self._tree is not None:
            self._tree.set_atlas(ds)
        self._selected_table.setRowCount(0)
        self._load_masks_btn.setEnabled(False)
        self._refresh_search()
        self._mark_downloaded(name)
        self._select_atlas_in_combo(name)
        self._end_atlas_work()
        if self._pending_layers:
            self._pending_layers = False
            for volume in ATLAS_VOLUME_LAYER_TYPES:
                self._add_layer(volume)

    def _on_atlas_error(self, exc: Exception) -> None:
        self._pending_layers = False
        self._end_atlas_work()
        show_error(str(exc))

    def _add_layer(self, volume: AtlasVolume) -> None:
        """Add one volume of the loaded atlas as a layer named `<atlas> <volume>`.

        With Make layers lazily unchecked, a volume still on disk is first read in a
        background thread and kept on the panel's atlas, so the masks worker and
        later clicks reuse it.

        Parameters
        ----------
        volume : {"template", "annotation", "hemispheres"}
            Atlas data variable to add.
        """
        if self._atlas is None or self._atlas_name is None:
            return
        da = self._atlas[volume]
        name = self._atlas_name
        if not self._lazy_layers_check.isChecked() and not isinstance(
            da.data, np.ndarray
        ):
            self._volume_reads += 1
            self._update_atlas_progress()
            worker = _read_volume(da)
            worker.returned.connect(
                lambda loaded: self._on_volume_read(loaded, volume, name)
            )
            worker.errored.connect(self._on_volume_read_error)
            worker.start()
            return
        self._show_volume(da, volume, name)

    def _on_volume_read(
        self, loaded: xr.DataArray, volume: AtlasVolume, name: str
    ) -> None:
        """Keep a volume read into memory on the atlas, then add its layer.

        Parameters
        ----------
        loaded : xarray.DataArray
            The volume with its voxels in memory.
        volume : {"template", "annotation", "hemispheres"}
            Atlas data variable it is.
        name : str
            Atlas the read was requested for; the volume is only kept if that atlas
            is still the loaded one.
        """
        if self._atlas is not None and self._atlas_name == name:
            self._atlas = self._atlas.assign({volume: loaded})
        self._end_volume_read()
        self._show_volume(loaded, volume, name)

    def _on_volume_read_error(self, exc: Exception) -> None:
        self._end_volume_read()
        show_error(str(exc))

    def _end_volume_read(self) -> None:
        self._volume_reads = max(self._volume_reads - 1, 0)
        self._update_atlas_progress()

    def _show_volume(self, da: xr.DataArray, volume: AtlasVolume, name: str) -> None:
        """Add an atlas volume as the `<name> <volume>` layer, on the GUI thread.

        Parameters
        ----------
        da : xarray.DataArray
            Volume to display, lazy or in memory.
        volume : {"template", "annotation", "hemispheres"}
            Atlas data variable it is, which picks the layer type.
        name : str
            Atlas name the layer is named after.
        """
        # Layer creation must run on the GUI thread, so the bar cannot animate;
        # painting it before the work still tells the user something is happening.
        self._atlas_progress.show()
        QApplication.processEvents()
        try:
            # Capture warnings from plot_napari (e.g. non-uniform spacing) and re-emit
            # them as napari notifications so they appear in the UI.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plot_napari(
                    da,
                    viewer=self.viewer,
                    layer_type=ATLAS_VOLUME_LAYER_TYPES[volume],
                    name=f"{name} {volume}",
                )
            for w in caught:
                if issubclass(w.category, UserWarning):
                    show_warning(str(w.message))
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
        finally:
            self._update_atlas_progress()

    def _remove_atlas(self) -> None:
        """Forget the loaded atlas: drop its data, tree dock and region tables."""
        self._atlas = None
        self._atlas_name = None

        tree, dock = self._tree, self._tree_dock
        self._tree = None
        self._tree_dock = None
        if dock is not None and tree is not None and tree.parent() is not None:
            self.viewer.window.remove_dock_widget(dock)
        if tree is not None:
            tree.deleteLater()

        self._selected_table.setRowCount(0)
        self._load_masks_btn.setEnabled(False)
        self._refresh_search()
        self._update_atlas_controls()

    # ------------------------------------------------------------------
    # Structure tree
    # ------------------------------------------------------------------

    def _toggle_tree(self) -> None:
        """Open the loaded atlas's structure tree in a right dock, or hide it.

        A second click while the dock is visible hides it. The widget outlives its
        dock: closing the dock orphans the tree (its parent becomes `None`), in which
        case it is docked again with its content intact.
        """
        if self._atlas is None:
            return
        tree = self._tree
        if tree is not None:
            try:
                tree.isVisible()  # Raises RuntimeError if Qt deleted it.
            except RuntimeError:
                tree = None
                self._tree_dock = None
        if tree is None:
            tree = StructureTreeWidget()
            tree.set_atlas(self._atlas)
            self._tree = tree
        if tree.parent() is None:
            self._tree_dock = self.viewer.window.add_dock_widget(
                tree, name=f"Atlas structures: {self._atlas_name}", area="right"
            )
        elif self._tree_dock is not None:
            # `isHidden` reads the dock's own flag; `isVisible` is also false while
            # the main window is hidden (headless tests) and would never toggle.
            if self._tree_dock.isHidden():
                self._tree_dock.show()
                self._tree_dock.raise_()
            else:
                self._tree_dock.hide()

    # ------------------------------------------------------------------
    # Region masks
    # ------------------------------------------------------------------

    def _refresh_search(self) -> None:
        """Filter the results table through `AtlasAccessor.search`."""
        atlas = self._atlas
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
        atlas = self._atlas
        region_ids = self._selected_region_ids()
        if atlas is None or not region_ids:
            show_error("Add at least one region.")
            return
        self._begin_masks_work()
        worker = _compute_masks(atlas, region_ids, self._side_combo.currentText())
        worker.returned.connect(
            lambda result: self._on_masks_worker_returned(atlas, result)
        )
        worker.errored.connect(self._on_masks_error)
        worker.start()

    def _on_masks_worker_returned(
        self, requested: xr.Dataset, result: tuple[xr.DataArray, xr.Dataset]
    ) -> None:
        """Keep the in-memory atlas volumes, then add the mask layers.

        Parameters
        ----------
        requested : xarray.Dataset
            Atlas Dataset the worker was given.
        result : tuple of (xarray.DataArray, xarray.Dataset)
            Masks and the atlas with `annotation` and `hemispheres` in memory.
        """
        masks, in_memory = result
        # Only replace the stored atlas if it was not removed or replaced meanwhile.
        if self._atlas is requested:
            self._atlas = in_memory
        self._on_masks_returned(masks)

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
        spec = FUSI_TEMPLATES.get(label)
        self._template_reference.setText(
            f"Reference atlas: {spec['reference_atlas']}" if spec else ""
        )

    def _update_template_controls(self) -> None:
        """Enable the template widgets for the current state: idle, loading or
        downloading, naming the templates still downloading in the notice."""
        busy = self._template_jobs > 0
        downloading = sorted(self._templates_downloading)
        if downloading:
            text = "Downloading…"
        elif busy:
            text = "Loading…"
        else:
            text = "Load template"
        self._load_template_btn.setText(text)
        self._template_combo.setEnabled(not busy)
        self._load_template_btn.setEnabled(not busy)
        self._template_notice.setText(f"Downloading {', '.join(downloading)}…")
        self._template_notice.setVisible(bool(downloading))

    def _begin_template_work(self) -> None:
        self._template_jobs += 1
        self._template_progress.show()
        self._update_template_controls()
        QApplication.processEvents()

    def _end_template_work(self, name: str) -> None:
        """Clear one template load; the busy state ends with the last one.

        Parameters
        ----------
        name : str
            Template whose load finished or failed.
        """
        self._template_jobs = max(self._template_jobs - 1, 0)
        self._templates_downloading.discard(name)
        if not self._template_jobs:
            self._template_progress.hide()
        self._update_template_controls()

    def _on_template_downloading(self, name: str) -> None:
        """Tell the user an in-flight template load is downloading `name` first.

        Parameters
        ----------
        name : str
            Template being downloaded.
        """
        self._templates_downloading.add(name)
        self._update_template_controls()

    def _load_template(self) -> None:
        self.load_template(
            self._template_combo.currentText(),
            lazy=self._template_lazy_check.isChecked(),
        )

    def load_template(self, name: str, *, lazy: bool = False) -> None:
        """Fetch a fUSI template in the background and add it as an Image layer.

        This is what the Load template button does; the CLI's `--template` option
        calls it directly.

        Parameters
        ----------
        name : str
            A key of `FUSI_TEMPLATES`, e.g. `"brunner_grillet_2021"`. The layer takes this
            name.
        lazy : bool, default: False
            Whether to leave the voxels on disk as a Dask array, as the Load lazily
            box does, instead of reading them into memory once.

        Raises
        ------
        KeyError
            If `name` is not a known template.
        """
        if name not in FUSI_TEMPLATES:
            raise KeyError(
                f"Unknown fUSI template {name!r}; expected one of "
                f"{sorted(FUSI_TEMPLATES)}."
            )
        self._begin_template_work()
        worker = _fetch_template(name, lazy)
        worker.yielded.connect(self._on_template_downloading)
        worker.returned.connect(lambda da: self._on_template_returned(da, name))
        worker.errored.connect(lambda exc: self._on_template_error(exc, name))
        worker.start()

    def _on_template_returned(self, da: xr.DataArray, name: str) -> None:
        """Add a fetched template as an Image layer.

        Parameters
        ----------
        da : xarray.DataArray
            Template VoxelData array.
        name : str
            Template name, used as the layer name.
        """
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plot_napari(da, viewer=self.viewer, name=name)
            for w in caught:
                if issubclass(w.category, UserWarning):
                    show_warning(str(w.message))
        except Exception as exc:  # noqa: BLE001
            show_error(str(exc))
        self._end_template_work(name)

    def _on_template_error(self, exc: Exception, name: str) -> None:
        self._end_template_work(name)
        show_error(str(exc))
