"""Data loading panel for the ConfUSIus napari plugin."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from confusius._napari._utils import show_napari_error

if TYPE_CHECKING:
    import napari


@thread_worker
def _load_file(path: Path, lazy: bool) -> xr.DataArray:
    """Load a fUSI data file in a background thread.

    Parameters
    ----------
    path : Path
        Path to a `.nii`, `.nii.gz`, `.scan`, or `.zarr` file/directory.
    lazy : bool
        Whether to return a lazy (Dask-backed) array without computing.
    """
    from confusius.io import load

    da = load(path)
    if not lazy:
        da = da.compute()
    return da


class _FileOrDirDialog(QFileDialog):
    """QFileDialog that accepts both files and directories.

    Qt's default `accept()` navigates into a directory when the user presses Open.
    Overriding it to call `done(Accepted)` directly when the selection is a directory
    lets users pick `.zarr` stores without the dialog diving inside them.
    """

    def accept(self) -> None:
        selected = self.selectedFiles()
        if selected and Path(selected[0]).is_dir():
            self.done(QDialog.DialogCode.Accepted)
        else:
            super().accept()


class DataPanel(QWidget):
    """Panel for loading fUSI data files into the napari viewer.

    Supported formats are whatever [`confusius.load`][confusius.load] accepts (`.nii`,
    `.nii.gz`, `.scan`, `.zarr`).

    Loading runs in a background thread so the UI stays responsive. A file is loaded
    automatically as soon as the user selects one via the Browse dialog, or when they
    press Enter in the path field.

    Parameters
    ----------
    viewer : napari.Viewer
        The active napari viewer instance.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # --- File group --------------------------------------------------
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(6)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to .nii / .nii.gz / .scan / .zarr …")
        # Pressing Enter in the path field triggers loading.
        self._path_edit.returnPressed.connect(self._load)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path_edit)
        path_row.addWidget(self._browse_btn)

        self._compute_check = QCheckBox("Load lazily")
        self._compute_check.setChecked(False)
        self._compute_check.setToolTip(
            "Return a lazy (Dask-backed) array without loading data into memory.\n"
            "Fast to open, but time scrubbing may be slow for large files.\n"
            "Uncheck to load the full array into memory first (recommended)."
        )

        file_layout.addLayout(path_row)
        file_layout.addWidget(self._compute_check)
        layout.addWidget(file_group)

        # --- Progress bar ------------------------------------------------
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumHeight(4)
        self._progress.hide()
        layout.addWidget(self._progress)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _begin_work(self) -> None:
        self._browse_btn.setEnabled(False)
        self._path_edit.setEnabled(False)
        self._progress.show()
        QApplication.processEvents()

    def _end_work(self) -> None:
        self._browse_btn.setEnabled(True)
        self._path_edit.setEnabled(True)
        self._progress.hide()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        start = self._path_edit.text().strip() or str(Path.home())

        dialog = _FileOrDirDialog(self, "Open fUSI Data", start)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter(
            "Supported files (*.nii *.nii.gz *.scan *.zarr);;"
            "NIfTI (*.nii *.nii.gz);;"
            "SCAN (*.scan);;"
            "Zarr (*.zarr);;"
            "All files (*)"
        )

        if dialog.exec() and dialog.selectedFiles():
            self._path_edit.setText(dialog.selectedFiles()[0])
            # Auto-load as soon as a file is selected.
            self._load()

    def _load(self) -> None:
        path_str = self._path_edit.text().strip()
        if not path_str:
            show_napari_error("Please select a file or directory.")
            return

        path = Path(path_str)
        if not path.exists():
            show_napari_error(f"Path not found: {path}")
            return

        self._begin_work()
        worker = _load_file(path, lazy=self._compute_check.isChecked())
        worker.returned.connect(self._on_load_returned)
        worker.errored.connect(self._on_load_error)
        worker.start()

    def _on_load_returned(self, da: xr.DataArray) -> None:
        try:
            from confusius.plotting.image import plot_napari

            _viewer, layer = plot_napari(da, viewer=self.viewer)
            layer.metadata["xarray"] = da
        except Exception as exc:  # noqa: BLE001
            show_napari_error(str(exc))
        self._end_work()

    def _on_load_error(self, exc: Exception) -> None:
        self._end_work()
        show_napari_error(str(exc))
