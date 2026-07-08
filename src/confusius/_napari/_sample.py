"""Sample dataset commands for the ConfUSIus napari plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QApplication, QProgressDialog

from confusius._utils.napari import convert_dataarray_to_layer_data
from confusius.datasets import fetch_nunez_elizalde_2022
from confusius.io import load

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from napari.types import FullLayerData

_SAMPLE_SUBJECT = "CR022"
"""Subject id for the napari sample recording."""

_SAMPLE_SESSION = "20201011"
"""Session id for the napari sample recording."""

_SAMPLE_TASK = "spontaneous"
"""Task label for the napari sample recording."""

_SAMPLE_ACQ = "slice03"
"""Acquisition label for the napari sample recording."""

_SAMPLE_RELATIVE_PATH = (
    f"sub-{_SAMPLE_SUBJECT}/ses-{_SAMPLE_SESSION}/fusi/"
    f"sub-{_SAMPLE_SUBJECT}_ses-{_SAMPLE_SESSION}"
    f"_task-{_SAMPLE_TASK}_acq-{_SAMPLE_ACQ}_pwd.nii.gz"
)
"""Path to the sample recording relative to the fetched BIDS root."""

_PROGRESS_SCALE = 1000
"""Integer range used by the Qt progress dialog."""


class SampleDownloadCancelledError(RuntimeError):
    """Raised when the user aborts the sample download."""


def _resolve_nunez_elizalde_2022_sample_path(
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Return the cached path of the bundled napari sample recording.

    Parameters
    ----------
    progress_callback : Callable[[int, int, str], None], optional
        Callback receiving cumulative downloaded bytes, total bytes to download,
        and a user-facing description.

    Returns
    -------
    pathlib.Path
        Path to the cached power Doppler recording.
    """
    bids_root = fetch_nunez_elizalde_2022(
        datasets="rawdata",
        subjects=_SAMPLE_SUBJECT,
        sessions=_SAMPLE_SESSION,
        tasks=_SAMPLE_TASK,
        acqs=_SAMPLE_ACQ,
        datatypes="fusi",
        progress_callback=progress_callback,
    )
    return bids_root / _SAMPLE_RELATIVE_PATH


def _update_progress_dialog(
    dialog: QProgressDialog, current: int, total: int, description: str
) -> None:
    """Refresh the sample progress dialog.

    Parameters
    ----------
    dialog : qtpy.QtWidgets.QProgressDialog
        Progress dialog to update.
    current : int
        Bytes downloaded so far.
    total : int
        Total bytes to download.
    description : str
        User-facing status message.
    """
    if total <= 0:
        dialog.setRange(0, 0)
        dialog.setLabelText(description)
    else:
        dialog.setRange(0, _PROGRESS_SCALE)
        dialog.setValue(min(_PROGRESS_SCALE, int(_PROGRESS_SCALE * current / total)))
        dialog.setLabelText(
            f"{description}\n{current / 1_048_576:.1f} / {total / 1_048_576:.1f} MiB"
        )
    QApplication.processEvents()
    if dialog.wasCanceled():
        raise SampleDownloadCancelledError("Sample download cancelled.")


def open_nunez_elizalde_2022_sample() -> list[FullLayerData]:
    """Return a sample recording from Nunez-Elizalde et al. (2022)."""
    viewer = napari.current_viewer()
    dialog = QProgressDialog(viewer.window._qt_window if viewer is not None else None)
    dialog.setWindowTitle("ConfUSIus sample")
    dialog.setLabelText("Checking sample cache...")
    dialog.setCancelButtonText("Abort")
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.show()

    try:
        sample_path = _resolve_nunez_elizalde_2022_sample_path(
            progress_callback=lambda current, total, description: (
                _update_progress_dialog(dialog, current, total, description)
            )
        )
        _update_progress_dialog(dialog, 0, 0, "Loading sample...")
        da = load(sample_path).compute()
        return [convert_dataarray_to_layer_data(da, name=sample_path.name)]
    except SampleDownloadCancelledError:
        show_info("Sample download cancelled.")
        raise
    finally:
        dialog.close()
