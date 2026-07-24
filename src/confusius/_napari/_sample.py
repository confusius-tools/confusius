"""Sample dataset commands for the ConfUSIus napari plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import napari
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QProgressDialog

from confusius.datasets import (
    fetch_cybis_pereira_2026,
    fetch_nunez_elizalde_2022,
)
from confusius.io import load

if TYPE_CHECKING:
    import xarray as xr
    from napari.types import FullLayerData

from collections.abc import Callable, Sequence

ProgressCallback = Callable[[int, int, str], None]

_AWAKE_MOUSE_SUBJECT = "CR022"
"""Subject id for the awake mouse sample."""

_AWAKE_MOUSE_SESSION = "20201011"
"""Session id for the awake mouse sample."""

_AWAKE_MOUSE_TASK = "spontaneous"
"""Task label for the awake mouse sample."""

_AWAKE_MOUSE_ACQ = "slice04"
"""Acquisition label for the awake mouse sample."""

_RAT_REGISTRATION_SUBJECT = "rat75"
"""Subject id for the rat registration sample pair."""

_RAT_REGISTRATION_SESSIONS = ("20220523", "20220524")
"""Session ids for the rat registration sample pair."""

_RAT_REGISTRATION_ACQ = "slice32"
"""Acquisition label for the rat registration sample pair."""

_PROGRESS_SCALE = 1000
"""Integer range used by the Qt progress dialog."""


@dataclass(frozen=True)
class SampleFileSpec:
    """One file to load as part of a sample."""

    path: Path
    name: str
    layer_kwargs: dict[str, object] | None = None


@dataclass(frozen=True)
class SampleSpec:
    """Definition of one napari sample entry."""

    title: str
    initial_status: str
    files_resolver: Callable[[ProgressCallback | None], Sequence[SampleFileSpec]]
    gamma: float | None = None
    affine_key: str | None = None


class SampleDownloadCancelledError(RuntimeError):
    """Raised when the user aborts the sample download."""


def _strip_known_suffixes(path: Path) -> str:
    """Return a display name derived from a sample file path.

    Parameters
    ----------
    path : pathlib.Path
        Sample file path.

    Returns
    -------
    str
        File name with `.nii` or `.nii.gz` removed when present.
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if path.suffix:
        return path.stem
    return name


def _resolve_awake_mouse_recording(
    progress_callback: ProgressCallback | None = None,
) -> list[SampleFileSpec]:
    """Return the awake mouse sample file to load.

    Parameters
    ----------
    progress_callback : collections.abc.Callable[[int, int, str], None], optional
        Callback receiving cumulative downloaded bytes, total bytes to download,
        and a user-facing description. If not provided, download progress is not
        reported.

    Returns
    -------
    list[confusius._napari._sample.SampleFileSpec]
        Single-file sample definition for the awake 2D+t mouse recording.
    """
    bids_root = fetch_nunez_elizalde_2022(
        datasets="rawdata",
        subjects=_AWAKE_MOUSE_SUBJECT,
        sessions=_AWAKE_MOUSE_SESSION,
        tasks=_AWAKE_MOUSE_TASK,
        acqs=_AWAKE_MOUSE_ACQ,
        datatypes="fusi",
        progress_callback=progress_callback,
    )
    path = (
        bids_root
        / f"sub-{_AWAKE_MOUSE_SUBJECT}"
        / f"ses-{_AWAKE_MOUSE_SESSION}"
        / "fusi"
        / (
            f"sub-{_AWAKE_MOUSE_SUBJECT}_ses-{_AWAKE_MOUSE_SESSION}"
            f"_task-{_AWAKE_MOUSE_TASK}_acq-{_AWAKE_MOUSE_ACQ}_pwd.nii.gz"
        )
    )
    return [SampleFileSpec(path=path, name=_strip_known_suffixes(path))]


def _resolve_rat_registration_pair(
    progress_callback: ProgressCallback | None = None,
) -> list[SampleFileSpec]:
    """Return the rat registration sample files to load.

    Parameters
    ----------
    progress_callback : collections.abc.Callable[[int, int, str], None], optional
        Callback receiving cumulative downloaded bytes, total bytes to download,
        and a user-facing description. This sample uses a fetcher without progress
        callbacks, so the argument is accepted for API consistency and otherwise
        ignored.

    Returns
    -------
    list[confusius._napari._sample.SampleFileSpec]
        Two-file sample definition for the rat angiography registration pair.
    """
    del progress_callback
    bids_root = fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects=_RAT_REGISTRATION_SUBJECT,
        sessions=list(_RAT_REGISTRATION_SESSIONS),
        datatypes="angio",
        acqs=_RAT_REGISTRATION_ACQ,
    )
    specs: list[SampleFileSpec] = []
    for session in _RAT_REGISTRATION_SESSIONS:
        path = (
            bids_root
            / f"sub-{_RAT_REGISTRATION_SUBJECT}"
            / f"ses-{session}"
            / "angio"
            / (
                f"sub-{_RAT_REGISTRATION_SUBJECT}_ses-{session}"
                f"_acq-{_RAT_REGISTRATION_ACQ}_rec-minframe2d_pwd.nii.gz"
            )
        )
        specs.append(
            SampleFileSpec(
                path=path,
                name=_strip_known_suffixes(path),
                layer_kwargs={
                    "colormap": "red"
                    if session == _RAT_REGISTRATION_SESSIONS[0]
                    else "cyan",
                    "blending": "additive",
                },
            )
        )
    return specs


_SAMPLE_SPECS = {
    "awake-mouse-recording": SampleSpec(
        title="ConfUSIus sample",
        initial_status="Checking sample cache...",
        files_resolver=_resolve_awake_mouse_recording,
        gamma=0.4,
    ),
    "rat-registration-pair": SampleSpec(
        title="ConfUSIus sample",
        initial_status="Checking sample cache...",
        files_resolver=_resolve_rat_registration_pair,
        gamma=0.4,
        affine_key="physical_to_qform",
    ),
}
"""Registered napari sample definitions."""


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

    Raises
    ------
    SampleDownloadCancelledError
        If the user cancels the progress dialog.
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


def _load_sample_dataarray(path: Path, affine_key: str | None) -> xr.DataArray:
    """Load one sample file and optionally switch affine spaces.

    Parameters
    ----------
    path : pathlib.Path
        Path to the sample file.
    affine_key : str, optional
        Key in `da.attrs["affines"]` to apply after loading. If not provided,
        the DataArray is returned in its default loaded coordinate frame.

    Returns
    -------
    xarray.DataArray
        Loaded sample data, optionally re-expressed in the requested affine space.
    """
    da = load(path).compute()
    if affine_key is not None:
        da, _ = da.fusi.affine.apply(affine_key)
    return da


def _dataarray_to_layer_data(
    da: xr.DataArray,
    name: str,
    gamma: float | None,
    layer_kwargs: dict[str, object] | None = None,
) -> FullLayerData:
    """Convert one loaded sample DataArray to napari layer data.

    Parameters
    ----------
    da : xarray.DataArray
        Loaded sample data.
    name : str
        Layer name to expose in napari.
    gamma : float, optional
        Default gamma to apply to image layers. If not provided, the converted
        layer data is left unchanged.
    layer_kwargs : dict[str, object], optional
        Extra napari layer keyword arguments to merge into the converted layer data.

    Returns
    -------
    napari.types.FullLayerData
        Napari layer-data tuple ready to be returned by a sample command.
    """
    from confusius._utils.napari import convert_dataarray_to_layer_data

    layer_data = convert_dataarray_to_layer_data(da, name=name)
    _, kwargs, layer_type = layer_data
    if gamma is not None and layer_type == "image":
        kwargs.setdefault("gamma", gamma)
    if layer_kwargs is not None:
        kwargs.update(layer_kwargs)
    return layer_data


def _apply_sample_axis_labels(
    viewer: napari.Viewer, axis_labels: tuple[str, ...]
) -> None:
    """Right-align sample axis labels onto the viewer dimension sliders.

    napari does not copy a layer's `axis_labels` onto `viewer.dims` for the
    sample/reader path, so the sliders would otherwise keep the default
    `-N ... -1` labels. Called after napari has added the sample layers, so
    `viewer.dims.ndim` already accounts for them; the labels are applied to the
    trailing axes to preserve any leading labels from higher-dimensional layers.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer whose dimension labels should be updated.
    axis_labels : tuple[str, ...]
        Axis labels of the highest-dimensional sample layer.
    """
    current = list(viewer.dims.axis_labels)
    count = len(axis_labels)
    if count <= len(current):
        current[-count:] = axis_labels
        viewer.dims.axis_labels = tuple(current)


def _open_sample(sample_key: str) -> list[FullLayerData]:
    """Open one registered napari sample.

    Parameters
    ----------
    sample_key : str
        Key into [`_SAMPLE_SPECS`][confusius._napari._sample._SAMPLE_SPECS].

    Returns
    -------
    list[napari.types.FullLayerData]
        One or more napari layer-data tuples for the requested sample.

    Raises
    ------
    KeyError
        If `sample_key` is not registered.
    SampleDownloadCancelledError
        If the user cancels the download dialog.
    """
    spec = _SAMPLE_SPECS[sample_key]
    viewer = napari.current_viewer()
    if viewer is not None:
        viewer.scale_bar.visible = True
    dialog = QProgressDialog(viewer.window._qt_window if viewer is not None else None)
    dialog.setWindowTitle(spec.title)
    dialog.setLabelText(spec.initial_status)
    dialog.setCancelButtonText("Abort")
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.show()

    try:
        sample_files = spec.files_resolver(
            lambda current, total, description: _update_progress_dialog(
                dialog, current, total, description
            )
        )
        layers: list[FullLayerData] = []
        for index, sample_file in enumerate(sample_files, start=1):
            description = (
                f"Loading sample {index}/{len(sample_files)}..."
                if len(sample_files) > 1
                else "Loading sample..."
            )
            _update_progress_dialog(dialog, 0, 0, description)
            da = _load_sample_dataarray(sample_file.path, affine_key=spec.affine_key)
            layers.append(
                _dataarray_to_layer_data(
                    da,
                    sample_file.name,
                    spec.gamma,
                    layer_kwargs=sample_file.layer_kwargs,
                )
            )
        # napari adds the returned layers after this command returns, so defer the
        # dims-label update to the next event-loop iteration when ndim is settled.
        # Viewer ndim matches the highest-dimensional layer; reuse its axis labels.
        if viewer is not None and layers:
            target = viewer
            axis_labels = tuple(
                max((kwargs["axis_labels"] for _, kwargs, _ in layers), key=len)
            )
            QTimer.singleShot(0, lambda: _apply_sample_axis_labels(target, axis_labels))
        return layers
    except SampleDownloadCancelledError:
        show_info("Sample download cancelled.")
        raise
    finally:
        dialog.close()


def open_awake_mouse_recording_sample() -> list[FullLayerData]:
    """Return an awake 2D+t mouse recording sample.

    Returns
    -------
    list[napari.types.FullLayerData]
        Single-layer napari sample data for the awake mouse recording.
    """
    return _open_sample("awake-mouse-recording")


def open_rat_registration_pair_sample() -> list[FullLayerData]:
    """Return a two-layer rat angiography sample for registration demos.

    Returns
    -------
    list[napari.types.FullLayerData]
        Two napari layers for the rat registration pair.
    """
    return _open_sample("rat-registration-pair")
