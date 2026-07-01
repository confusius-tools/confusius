"""Progress-layer helpers for the napari registration panel."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QDockWidget, QWidget

from confusius._dims import TIME_DIM
from confusius._napari._qt import find_main_window
from confusius._napari._registration._panel_utils import (
    _gamma_needs_reset,
    _get_image_display_kwargs_from_layer,
    _preserve_view,
)
from confusius._napari._registration._progress import (
    NapariProgressBridge,
    NapariRegistrationProgressReporter,
    NapariRegistrationProgressReporterBridge,
    make_napari_progress_factory,
)
from confusius.plotting.napari import plot_napari
from confusius.registration import resample_like

if TYPE_CHECKING:
    import numpy.typing as npt
    from napari.layers import Image

    from confusius._napari._registration._metric_plotter import (
        RegistrationMetricPlotter,
    )
    from confusius._napari._registration._panel import RegistrationPanel


def setup_volumewise_progress(
    panel: "RegistrationPanel",
    *,
    moving_layer: "Image",
    moving: xr.DataArray,
    layer_name: str,
    scale_mode: str,
) -> NapariRegistrationProgressReporter:
    """Create volumewise preview layers and a progress reporter.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose viewer and progress widgets are updated.
    moving_layer : napari.layers.Image
        Source moving layer shown in the viewer.
    moving : xarray.DataArray
        Moving data used to seed the preview layer.
    layer_name : str
        Name for the live output layer.
    scale_mode : str
        Registration scaling mode used to decide preview gamma handling.

    Returns
    -------
    NapariRegistrationProgressReporter
        Worker-side reporter that forwards completed-frame updates back to the
        panel.
    """
    teardown_volumewise_progress(panel, remove_layer=True)

    moving_display_kwargs = _get_image_display_kwargs_from_layer(moving_layer)
    moving_display_kwargs["colormap"] = "red"
    if _gamma_needs_reset(scale_mode):
        moving_display_kwargs["gamma"] = 1.0

    display_kwargs = dict(moving_display_kwargs)
    display_kwargs["colormap"] = "cyan"
    display_kwargs["blending"] = "additive"
    contrast_limits = tuple(calc_data_range(moving.data))
    preview_data = np.full(
        moving.shape,
        fill_value=float(np.min(moving.data)),
        dtype=np.asarray(moving.data).dtype,
    )
    preview = xr.DataArray(
        preview_data,
        dims=moving.dims,
        coords=moving.coords,
        attrs=moving.attrs.copy(),
    )

    with _preserve_view(panel.viewer):
        try:
            moving_preview_layer = panel._get_layer_by_name(
                panel._volumewise_moving_preview_layer_name()
            )
            if moving_preview_layer is None:
                _, moving_preview_layer = plot_napari(
                    moving,
                    viewer=panel.viewer,
                    name=panel._volumewise_moving_preview_layer_name(),
                    show_colorbar=False,
                    contrast_limits=contrast_limits,
                    **moving_display_kwargs,
                )
            else:
                moving_preview_layer = cast("Image", moving_preview_layer)
                panel._set_image_layer_data(
                    moving_preview_layer, np.asarray(moving.data)
                )
                moving_preview_layer.colormap = moving_display_kwargs["colormap"]
                moving_preview_layer.gamma = float(
                    moving_display_kwargs.get("gamma", 1.0)
                )
                moving_preview_layer.contrast_limits = contrast_limits

            fixed_preview_layer = panel._get_layer_by_name(
                panel._volume_fixed_preview_layer_name()
            )
            if fixed_preview_layer is not None:
                fixed_preview_layer.visible = False

            _, layer = plot_napari(
                preview,
                viewer=panel.viewer,
                name=layer_name,
                show_colorbar=False,
                contrast_limits=contrast_limits,
                **display_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            panel._set_error(f"Could not create progress layer: {exc}")
            raise
    bridge = NapariRegistrationProgressReporterBridge()
    bridge.frame_progress.connect(
        lambda completed_frames, total_frames: update_volumewise_progress_bar(
            panel, completed_frames, total_frames
        )
    )
    bridge.frame_completed.connect(
        lambda frame_index, frame_data: update_volumewise_progress_frame(
            panel, frame_index, frame_data
        )
    )

    panel._volumewise_progress_bridge = bridge
    panel._volumewise_progress_layer = cast("Image", layer)
    panel._volumewise_moving_preview_layer = cast("Image", moving_preview_layer)
    panel._volumewise_progress_time_axis = moving.dims.index(TIME_DIM)
    panel._volumewise_progress_total = moving.sizes[TIME_DIM]
    panel._progress.setRange(0, panel._volumewise_progress_total)
    panel._progress.setValue(0)
    return NapariRegistrationProgressReporter(
        bridge,
        n_frames=panel._volumewise_progress_total,
    )


def update_volumewise_progress_bar(
    panel: "RegistrationPanel", completed_frames: int, total_frames: int
) -> None:
    """Update the volumewise progress bar from completed-frame counts.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose progress bar is updated.
    completed_frames : int
        Number of frames completed so far.
    total_frames : int
        Total number of frames expected for the run.

    Returns
    -------
    None
        Updates the panel progress bar in place.
    """
    panel._volumewise_progress_total = total_frames
    panel._progress.setRange(0, total_frames)
    panel._progress.setValue(min(completed_frames, total_frames))


def update_volumewise_progress_frame(
    panel: "RegistrationPanel", frame_index: int, frame_data: object
) -> None:
    """Write a completed frame into the volumewise preview layer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose volumewise preview layer is updated.
    frame_index : int
        Time index of the completed frame.
    frame_data : object
        Registered frame data emitted by the worker.

    Returns
    -------
    None
        Writes the completed frame into the preview layer when valid.
    """
    layer = panel._volumewise_progress_layer
    time_axis = panel._volumewise_progress_time_axis
    if layer is None or time_axis is None:
        return
    if not isinstance(frame_data, np.ndarray):
        return
    if frame_index < 0 or frame_index >= layer.data.shape[time_axis]:
        return
    slicer: list[int | slice] = [slice(None) for _ in range(layer.data.ndim)]
    slicer[time_axis] = frame_index
    np.asarray(layer.data)[tuple(slicer)] = frame_data
    layer.refresh()


def teardown_volumewise_progress(
    panel: "RegistrationPanel", *, remove_layer: bool
) -> None:
    """Drop volumewise progress-layer references and optionally remove the layer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose volumewise progress state is cleared.
    remove_layer : bool
        Whether to also remove the live output layer from the viewer.

    Returns
    -------
    None
        Clears volumewise progress state stored on the panel.
    """
    if remove_layer and panel._volumewise_progress_layer is not None:
        try:
            panel.viewer.layers.remove(panel._volumewise_progress_layer)
        except (KeyError, ValueError):
            pass
    panel._volumewise_progress_bridge = None
    panel._volumewise_progress_layer = None
    panel._volumewise_progress_time_axis = None
    panel._volumewise_progress_total = None


def setup_volume_progress(
    panel: "RegistrationPanel",
    *,
    moving_layer: "Image",
    fixed_layer: "Image",
    moving: xr.DataArray,
    fixed: xr.DataArray,
    layer_name: str,
    initial_transform: "npt.NDArray[np.floating] | None" = None,
    scale_mode: str,
):
    """Create between-scan preview layers and a progress-plotter factory.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose viewer and progress widgets are updated.
    moving_layer : napari.layers.Image
        Source moving layer shown in the viewer.
    fixed_layer : napari.layers.Image
        Source fixed layer shown in the viewer.
    moving : xarray.DataArray
        Moving data used to seed the live preview.
    fixed : xarray.DataArray
        Fixed data defining the output grid.
    layer_name : str
        Name for the live registered preview layer.
    initial_transform : numpy.ndarray, optional
        Initial affine used to seed the preview before optimization starts.
    scale_mode : str
        Registration scaling mode used to decide preview gamma handling.

    Returns
    -------
    callable or None
        Progress-factory callback for `register_volume`, or `None` if the
        preview layer could not be created.
    """
    teardown_volume_progress(panel)

    fixed_display_kwargs = _get_image_display_kwargs_from_layer(fixed_layer)
    fixed_display_kwargs["colormap"] = "red"

    moving_display_kwargs = _get_image_display_kwargs_from_layer(moving_layer)
    moving_display_kwargs["colormap"] = "cyan"
    moving_display_kwargs["blending"] = "additive"
    if _gamma_needs_reset(scale_mode):
        fixed_display_kwargs["gamma"] = 1.0
        moving_display_kwargs["gamma"] = 1.0

    display_kwargs = dict(moving_display_kwargs)
    display_kwargs["colormap"] = "cyan"
    display_kwargs["blending"] = "additive"

    try:
        seed_transform = (
            np.asarray(initial_transform, dtype=float)
            if initial_transform is not None
            else np.eye(fixed.ndim + 1, dtype=float)
        )
        preview = resample_like(
            moving,
            fixed,
            seed_transform,
            interpolation="linear",
        )
        preview_contrast_limits = tuple(calc_data_range(preview.data))
    except Exception as exc:  # noqa: BLE001
        panel._set_error(f"Could not seed progress layer: {exc}")
        preview = xr.DataArray(
            np.zeros(fixed.shape, dtype=np.float32),
            coords=fixed.coords,
            dims=fixed.dims,
            attrs=fixed.attrs.copy(),
        )
        preview_contrast_limits = tuple(calc_data_range(preview.data))

    with _preserve_view(panel.viewer):
        try:
            fixed_preview_layer = panel._get_layer_by_name(
                panel._volume_fixed_preview_layer_name()
            )
            if fixed_preview_layer is None:
                _, fixed_preview_layer = plot_napari(
                    fixed,
                    viewer=panel.viewer,
                    name=panel._volume_fixed_preview_layer_name(),
                    show_colorbar=False,
                    **fixed_display_kwargs,
                )
            else:
                fixed_preview_layer = cast("Image", fixed_preview_layer)
                panel._set_image_layer_data(fixed_preview_layer, np.asarray(fixed.data))
                fixed_preview_layer.colormap = fixed_display_kwargs["colormap"]
                fixed_preview_layer.gamma = float(
                    fixed_display_kwargs.get("gamma", 1.0)
                )
                fixed_preview_layer.visible = True

            moving_preview_layer = panel._get_layer_by_name(
                panel._volume_moving_preview_layer_name()
            )
            if moving_preview_layer is None:
                _, moving_preview_layer = plot_napari(
                    preview,
                    viewer=panel.viewer,
                    name=panel._volume_moving_preview_layer_name(),
                    show_colorbar=False,
                    contrast_limits=preview_contrast_limits,
                    **moving_display_kwargs,
                )
            else:
                moving_preview_layer = cast("Image", moving_preview_layer)
                panel._set_image_layer_data(
                    moving_preview_layer, np.asarray(preview.data)
                )
                moving_preview_layer.colormap = moving_display_kwargs["colormap"]
                moving_preview_layer.blending = moving_display_kwargs["blending"]
                moving_preview_layer.gamma = float(
                    moving_display_kwargs.get("gamma", 1.0)
                )
                moving_preview_layer.contrast_limits = preview_contrast_limits
            moving_preview_layer.visible = False

            _, layer = plot_napari(
                preview,
                viewer=panel.viewer,
                name=layer_name,
                show_colorbar=False,
                contrast_limits=preview_contrast_limits,
                **display_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            panel._set_error(f"Could not create progress layer: {exc}")
            return None

    bridge = NapariProgressBridge()
    bridge.iterated.connect(lambda arr: update_progress_layer(panel, arr))
    panel._progress_bridge = bridge
    panel._progress_layer = cast("Image", layer)
    panel._progress_fixed_layer = cast("Image", fixed_preview_layer)
    panel._progress_moving_layer = cast("Image", moving_preview_layer)
    panel._progress_moving_layer.visible = False

    ensure_metric_plotter(panel)
    plotter = panel._metric_plotter
    if plotter is not None:
        plotter.reset()
        bridge.metric_updated.connect(plotter.add_metric)
    return make_napari_progress_factory(bridge)


def update_progress_layer(panel: "RegistrationPanel", arr: object) -> None:
    """Write an intermediate resampled array into the volume preview layer.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose between-scan preview layer is updated.
    arr : object
        Intermediate resampled array emitted by the worker.

    Returns
    -------
    None
        Replaces the preview-layer data when the emitted array is valid.
    """
    layer = panel._progress_layer
    if layer is None:
        return
    if not isinstance(arr, np.ndarray):
        return
    if arr.shape != layer.data.shape:
        return
    panel._set_image_layer_data(layer, arr)


def teardown_volume_progress(panel: "RegistrationPanel") -> None:
    """Remove the volume progress preview layer and bridge references.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose between-scan progress state is cleared.

    Returns
    -------
    None
        Clears between-scan progress state stored on the panel.
    """
    if panel._progress_layer is not None:
        try:
            panel.viewer.layers.remove(panel._progress_layer)
        except (KeyError, ValueError):
            pass
        panel._progress_layer = None
    panel._progress_bridge = None


def ensure_metric_plotter(
    panel: "RegistrationPanel",
) -> "RegistrationMetricPlotter | None":
    """Return the right-dock metric plotter, creating it on first use.

    Parameters
    ----------
    panel : RegistrationPanel
        Registration panel whose metric plotter should be available.

    Returns
    -------
    RegistrationMetricPlotter or None
        The docked metric plotter widget.
    """
    if panel._metric_plotter is None:
        from confusius._napari._registration._metric_plotter import (
            RegistrationMetricPlotter,
        )

        panel._metric_plotter = RegistrationMetricPlotter(panel.viewer)

    if panel._metric_dock is None or panel._metric_plotter.parent() is None:
        dock = panel.viewer.window.add_dock_widget(
            panel._metric_plotter,
            name="Registration Metric",
            area="right",
        )
        panel._metric_dock = dock

        def _settle_layout() -> None:
            main_win = find_main_window(dock)
            if main_win is None:
                return
            from qtpy.QtCore import QSize

            central = main_win.centralWidget()
            if central is None:
                return
            central.setMinimumSize(QSize(0, 0))
            for widget in central.findChildren(QWidget):
                widget.setMinimumSize(QSize(0, 0))
            for side_dock in main_win.findChildren(QDockWidget):
                if side_dock is dock:
                    continue
                side_dock.setMinimumHeight(0)
                widget = side_dock.widget()
                if widget is not None:
                    widget.setMinimumSize(QSize(0, 0))
            current = main_win.size()
            if current.height() < 800:
                main_win.resize(current.width(), 800)
            main_win.resizeDocks([dock], [220], Qt.Orientation.Vertical)

        QTimer.singleShot(200, _settle_layout)

    return panel._metric_plotter
