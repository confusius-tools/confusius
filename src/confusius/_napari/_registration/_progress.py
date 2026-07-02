"""Napari-layer-backed progress reporting for `register_volume`.

This module provides a progress reporter that mirrors the matplotlib-based
[`MatplotlibRegistrationProgressPlotter`][confusius.registration.MatplotlibRegistrationProgressPlotter] but
streams the intermediate resampled volume into a napari Image layer instead of a
matplotlib figure.

The reporter is intentionally split into two pieces so that napari layers can be
constructed and signal slots connected on the GUI thread before the registration worker
thread starts:

- [`NapariProgressBridge`][confusius._napari._registration._progress.NapariProgressBridge]
  is a lightweight `QObject` that lives on the GUI thread and exposes Qt signals. The
  worker thread calls `emit` on it; Qt marshals the slot invocations back to the GUI
  thread via an automatically-detected queued connection.
- [`NapariRegistrationProgressPlotter`][confusius._napari._registration._progress.NapariRegistrationProgressPlotter]
  implements the [`RegistrationProgress`][confusius.registration.RegistrationProgress]
  protocol. It is constructed inside `register_volume` (i.e. on the worker thread) and
  resamples the moving image at every iteration using the current tentative transform,
  forwarding the resulting array to the bridge.

Connection lifecycle:

1. The panel constructs a `NapariProgressBridge` on the GUI thread and connects its
  `iterated` signal to a slot that writes the array into the resampled napari layer.
2. The panel builds a factory (via
  [`make_napari_progress_factory`][confusius._napari._registration._progress.make_napari_progress_factory])
  that closes over the bridge and returns a `NapariRegistrationProgressPlotter` instance when called
  by `register_volume`.
3. `register_volume` instantiates the progress inside the worker thread and wires it to
  SimpleITK's iteration and end events as usual.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np
from qtpy.QtCore import QObject, Signal

from confusius.registration.progress import _resample_intermediate

if TYPE_CHECKING:
    import SimpleITK as sitk
    import xarray as xr

    from confusius.registration import RegistrationDiagnostics, RegistrationProgress


class NapariProgressBridge(QObject):
    """Thread-boundary signal bridge for napari registration progress.

    Construct this on the GUI thread before starting the registration worker. Connect
    `iterated` to a slot that mutates a napari layer (e.g. writes `layer.data = arr`);
    the slot will be invoked on the GUI thread thanks to Qt's automatic cross-thread
    connection. The bridge itself never touches the napari layer, keeping a clean
    separation between the worker's data path and the GUI update path.

    See Also
    --------
    NapariRegistrationProgressPlotter : Worker-side reporter that emits via this bridge.
    """

    iterated = Signal(object)
    """Emitted at every optimizer iteration with the resampled moving image as a numpy
    array in numpy axis order (matching `fixed`)."""

    metric_updated = Signal(float)
    """Emitted at every optimizer iteration with the current optimizer metric value (a
    float)."""

    finished = Signal()
    """Emitted once when the registration end event fires."""


class NapariRegistrationProgressPlotter:
    """Napari-layer progress reporter for `register_volume`.

    Implements the [`RegistrationProgress`][confusius.registration.RegistrationProgress]
    protocol. Stores the registration method and SimpleITK images it needs to resample
    the moving image at each iteration. The resampled array is forwarded to the bridge
    via a Qt signal, so this object is safe to call from the SimpleITK command callback
    running on the worker thread.

    Parameters
    ----------
    bridge : NapariProgressBridge
        GUI-thread signal bridge. Stored by reference; never accessed for GUI APIs from
        this object.
    registration_method : SimpleITK.ImageRegistrationMethod
        Active registration method whose `GetInitialTransform` is used to resample the
        moving image at each iteration.
    fixed_img : SimpleITK.Image
        Fixed image defining the resample grid.
    moving_img : SimpleITK.Image
        Moving image to resample.
    plot_metric : bool, default: True
        Whether to emit `metric_updated` on each iteration. Kept aligned with the
        matplotlib plotter factory signature.
    plot_composite : bool, default: True
        Kept for signature compatibility with the matplotlib plotter factory. The
        napari preview always shows the resampled moving image directly.
    resample_kwargs : dict, optional
        Extra keyword arguments for the intermediate resample. Supported keys are
        `interpolation`, `fill_value`, and `sitk_threads`.
    """

    def __init__(
        self,
        bridge: NapariProgressBridge,
        registration_method: "sitk.ImageRegistrationMethod",
        fixed_img: "sitk.Image",
        moving_img: "sitk.Image",
        *,
        plot_metric: bool = True,
        plot_composite: bool = True,
        resample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._bridge = bridge
        self._method = registration_method
        self._fixed_img = fixed_img
        self._moving_img = moving_img
        _kw = dict(resample_kwargs or {})
        self._interpolation = cast(
            'Literal["linear", "nearest", "bspline"]',
            _kw.get("interpolation", "linear"),
        )
        self._fill_value = float(_kw.get("fill_value", 0.0))
        self._sitk_threads = int(_kw.get("sitk_threads", -1))
        self._plot_metric = plot_metric
        del plot_composite

    def update(self) -> None:
        """Resample the moving image with the current transform and emit it.

        Called at every SimpleITK iteration event from the worker thread. The
        resampled array is sent to the GUI thread via `bridge.iterated`; the
        emit is thread-safe and does not require this object to live on the
        GUI thread. The current optimizer metric value is also forwarded via
        `bridge.metric_updated` so a metric-curve plotter can track
        convergence.
        """
        import SimpleITK as sitk

        if self._plot_metric:
            self._bridge.metric_updated.emit(float(self._method.GetMetricValue()))

        resampled = _resample_intermediate(
            self._method,
            self._moving_img,
            self._fixed_img,
            interpolation=self._interpolation,
            fill_value=self._fill_value,
            sitk_threads=self._sitk_threads,
        )
        # .T restores numpy axis order (inverse of the .T used when building
        # the SITK image), matching what `register_volume` produces.
        arr = np.asarray(sitk.GetArrayFromImage(resampled).T)
        self._bridge.iterated.emit(arr)

    def close(self) -> None:
        """Signal that the registration run has ended.

        Called at the SimpleITK end event. The final resampled state is
        available on the bridge via the last `iterated` payload; the panel is
        responsible for retrieving/refreshing the layer from `register_volume`'s
        returned DataArray, so this signal is informational (e.g. to stop a
        spinner or mark the layer as finalised).
        """
        self._bridge.finished.emit()


class NapariRegistrationProgressReporterBridge(QObject):
    """Thread-boundary signal bridge for volumewise registration progress."""

    frame_progress = Signal(int, int)
    """Emitted with `(completed_frames, total_frames)`."""

    frame_completed = Signal(int, object)
    """Emitted with `(frame_index, registered_frame_array)` when one frame
    finishes."""

    finished = Signal()
    """Emitted once when the volumewise run ends."""


class NapariRegistrationProgressReporter:
    """Aggregate per-frame progress for `register_volumewise` on the GUI thread.

    Parameters
    ----------
    bridge : NapariRegistrationProgressReporterBridge
        GUI-thread signal bridge used to forward progress updates.
    n_frames : int
        Number of frames that will be registered.
    """

    def __init__(
        self,
        bridge: NapariRegistrationProgressReporterBridge,
        *,
        n_frames: int,
    ) -> None:
        self._bridge = bridge
        self._n_frames = n_frames
        self._completed_frames: set[int] = set()
        self._lock = Lock()

    def frame_completed(
        self,
        frame_index: int,
        registered_frame: "xr.DataArray",
        diagnostics: "RegistrationDiagnostics",
    ) -> None:
        """Emit one completed frame for GUI-side layer updates.

        Parameters
        ----------
        frame_index : int
            Index of the completed frame.
        registered_frame : xarray.DataArray
            Registered frame data to write into the napari output layer.
        diagnostics : confusius.registration.RegistrationDiagnostics
            Diagnostics collected for the completed frame.
        """
        with self._lock:
            del diagnostics
            self._completed_frames.add(frame_index)
            completed = len(self._completed_frames)
            total = self._n_frames
        self._bridge.frame_progress.emit(completed, total)
        self._bridge.frame_completed.emit(
            frame_index, np.asarray(registered_frame.values)
        )

    def close(self) -> None:
        """Signal the end of the volumewise run."""
        self._bridge.finished.emit()


def make_napari_progress_factory(
    bridge: NapariProgressBridge,
) -> "Callable[..., RegistrationProgress]":
    """Return a progress-plotter factory bound to a bridge.

    The returned callable has the signature expected by `register_volume`'s
    `progress_plotter` argument—it accepts `(registration_method, fixed_img, moving_img,
    *, plot_metric, plot_composite, resample_kwargs)` and returns a
    [`NapariRegistrationProgressPlotter`][confusius._napari._registration._progress.NapariRegistrationProgressPlotter]
    instance wrapping `bridge`.

    Parameters
    ----------
    bridge : NapariProgressBridge
        GUI-thread bridge the constructed reporter will emit through.

    Returns
    -------
    callable
        Factory suitable as the `progress_plotter` argument of
        [`register_volume`][confusius.registration.register_volume].
    """

    def factory(
        registration_method: "sitk.ImageRegistrationMethod",
        fixed_img: "sitk.Image",
        moving_img: "sitk.Image",
        *,
        plot_metric: bool = True,
        plot_composite: bool = True,
        resample_kwargs: dict[str, Any] | None = None,
    ) -> "RegistrationProgress":
        """Build a NapariRegistrationProgressPlotter wrapping the captured bridge.

        Parameters
        ----------
        registration_method : SimpleITK.ImageRegistrationMethod
            Active registration method whose transform is sampled at every iteration.
        fixed_img : SimpleITK.Image
            Fixed reference image defining the resample grid.
        moving_img : SimpleITK.Image
            Moving image to resample.
        plot_metric : bool, default: True
            Whether to emit `metric_updated` on each iteration.
        plot_composite : bool, default: True
            Kept for signature compatibility with the matplotlib plotter factory.
        resample_kwargs : dict, optional
            Extra keyword arguments for the intermediate resample. Supported keys are
            `interpolation`, `fill_value`, and `sitk_threads`.

        Returns
        -------
        RegistrationProgress
            Progress reporter ready to be wired to SimpleITK's iteration and
            end events by `register_volume`.
        """
        return NapariRegistrationProgressPlotter(
            bridge,
            registration_method,
            fixed_img,
            moving_img,
            plot_metric=plot_metric,
            plot_composite=plot_composite,
            resample_kwargs=resample_kwargs,
        )

    return factory
