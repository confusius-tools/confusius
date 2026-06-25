"""Napari-layer-backed progress reporting for ``register_volume``.

This module provides a progress reporter that mirrors the matplotlib-based
[`RegistrationProgressPlotter`][confusius.registration.RegistrationProgressPlotter]
but streams the intermediate resampled volume into a napari Image layer instead
of a matplotlib figure.

The reporter is intentionally split into two pieces so that napari layers can be
constructed and signal slots connected on the GUI thread before the
registration worker thread starts:

- [`NapariProgressBridge`][confusius._napari._registration._progress.NapariProgressBridge]
  is a lightweight `QObject` that lives on the GUI thread and exposes Qt
  signals. The worker thread calls `emit` on it; Qt marshals the slot
  invocations back to the GUI thread via an automatically-detected queued
  connection.
- [`NapariVolumeProgress`][confusius._napari._registration._progress.NapariVolumeProgress]
  implements the
  [`RegistrationProgress`][confusius.registration.RegistrationProgress]
  protocol. It is constructed inside `register_volume` (i.e. on the worker
  thread) and resamples the moving image at every iteration using the current
  tentative transform, forwarding the resulting array to the bridge.

Connection lifecycle:

1. The panel constructs a `NapariProgressBridge` on the GUI thread and connects
   its `iterated` signal to a slot that writes the array into the resampled
   napari layer.
2. The panel builds a factory (via
   [`make_napari_progress_factory`][confusius._napari._registration._progress.make_napari_progress_factory])
   that closes over the bridge and returns a `NapariVolumeProgress` instance
   when called by `register_volume`.
3. `register_volume` instantiates the progress inside the worker thread and
   wires it to SimpleITK's iteration and end events as usual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from qtpy.QtCore import QObject, Signal

from confusius.registration._progress import _resample_intermediate

if TYPE_CHECKING:
    import SimpleITK as sitk

    from confusius.registration import RegistrationProgress


class NapariProgressBridge(QObject):
    """Thread-boundary signal bridge for napari registration progress.

    Construct this on the GUI thread before starting the registration worker.
    Connect `iterated` to a slot that mutates a napari layer (e.g. writes
    `layer.data = arr`); the slot will be invoked on the GUI thread thanks to
    Qt's automatic cross-thread connection. The bridge itself never touches the
    napari layer, keeping a clean separation between the worker's data path and
    the GUI update path.

    See Also
    --------
    NapariVolumeProgress : Worker-side reporter that emits via this bridge.
    """

    iterated = Signal(object)
    """:pyqtSignal: Emitted at every optimizer iteration with the resampled
    moving image as a numpy array in numpy axis order (matching `fixed`)."""

    metric_updated = Signal(float)
    """:pyqtSignal: Emitted at every optimizer iteration with the current
    optimizer metric value (a float)."""

    finished = Signal()
    """:pyqtSignal: Emitted once when the registration end event fires."""


class NapariVolumeProgress:
    """Napari-layer progress reporter for `register_volume`.

    Implements the
    [`RegistrationProgress`][confusius.registration.RegistrationProgress]
    protocol. Stores the registration method and SimpleITK images it needs to
    resample the moving image at each iteration. The resampled array is
    forwarded to the bridge via a Qt signal, so this object is safe to call
    from the SimpleITK command callback running on the worker thread.

    Parameters
    ----------
    bridge : NapariProgressBridge
        GUI-thread signal bridge. Stored by reference; never accessed for GUI
        APIs from this object.
    registration_method : SimpleITK.ImageRegistrationMethod
        Active registration method whose `GetInitialTransform` is used to
        resample the moving image at each iteration.
    fixed_img : SimpleITK.Image
        Fixed image defining the resample grid.
    moving_img : SimpleITK.Image
        Moving image to resample.
    plot_metric : bool, default: True
        Currently unused by the napari path; kept for signature compatibility
        with the matplotlib plotter factory.
    plot_composite : bool, default: True
        Currently unused by the napari path (the resampled layer *is* the
        composite view); kept for signature compatibility.
    resample_kwargs : dict, optional
        Extra keyword arguments forwarded to
        [`_resample_intermediate`][confusius.registration._progress._resample_intermediate].
        Must include `"default_value"`; `interpolation` defaults to `"linear"`.
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
        self._resample_kwargs = dict(resample_kwargs or {})
        # The resampled layer acts as the composite view; the matplotlib-style
        # composite overlay is always implied, regardless of plot_composite.
        self._plot_metric = plot_metric
        self._plot_composite = plot_composite

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
            self._resample_kwargs,
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


def make_napari_progress_factory(
    bridge: NapariProgressBridge,
) -> "Callable[..., RegistrationProgress]":
    """Return a progress-plotter factory bound to a bridge.

    The returned callable has the signature expected by `register_volume`'s
    `progress_plotter` argument — it accepts
    `(registration_method, fixed_img, moving_img, *, plot_metric,
    plot_composite, resample_kwargs)` and returns a
    [`NapariVolumeProgress`][confusius._napari._registration._progress.NapariVolumeProgress]
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
        """Build a NapariVolumeProgress wrapping the captured bridge.

        Parameters
        ----------
        registration_method : SimpleITK.ImageRegistrationMethod
            Active registration method whose transform is sampled at every
            iteration.
        fixed_img : SimpleITK.Image
            Fixed reference image defining the resample grid.
        moving_img : SimpleITK.Image
            Moving image to resample.
        plot_metric : bool, default: True
            Unused by the napari path; kept for signature compatibility.
        plot_composite : bool, default: True
            Unused by the napari path; kept for signature compatibility.
        resample_kwargs : dict, optional
            Extra keyword arguments forwarded to
            [`_resample_intermediate`][confusius.registration._progress._resample_intermediate].

        Returns
        -------
        RegistrationProgress
            Progress reporter ready to be wired to SimpleITK's iteration and
            end events by `register_volume`.
        """
        return NapariVolumeProgress(
            bridge,
            registration_method,
            fixed_img,
            moving_img,
            plot_metric=plot_metric,
            plot_composite=plot_composite,
            resample_kwargs=resample_kwargs,
        )

    return factory
