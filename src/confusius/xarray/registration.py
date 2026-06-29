"""Xarray accessor for registration."""

from collections.abc import Callable, Sequence
from threading import Event
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration.progress import RegistrationProgress
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.volume import register_volume
from confusius.registration.volumewise import register_volumewise
from confusius.registration.volumewise_progress import VolumewiseProgressReporter


class FUSIRegistrationAccessor:
    """Accessor for registration operations on fUSI data.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.open_zarr("output.zarr")["power_doppler"]
    >>> registered = data.fusi.register.volumewise(reference_time=0)
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def to_volume(
        self,
        fixed: xr.DataArray,
        *,
        fixed_mask: xr.DataArray | None = None,
        moving_mask: xr.DataArray | None = None,
        transform: Literal["translation", "rigid", "affine", "bspline"] = "rigid",
        metric: Literal["correlation", "mattes_mi"] = "correlation",
        number_of_histogram_bins: int = 50,
        learning_rate: float | Literal["auto"] = "auto",
        number_of_iterations: int = 100,
        convergence_minimum_value: float = 1e-6,
        convergence_window_size: int = 10,
        initialization: Literal["center_geometry", "center_moments"]
        | npt.NDArray[np.floating]
        | None = "center_geometry",
        optimizer_weights: list[float] | None = None,
        mesh_size: tuple[int, int, int] = (10, 10, 10),
        use_multi_resolution: bool = False,
        shrink_factors: Sequence[int] = (6, 2, 1),
        smoothing_sigmas: Sequence[int] = (6, 2, 1),
        resample: bool = False,
        resample_interpolation: Literal["linear", "bspline"] = "linear",
        fill_value: float | None = None,
        sitk_threads: int = -1,
        show_progress: bool = False,
        plot_metric: bool = True,
        plot_composite: bool = True,
        progress_plotter: Callable[..., RegistrationProgress] | None = None,
        abort_event: Event | None = None,
    ) -> "tuple[xr.DataArray, npt.NDArray[np.floating] | xr.DataArray | None, RegistrationDiagnostics]":  # noqa: E501
        """Register this volume to a fixed reference volume.

        Parameters
        ----------
        fixed : xarray.DataArray
            Reference volume to register to.
        fixed_mask : xarray.DataArray, optional
            Boolean mask for the fixed volume.
        moving_mask : xarray.DataArray, optional
            Boolean mask for this moving volume.
        transform : {"translation", "rigid", "affine", "bspline"}, default: "rigid"
            Type of transform to use for registration.
        metric : {"correlation", "mattes_mi"}, default: "correlation"
            Similarity metric for registration.
        number_of_histogram_bins : int, default: 50
            Number of histogram bins (only used when `metric="mattes_mi"`).
        learning_rate : float or "auto", default: "auto"
            Optimizer step size in normalised units (after
            `SetOptimizerScalesFromPhysicalShift`). `"auto"` re-estimates the rate at
            every iteration. A float uses that value directly; if registration diverges
            or fails to converge, reduce it.
        number_of_iterations : int, default: 100
            Maximum number of optimizer iterations.
        convergence_minimum_value : float, default: 1e-6
            Convergence threshold for early stopping.
        convergence_window_size : int, default: 10
            Window size for convergence check.
        initialization : {"center_geometry", "center_moments"} or (N+1, N+1) numpy.ndarray, default: "center_geometry"
            Initial transform mapping `fixed` to `moving` coordinates, applied before
            optimization:

            - `"center_geometry"`: aligns image centers.
            - `"center_moments"`: aligns centers of mass.
            - `(N+1, N+1)` homogeneous affine matrix: uses a precomputed affine
              transform.
            - `None`: uses the identity transform.

            For `transform="bspline"`, centering modes are ignored but affine
            initialization is supported.
        optimizer_weights : list of float, optional
            Per-parameter weights applied on top of auto-estimated scales via
            `SetOptimizerWeights()`. If not provided, no additional weighting is
            applied. The weight for each parameter is multiplied into the effective step
            size: `0` freezes a parameter, values in `(0, 1)` slow it down, `1` leaves
            it unchanged. For the 3D Euler transform the order is `[angleX, angleY,
            angleZ, tx, ty, tz]`; to disable rotations around x and y use `[0, 0, 1, 1,
            1, 1]`.
        mesh_size : tuple of int, default: (10, 10, 10)
            BSpline mesh size. Only used when `transform="bspline"`.
        use_multi_resolution : bool, default: False
            Whether to use a multi-resolution pyramid during registration.
        shrink_factors : sequence of int, default: (6, 2, 1)
            Downsampling factor at each pyramid level, from coarsest to
            finest. Only used when `use_multi_resolution=True`.
        smoothing_sigmas : sequence of int, default: (6, 2, 1)
            Gaussian smoothing sigma (in voxels) at each pyramid level, from
            coarsest to finest. Only used when `use_multi_resolution=True`.
        resample : bool, default: False
            Whether to resample the moving volume into the fixed volume's
            space. When `False` (the default), only the transform is
            estimated and the moving volume is returned unchanged.
        resample_interpolation : {"linear", "bspline"}, default: "linear"
            Interpolation method used for the final resample step.
        fill_value : float, optional
            Fill value for voxels outside the moving image's field of view after
            resampling. If not provided, defaults to the minimum of the moving
            image. See [`register_volume`][confusius.registration.register_volume].
        sitk_threads : int, default: -1
            Number of threads SimpleITK may use internally.
        show_progress : bool, default: False
            Whether to display a live progress plot during registration.
        plot_metric : bool, default: True
            Whether to include the optimizer metric curve in the progress
            plot. Ignored when `show_progress=False`.
        plot_composite : bool, default: True
            Whether to include a fixed/moving composite overlay in the
            progress plot. Ignored when `show_progress=False`.
        progress_plotter : callable, optional
            Custom progress reporter factory. See
            [`register_volume`][confusius.registration.register_volume].
        abort_event : threading.Event, optional
            Cooperative cancellation flag.

        Returns
        -------
        registered : xarray.DataArray
            Registered volume. When `resample=True`, resampled onto the
            fixed grid; otherwise the original moving volume with registration
            metadata added.
        affine : (N+1, N+1) numpy.ndarray or xarray.DataArray or None
            Estimated registration transform.  For linear transforms, a
            homogeneous affine matrix.  For `transform="bspline"`, a DataArray
            encoding the B-spline control-point grid.
        diagnostics : confusius.registration.RegistrationDiagnostics
            Per-iteration metric values and optimizer stop condition. See
            [`register_volume`][confusius.registration.register_volume].

        Examples
        --------
        >>> moving.fusi.register.to_volume(fixed)
        >>> moving.fusi.register.to_volume(fixed, resample=True)
        >>> moving.fusi.register.to_volume(fixed, show_progress=True)
        """
        return register_volume(
            self._obj,
            fixed,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            transform_type=transform,
            metric=metric,
            number_of_histogram_bins=number_of_histogram_bins,
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations,
            convergence_minimum_value=convergence_minimum_value,
            convergence_window_size=convergence_window_size,
            initialization=initialization,
            optimizer_weights=optimizer_weights,
            mesh_size=mesh_size,
            use_multi_resolution=use_multi_resolution,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            resample=resample,
            resample_interpolation=resample_interpolation,
            fill_value=fill_value,
            sitk_threads=sitk_threads,
            show_progress=show_progress,
            plot_metric=plot_metric,
            plot_composite=plot_composite,
            progress_plotter=cast(
                "Callable[..., RegistrationProgress] | None", progress_plotter
            ),
            abort_event=abort_event,
        )

    def volumewise(
        self,
        *,
        reference_time: int = 0,
        n_jobs: int = -1,
        transform: Literal["translation", "rigid", "affine"] = "rigid",
        metric: Literal["correlation", "mattes_mi"] = "correlation",
        number_of_histogram_bins: int = 50,
        learning_rate: float | Literal["auto"] = 0.01,
        number_of_iterations: int = 100,
        convergence_minimum_value: float = 1e-6,
        convergence_window_size: int = 10,
        initialization: Literal["center_geometry", "center_moments"]
        | None = "center_geometry",
        optimizer_weights: list[float] | None = None,
        use_multi_resolution: bool = False,
        shrink_factors: Sequence[int] = (6, 2, 1),
        smoothing_sigmas: Sequence[int] = (6, 2, 1),
        resample_interpolation: Literal["linear", "bspline"] = "linear",
        show_progress: bool = True,
        progress_reporter: VolumewiseProgressReporter | None = None,
        abort_event: Event | None = None,
        keep_diagnostics: bool = False,
    ) -> xr.DataArray:
        """Register all volumes to a reference time point.

        Parameters
        ----------
        reference_time : int, default: 0
            Index of the time point to use as registration target.
        n_jobs : int, default: -1
            Number of parallel jobs. -1 uses all available CPUs.
            Use 1 for serial processing.
        transform : {"translation", "rigid", "affine"}, default: "rigid"
            Type of transform to use for registration.
        metric : {"correlation", "mattes_mi"}, default: "correlation"
            Similarity metric for registration.
        number_of_histogram_bins : int, default: 50
            Number of histogram bins (only used when `metric="mattes_mi"`).
        learning_rate : float or "auto", default: 0.01
            Optimizer step size in normalised units (after
            `SetOptimizerScalesFromPhysicalShift`). `"auto"` re-estimates the rate at
            every iteration. A float uses that value directly; if registration diverges
            or fails to converge, reduce it.
        number_of_iterations : int, default: 100
            Maximum number of optimizer iterations.
        convergence_minimum_value : float, default: 1e-6
            Convergence threshold for early stopping.
        convergence_window_size : int, default: 10
            Window size for convergence check.
        initialization : {"center_geometry", "center_moments"}, default: "center_geometry"
            Initial transform mapping `fixed` to `moving` coordinates, applied before
            optimization:

            - `"center_geometry"`: aligns image centers.
            - `"center_moments"`: aligns centers of mass.
            - `None`: uses the identity transform.

        optimizer_weights : list of float, optional
            Per-parameter weights applied on top of auto-estimated scales via
            `SetOptimizerWeights()`. If not provided, no additional weighting is
            applied. The weight for each parameter is multiplied into the effective step
            size: `0` freezes a parameter, values in `(0, 1)` slow it down, `1` leaves
            it unchanged. For the 3D Euler transform the order is `[angleX, angleY,
            angleZ, tx, ty, tz]`; to disable rotations around x and y use `[0, 0, 1, 1,
            1, 1]`.
        use_multi_resolution : bool, default: False
            Whether to use a multi-resolution pyramid during registration.
        shrink_factors : sequence of int, default: (6, 2, 1)
            Downsampling factor at each pyramid level, from coarsest to
            finest. Only used when `use_multi_resolution=True`.
        smoothing_sigmas : sequence of int, default: (6, 2, 1)
            Gaussian smoothing sigma (in voxels) at each pyramid level, from
            coarsest to finest. Only used when `use_multi_resolution=True`.
        resample_interpolation : {"linear", "bspline"}, default: "linear"
            Interpolation method used for the final resample step.
        show_progress : bool, default: True
            Whether to display a progress bar while registering volumes.
        progress_reporter : VolumewiseProgressReporter, optional
            Thread-safe reporter notified whenever one frame completes.
        abort_event : threading.Event, optional
            Cooperative cancellation flag shared across frames.
        keep_diagnostics : bool, default: False
            Whether to keep per-frame registration diagnostics on the result.
            See
            [`register_volumewise`][confusius.registration.register_volumewise]
            for the full description.

        Returns
        -------
        xarray.DataArray
            Registered data with same coordinates and attributes.

        Examples
        --------
        >>> data.fusi.register.volumewise(reference_time=0)
        >>> data.fusi.register.volumewise(reference_time=0, transform="translation")
        """
        return register_volumewise(
            self._obj,
            reference_time=reference_time,
            n_jobs=n_jobs,
            transform=transform,
            metric=metric,
            number_of_histogram_bins=number_of_histogram_bins,
            learning_rate=learning_rate,
            number_of_iterations=number_of_iterations,
            convergence_minimum_value=convergence_minimum_value,
            convergence_window_size=convergence_window_size,
            initialization=initialization,
            optimizer_weights=optimizer_weights,
            use_multi_resolution=use_multi_resolution,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            resample_interpolation=resample_interpolation,
            show_progress=show_progress,
            progress_reporter=progress_reporter,
            abort_event=abort_event,
            keep_diagnostics=keep_diagnostics,
        )
