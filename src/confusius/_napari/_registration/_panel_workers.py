"""Worker entry points for the napari registration panel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Event
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from confusius._napari._registration._progress import NapariRegistrationProgressReporter
from confusius.registration import register_volume, register_volumewise

if TYPE_CHECKING:
    import numpy.typing as npt

    from confusius.registration import RegistrationDiagnostics, RegistrationProgress


def _run_register_volume(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    transform_type: Literal["translation", "rigid", "affine", "bspline"],
    metric: Literal["correlation", "mattes_mi"],
    learning_rate: float | Literal["auto"],
    number_of_iterations: int,
    use_multi_resolution: bool,
    resample_interpolation: Literal["linear", "bspline"],
    mesh_size: tuple[int, int, int] = (10, 10, 10),
    number_of_histogram_bins: int = 50,
    convergence_minimum_value: float = 1e-6,
    convergence_window_size: int = 10,
    center_initialization: Literal["center_geometry", "center_moments"]
    | None = "center_geometry",
    initial_transform: "npt.NDArray[np.floating]" | None = None,
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    fill_value: float | None = None,
    progress_plotter: Callable[..., "RegistrationProgress"] | None = None,
    abort_event: Event | None = None,
) -> tuple[
    xr.DataArray,
    "npt.NDArray[np.floating]" | xr.DataArray,
    "RegistrationDiagnostics",
]:
    """Run `register_volume` from the GUI.

    Parameters
    ----------
    moving : xarray.DataArray
        Moving volume.
    fixed : xarray.DataArray
        Fixed reference volume.
    transform_type : {"translation", "rigid", "affine", "bspline"}
        Registration model.
    metric : {"correlation", "mattes_mi"}
        Similarity metric.
    learning_rate : float or {"auto"}
        Optimizer learning rate.
    number_of_iterations : int
        Maximum number of optimizer iterations.
    use_multi_resolution : bool
        Whether to enable the registration pyramid.
    resample_interpolation : {"linear", "bspline"}
        Interpolator for the resampled output.
    mesh_size : tuple of int, default: (10, 10, 10)
        B-spline mesh size.
    number_of_histogram_bins : int
        Histogram bins for Mattes MI metric.
    convergence_minimum_value : float
        Convergence threshold.
    convergence_window_size : int
        Window size for convergence estimation.
    center_initialization : {"center_geometry", "center_moments"} or None
        Center-based transform initializer.
    initial_transform : numpy.ndarray, optional
        Pre-computed affine transform used as a warm start before optimization.
    shrink_factors : sequence of int
        Shrink factors per resolution level.
    smoothing_sigmas : sequence of int
        Smoothing sigmas per resolution level.
    fill_value : float or None
        Fill value for resampled output outside input domain.
    progress_plotter : callable, optional
        Optional progress-plotter factory forwarded to `register_volume`.
    abort_event : threading.Event, optional
        Cooperative cancellation flag forwarded to `register_volume`.

    Returns
    -------
    registered : xarray.DataArray
        Resampled registered volume.
    transform : numpy.ndarray or xarray.DataArray
        Estimated transform.
    diagnostics : confusius.registration.RegistrationDiagnostics
        Optimizer diagnostics.
    """
    return register_volume(
        moving,
        fixed,
        transform_type=transform_type,
        metric=metric,
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
        use_multi_resolution=use_multi_resolution,
        resample=True,
        resample_interpolation=resample_interpolation,
        mesh_size=mesh_size,
        number_of_histogram_bins=number_of_histogram_bins,
        convergence_minimum_value=convergence_minimum_value,
        convergence_window_size=convergence_window_size,
        initialization=(
            center_initialization if initial_transform is None else initial_transform
        ),
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        fill_value=fill_value,
        show_progress=progress_plotter is not None,
        progress_plotter=progress_plotter,
        abort_event=abort_event,
    )


def _run_register_volumewise(
    data: xr.DataArray,
    *,
    reference_time: int,
    n_jobs: int,
    transform: Literal["translation", "rigid", "affine"],
    metric: Literal["correlation", "mattes_mi"],
    learning_rate: float | Literal["auto"] = 0.01,
    number_of_iterations: int = 100,
    use_multi_resolution: bool,
    resample_interpolation: Literal["linear", "bspline"],
    number_of_histogram_bins: int = 50,
    convergence_minimum_value: float = 1e-6,
    convergence_window_size: int = 10,
    initialization: Literal["center_geometry", "center_moments"]
    | None = "center_geometry",
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    keep_diagnostics: bool = False,
    abort_event: Event | None = None,
    progress_reporter: NapariRegistrationProgressReporter | None = None,
) -> xr.DataArray:
    """Run `register_volumewise` from the GUI.

    Parameters
    ----------
    data : xarray.DataArray
        Time-series data to motion-correct.
    reference_time : int
        Reference frame index.
    n_jobs : int
        Number of joblib workers to use.
    transform : {"translation", "rigid", "affine"}
        Registration model.
    metric : {"correlation", "mattes_mi"}
        Similarity metric.
    learning_rate : float or {"auto"}, default: 0.01
        Optimizer learning rate.
    number_of_iterations : int
        Maximum number of optimizer iterations per frame.
    use_multi_resolution : bool
        Whether to enable the registration pyramid.
    resample_interpolation : {"linear", "bspline"}
        Interpolator for the resampled output.
    number_of_histogram_bins : int
        Histogram bins for Mattes MI metric.
    convergence_minimum_value : float
        Convergence threshold.
    convergence_window_size : int
        Window size for convergence estimation.
    initialization : {"center_geometry", "center_moments"} or None
        Transform initializer.
    shrink_factors : tuple of int or None
        Shrink factors per resolution level.
    smoothing_sigmas : tuple of int or None
        Smoothing sigmas per resolution level.
    keep_diagnostics : bool
        Store detailed optimization diagnostics.
    abort_event : threading.Event, optional
        Cooperative cancellation flag forwarded to `register_volumewise`.
    progress_reporter : NapariRegistrationProgressReporter, optional
        GUI-thread bridge-backed reporter forwarded to `register_volumewise`.

    Returns
    -------
    xarray.DataArray
        Registered time series.
    """
    return register_volumewise(
        data,
        reference_time=reference_time,
        n_jobs=n_jobs,
        transform=transform,
        metric=metric,
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
        use_multi_resolution=use_multi_resolution,
        resample_interpolation=resample_interpolation,
        number_of_histogram_bins=number_of_histogram_bins,
        convergence_minimum_value=convergence_minimum_value,
        convergence_window_size=convergence_window_size,
        initialization=initialization,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        keep_diagnostics=keep_diagnostics,
        show_progress=False,
        abort_event=abort_event,
        progress_reporter=progress_reporter,
    )
