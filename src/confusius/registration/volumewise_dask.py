"""Dask-parallelized volumewise registration for fUSI data.

Prototype for comparison against
[`register_volumewise`][confusius.registration.register_volumewise]'s joblib
backend. See `benchmark/README.md` for the timing comparison.
"""

from collections.abc import Sequence
from typing import Literal

import dask
import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration._utils import validate_intensity_scaling
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.motion import create_motion_dataframe
from confusius.registration.volume import register_volume
from confusius.validation import ensure_voxeldata


def register_volumewise_dask(
    data: xr.DataArray,
    *,
    reference_time: int = 0,
    transform: Literal["translation", "rigid", "affine"] = "rigid",
    metric: Literal["correlation", "mattes_mi"] = "correlation",
    intensity_scaling: Literal["none", "db", "sqrt"] | float = "none",
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
    fill_value: float | None = None,
    sitk_threads: int = 1,
    keep_diagnostics: bool = False,
) -> xr.DataArray:
    """Register all volumes in a fUSI recording, parallelized with Dask.

    Same behavior/return value as
    [`register_volumewise`][confusius.registration.register_volumewise], minus
    `n_jobs`, `show_progress`, `progress_reporter`, and `abort_event` (dropped
    for this prototype — see `benchmark/README.md`). Parallelism and where
    frames execute (threads, processes, or a distributed cluster) is
    controlled entirely by the ambient Dask scheduler/`Client`, exactly like
    any other `dask.compute()` call.

    Unlike `register_volumewise`, this never materializes `data` up front:
    each frame is read via lazy `.isel(time=t)` indexing inside a
    `dask.delayed` task, so an h5py- or dask-backed input is only pulled into
    memory frame-by-frame as the scheduler executes tasks, not all at once
    before registration starts.

    Gzip-compressed NIfTI input is a notable exception: gzip has no random
    access, so every independent per-frame read against an unpersisted
    gzip-backed array re-decompresses from the start of the file, and Dask
    does not cache across separate reads. For that backing specifically,
    materializing once with `.compute()` (or persisting with `.persist()`,
    at the cost of losing the RAM benefit above) before calling this function
    is faster in practice — see confusius-tools/confusius#439 for
    measurements.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array with a `time` dimension to register.
    reference_time : int, default: 0
        Index of the time point to use as registration target.
    transform : {"translation", "rigid", "affine"}, default: "rigid"
        Transform model to use during registration. `"translation"` allows
        only shifts. `"rigid"` adds rotation. `"affine"` adds scaling and
        shearing. B-spline is not available for motion correction.
    metric : {"correlation", "mattes_mi"}, default: "correlation"
        Similarity metric. `"correlation"` (normalized cross-correlation) is
        appropriate for same-modality registration. `"mattes_mi"` (Mattes
        mutual information) is better suited for multi-modal registration or
        when the intensity relationship between images is non-linear.
    intensity_scaling : {"none", "db", "sqrt"} or float, default: "none"
        Intensity transform applied to the reference volume and to every frame, only
        for the registration optimizer. Floats apply power scaling with that
        exponent; `"sqrt"` is an alias for `0.5`. Returned/resampled data keeps the
        original input intensities.
    number_of_histogram_bins : int, default: 50
        Number of histogram bins used by Mattes mutual information. Only
        relevant when `metric="mattes_mi"`.
    learning_rate : float or "auto", default: 0.01
        Optimizer step size in normalised units (after `SetOptimizerScalesFromPhysicalShift`).
        `"auto"` re-estimates the rate at every iteration. A float uses that
        value directly; increase it for large inter-volume shifts, or reduce it if
        registration creates motion in otherwise stable data.
    number_of_iterations : int, default: 100
        Maximum number of optimizer iterations.
    convergence_minimum_value : float, default: 1e-6
        Convergence threshold. Optimization stops early when the estimated
        energy profile falls below this value.
    convergence_window_size : int, default: 10
        Number of recent metric values used to estimate the energy profile
        for convergence checking.
    initialization : {"center_geometry", "center_moments"}, default: "center_geometry"
        Initial transform mapping `fixed` to `moving` coordinates, applied before
        optimization:

        - `"center_geometry"`: aligns image centers.
        - `"center_moments"`: aligns centers of mass.
        - `None`: uses the identity transform.

    optimizer_weights : list of float, optional
        Per-parameter weights applied on top of the auto-estimated world shift
        scales. If not provided, identity weights are used. A list is passed directly to
        SimpleITK's `SetOptimizerWeights`; its length must match the number of transform
        parameters (3 for 2D rigid, 6 for 3D rigid, 6 for 2D affine, 12 for 3D affine).
        The weight for each parameter is multiplied into the effective step size: `0`
        freezes a parameter entirely, values in `(0, 1)` slow it down, and `1` leaves it
        unchanged. For the 3D Euler transform the parameter order is `[angleX, angleY,
        angleZ, tx, ty, tz]`; to disable rotations around x and y set weights to `[0, 0,
        1, 1, 1, 1]`.
    use_multi_resolution : bool, default: False
        Whether to use a multi-resolution pyramid during registration. When
        `True`, registration proceeds from a coarse downsampled version of
        the images to the full resolution, which improves convergence for large
        displacements and reduces the risk of local minima.
    shrink_factors : sequence of int, default: (6, 2, 1)
        Downsampling factor at each pyramid level, from coarsest to finest.
        Must have the same length as `smoothing_sigmas`. Only used when
        `use_multi_resolution=True`.
    smoothing_sigmas : sequence of int, default: (6, 2, 1)
        Gaussian smoothing sigma (in voxels) applied at each pyramid level,
        from coarsest to finest. Must have the same length as
        `shrink_factors`. Only used when `use_multi_resolution=True`.
    resample_interpolation : {"linear", "bspline"}, default: "linear"
        Interpolator used when resampling each volume onto the reference grid.
        `"linear"` is fast and appropriate for motion correction.
        `"bspline"` (3rd-order B-spline) produces smoother results at the
        cost of speed.
    fill_value : float, optional
        Fill value for voxels outside each moving volume's field of view after
        resampling. If not provided, defaults to that volume's minimum value.
    sitk_threads : int, default: 1
        SimpleITK threads per frame. Keep at 1 when running many frames
        concurrently to avoid over-subscribing the CPU.
    keep_diagnostics : bool, default: False
        Whether to keep the full per-frame `RegistrationDiagnostics` list on
        the returned DataArray under `attrs["registration_diagnostics"]`.

    Returns
    -------
    xarray.DataArray
        Registered data with the same coordinates as input, input attributes,
        and added motion metadata in `attrs["reference_time"]` and
        `attrs["motion_params"]`.
    """
    if "time" not in data.dims:
        raise ValueError("Time dimension 'time' not found in data")

    validate_intensity_scaling(intensity_scaling, "intensity_scaling")

    data_moved = ensure_voxeldata(
        data,
        require_time=True,
        allow_pose=False,
        allow_extra_dims=False,
    )

    n_frames = data_moved.sizes["time"]
    ref_da = data_moved.isel(time=reference_time)

    def _register_one(
        volume: xr.DataArray,
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], RegistrationDiagnostics
    ]:
        registered_da, frame_affine, frame_diag = register_volume(
            volume,
            ref_da,
            transform_type=transform,
            metric=metric,
            # The reference is a frame of the same recording, so one scaling
            # applies to both sides.
            fixed_intensity_scaling=intensity_scaling,
            moving_intensity_scaling=intensity_scaling,
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
            resample=True,
            resample_interpolation=resample_interpolation,
            fill_value=fill_value,
            sitk_threads=sitk_threads,
            show_progress=False,
            abort_event=None,
        )
        return registered_da.values, frame_affine, frame_diag

    delayed_register_one = dask.delayed(_register_one)
    tasks = [delayed_register_one(data_moved.isel(time=t)) for t in range(n_frames)]
    # Single dask.compute() call: the ambient scheduler (threaded, processes,
    # or a distributed Client) decides how many frames run concurrently and
    # streams each frame's isel(time=t) read from the underlying lazy backing
    # store, instead of the eager `data_moved.values` full-array load used by
    # the joblib backend.
    results = dask.compute(*tasks)

    output = np.stack([registered for registered, _, _ in results])
    affines = [affine for _, affine, _ in results]
    diagnostics = [diag for _, _, diag in results]

    time_coords = (
        data_moved.coords["time"].values if "time" in data_moved.coords else None
    )
    motion_df = create_motion_dataframe(
        affines=affines, reference=ref_da, time_coords=time_coords
    )
    motion_df["final_metric_value"] = [d.final_metric_value for d in diagnostics]
    motion_df["n_iterations"] = [d.n_iterations for d in diagnostics]
    motion_df["status"] = [d.status for d in diagnostics]

    result = xr.DataArray(
        output,
        coords=data_moved.coords,
        dims=data_moved.dims,
        attrs=data.attrs.copy(),
    )
    result.attrs["reference_time"] = reference_time
    result.attrs["motion_params"] = motion_df
    if keep_diagnostics:
        result.attrs["registration_diagnostics"] = diagnostics

    return result.transpose(*data.dims)
