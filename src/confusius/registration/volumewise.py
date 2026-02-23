"""Volumewise registration for fUSI data."""

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration.motion import create_motion_dataframe
from confusius.registration.volume import register_volume


def register_volumewise(
    data: xr.DataArray,
    *,
    reference_time: int = 0,
    n_jobs: int = -1,
    transform: Literal["translation", "rigid", "affine"] = "rigid",
    metric: Literal["correlation", "mattes_mi"] = "correlation",
    number_of_histogram_bins: int = 50,
    learning_rate: float | Literal["auto"] = "auto",
    number_of_iterations: int = 100,
    convergence_minimum_value: float = 1e-6,
    convergence_window_size: int = 10,
    initialization: Literal["geometry", "moments", "none"] = "geometry",
    optimizer_weights: list[float] | None = None,
    use_multi_resolution: bool = False,
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    resample_interpolation: Literal["linear", "bspline"] = "linear",
) -> xr.DataArray:
    """Register all volumes in a fUSI recording to a reference volume.

    Parameters
    ----------
    data : xarray.DataArray
        Input data to register.
    reference_time : int, default: 0
        Index of the time point to use as registration target.
    n_jobs : int, default: -1
        Number of parallel jobs. -1 uses all available CPUs. Use 1 for serial
        processing.
    transform : {"translation", "rigid", "affine"}, default: "rigid"
        Transform model to use during registration. ``"translation"`` allows
        only shifts. ``"rigid"`` adds rotation. ``"affine"`` adds scaling and
        shearing. B-spline is not available for motion correction.
    metric : {"correlation", "mattes_mi"}, default: "correlation"
        Similarity metric. ``"correlation"`` (normalized cross-correlation) is
        appropriate for same-modality registration. ``"mattes_mi"`` (Mattes
        mutual information) is better suited for multi-modal registration or
        when the intensity relationship between images is non-linear.
    number_of_histogram_bins : int, default: 50
        Number of histogram bins used by Mattes mutual information. Only
        relevant when ``metric="mattes_mi"``.
    learning_rate : float or "auto", default: "auto"
        Optimizer step size in normalised units (after ``SetOptimizerScalesFromPhysicalShift``).
        ``"auto"`` re-estimates the rate at every iteration. A float uses that
        value directly; if registration diverges or fails to converge, reduce
        it.
    number_of_iterations : int, default: 100
        Maximum number of optimizer iterations.
    convergence_minimum_value : float, default: 1e-6
        Convergence threshold. Optimization stops early when the estimated
        energy profile falls below this value.
    convergence_window_size : int, default: 10
        Number of recent metric values used to estimate the energy profile
        for convergence checking.
    initialization : {"geometry", "moments", "none"}, default: "geometry"
        Transform initializer applied before optimization. ``"geometry"``
        aligns the image centers (safe default, no assumptions about content).
        ``"moments"`` aligns centers of mass (better when images are offset
        but share the same content). ``"none"`` uses the identity transform.
    optimizer_weights : list of float or None, default: None
        Per-parameter weights applied on top of the auto-estimated physical
        shift scales. ``None`` uses identity weights (all ones). A list is
        passed directly to SimpleITK's ``SetOptimizerWeights``; its length
        must match the number of transform parameters (3 for 2D rigid, 6 for
        3D rigid, 6 for 2D affine, 12 for 3D affine). The weight for each
        parameter is multiplied into the effective step size: ``0`` freezes a
        parameter entirely, values in ``(0, 1)`` slow it down, and ``1``
        leaves it unchanged. For the 3D Euler transform the parameter order is
        ``[angleX, angleY, angleZ, tx, ty, tz]``; to disable rotations around
        x and y set weights to ``[0, 0, 1, 1, 1, 1]``.
    use_multi_resolution : bool, default: False
        Whether to use a multi-resolution pyramid during registration. When
        ``True``, registration proceeds from a coarse downsampled version of
        the images to the full resolution, which improves convergence for large
        displacements and reduces the risk of local minima.
    shrink_factors : sequence of int, default: (6, 2, 1)
        Downsampling factor at each pyramid level, from coarsest to finest.
        Must have the same length as ``smoothing_sigmas``. Only used when
        ``use_multi_resolution=True``.
    smoothing_sigmas : sequence of int, default: (6, 2, 1)
        Gaussian smoothing sigma (in voxels) applied at each pyramid level,
        from coarsest to finest. Must have the same length as
        ``shrink_factors``. Only used when ``use_multi_resolution=True``.
    resample_interpolation : {"linear", "bspline"}, default: "linear"
        Interpolator used when resampling each volume onto the reference grid.
        ``"linear"`` is fast and appropriate for motion correction.
        ``"bspline"`` (3rd-order B-spline) produces smoother results at the
        cost of speed.

    Returns
    -------
    xarray.DataArray
        Registered data with same coordinates and attributes as input.
    """
    from joblib import Parallel, delayed
    from joblib_progress import joblib_progress

    if "time" not in data.dims:
        raise ValueError("Time dimension 'time' not found in data")

    data_moved = data.transpose("time", ...)

    if data_moved.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D data, got {data_moved.ndim}D")

    n_frames = data_moved.sizes["time"]
    ref_da = data_moved.isel(time=reference_time)

    with joblib_progress("Registering volumes...", total=n_frames):
        results = cast(
            "list[tuple[xr.DataArray, npt.NDArray[np.float64] | None]]",
            Parallel(n_jobs=n_jobs)(
                delayed(register_volume)(
                    volume,
                    ref_da,
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
                    resample=True,
                    resample_interpolation=resample_interpolation,
                    show_progress=False,
                )
                for volume in data_moved
            ),
        )

    arr = data_moved.values
    output = np.zeros_like(arr)
    affines: list[npt.NDArray[np.float64] | None] = []
    for t, (registered_da, frame_affine) in enumerate(results):
        output[t] = registered_da.values
        affines.append(frame_affine)

    time_coords = (
        data_moved.coords["time"].values if "time" in data_moved.coords else None
    )
    motion_df = create_motion_dataframe(
        affines=affines, reference=ref_da, time_coords=time_coords
    )

    result = xr.DataArray(
        output,
        coords=data_moved.coords,
        dims=data_moved.dims,
        attrs=data.attrs.copy(),
    )

    result.attrs["registration"] = "volumewise"
    result.attrs["reference_time"] = reference_time
    result.attrs["motion_params"] = motion_df

    return result.transpose(*data.dims)
