"""Volumewise registration for fUSI data."""

from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.io import is_h5py_backed
from confusius.registration._utils import validate_intensity_scaling
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.motion import create_motion_dataframe
from confusius.registration.volume import register_volume
from confusius.validation import ensure_voxeldata
from confusius.validation.mask import check_spatial_alignment

if TYPE_CHECKING:
    from threading import Event

    from confusius.registration.volumewise_progress import VolumewiseProgressReporter


def register_volumewise(
    data: xr.DataArray,
    *,
    reference_time: int | None = None,
    fixed: xr.DataArray | None = None,
    n_jobs: int = -1,
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
    show_progress: bool = True,
    progress_reporter: "VolumewiseProgressReporter | None" = None,
    abort_event: "Event | None" = None,
    keep_diagnostics: bool = False,
) -> xr.DataArray:
    """Register all volumes in a fUSI recording to a reference volume.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array with a `time` dimension to register.
    reference_time : int, optional
        Index of the time point of `data` to use as registration target. If not
        provided, and `fixed` is not provided either, the first time point (`0`)
        is used. Cannot be combined with `fixed`.
    fixed : xarray.DataArray, optional
        Spatial-only VoxelData array to register every frame to, for example the
        mean of a few low-motion frames of `data`. Must share `data`'s voxel grid
        (same `k`/`j`/`i` coordinates and voxel-to-world affine) and have no
        `time` dimension. `intensity_scaling` is applied to it as to any reference
        frame. If not provided, the frame at `reference_time` is used. Cannot be
        combined with `reference_time`.
    n_jobs : int, default: -1
        Number of parallel jobs. Negative values resolve to `max(1, os.cpu_count() + 1
        + n_jobs)`, so `-1` means all CPUs, `-2` means all minus one, and so on.
        Use `1` for serial processing.
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
    show_progress : bool, default: True
        Whether to display a progress bar while registering volumes.
    progress_reporter : VolumewiseProgressReporter, optional
        Thread-safe reporter notified whenever one frame completes. Useful for GUI
        progress bars or progressively filling an output layer while frames finish.
    abort_event : threading.Event, optional
        Cooperative cancellation flag shared across frames. If set before or during
        execution, in-flight frame registrations stop at the next optimiser iteration
        boundary and this function returns the partial dataset collected so far. Frames
        that were not started are left blank (filled with the data minimum), and
        per-frame `motion_params` rows are marked via the diagnostics status.
    keep_diagnostics : bool, default: False
        Whether to keep the full per-frame
        [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
        list on the returned DataArray under
        `attrs["registration_diagnostics"]`. Disabled by default because each
        diagnostics object carries the full optimizer metric trace, which adds
        up over long recordings. The cheap per-frame summaries
        (`final_metric_value`, `n_iterations`) are always added to
        `motion_params` regardless of this flag.

    Returns
    -------
    xarray.DataArray
        Registered data with the same coordinates as input, input attributes,
        and added motion metadata in `attrs["reference_time"]` (the index of the
        frame used as target, or `None` when `fixed` was given) and
        `attrs["motion_params"]`. `motion_params` always carries per-frame
        `final_metric_value`, `n_iterations`, and `status` columns. When
        `keep_diagnostics=True`, `attrs["registration_diagnostics"]` also
        carries a list of
        [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
        (one entry per frame) with the full per-iteration metric trace.

    Raises
    ------
    ValueError
        If both `reference_time` and `fixed` are given, or if `fixed` has a `time`
        dimension, is not a VoxelData array, or does not share `data`'s voxel
        grid.
    TypeError
        If `n_jobs != 1` and `data` is backed by an h5py dataset. See Notes.

    Notes
    -----
    SCAN files are HDF5 files loaded lazily via h5py. h5py datasets cannot be pickled,
    so they cannot be passed to joblib workers for parallel processing. Materialize the
    data before calling this function:

    ```python
    import confusius as cf

    fusi = cf.load("recording.scan").compute()  # load into memory first
    fusi = cf.registration.register_volumewise(fusi)
    ```

    Alternatively, use `n_jobs=1` for serial processing (slower but works with lazy
    SCAN data).
    """
    from joblib import Parallel, delayed

    if "time" not in data.dims:
        raise ValueError("Time dimension 'time' not found in data")

    if reference_time is not None and fixed is not None:
        raise ValueError("Pass either 'reference_time' or 'fixed', not both.")

    validate_intensity_scaling(intensity_scaling, "intensity_scaling")

    if n_jobs != 1 and is_h5py_backed(data):
        raise TypeError(
            "Data is backed by an h5py dataset, which cannot be serialized for "
            "parallel processing with joblib. Call .compute() to materialize the "
            "data into memory before calling register_volumewise, or use n_jobs=1 "
            "for serial processing."
        )

    data_moved = ensure_voxeldata(
        data,
        require_time=True,
        allow_pose=False,
        allow_extra_dims=False,
    )

    n_frames = data_moved.sizes["time"]
    if fixed is None:
        reference_time = 0 if reference_time is None else reference_time
        ref_da = data_moved.isel(time=reference_time)
    else:
        if "time" in fixed.dims:
            raise ValueError(
                "'fixed' must be a spatial-only VoxelData array without a time "
                f"dimension; got dims {fixed.dims}."
            )
        ref_da = ensure_voxeldata(
            fixed, require_time=False, allow_pose=False, allow_extra_dims=False
        )
        # Output reuses `data`'s shape and coordinates, so the target must live on
        # the same voxel grid.
        check_spatial_alignment(ref_da, data_moved, "'fixed'")
    # Materialize the target once: a lazy `fixed` (for example a dask mean of a
    # lazily loaded recording) would otherwise be recomputed for every frame, and
    # an h5py-backed one cannot be sent to joblib workers at all.
    ref_da = ref_da.compute()

    progress_context = nullcontext()
    if show_progress:
        from joblib_progress import joblib_progress

        progress_context = joblib_progress("Registering volumes...", total=n_frames)

    parallel_kwargs: dict[str, object] = {"n_jobs": n_jobs}
    if abort_event is not None or progress_reporter is not None:
        # Use threads when cancellation or progress reporting is enabled so
        # every worker sees the shared reporter / event instance.
        parallel_kwargs["prefer"] = "threads"

    aborted_affine = np.eye(ref_da.ndim + 1, dtype=float)
    aborted_diagnostics = RegistrationDiagnostics(
        metric=metric,
        metric_values=np.empty(0, dtype=float),
        final_metric_value=float("nan"),
        n_iterations=0,
        stop_condition="Registration aborted before frame started.",
        status="aborted",
    )

    def _register_one(
        frame_index: int,
        volume: xr.DataArray,
    ) -> tuple[int, xr.DataArray, npt.NDArray[np.floating], RegistrationDiagnostics]:
        # Once aborted, skip cheaply: building SimpleITK images and resampling is
        # pure-Python/GIL-bound work that, multiplied across joblib threads, starves the
        # GUI thread. Return the original frame with a zero-iteration "aborted"
        # diagnostic instead.
        if abort_event is not None and abort_event.is_set():
            return (
                frame_index,
                volume,
                aborted_affine.copy(),
                aborted_diagnostics,
            )

        registered_da, frame_affine, frame_diag = register_volume(
            volume,
            ref_da,
            transform_type=transform,
            metric=metric,
            # The reference is a frame of the same recording, or a user-provided
            # volume of the same modality, so one scaling applies to both sides.
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
            # Restrict SimpleITK to 1 thread per worker to avoid
            # over-subscribing the CPU when joblib spawns many workers.
            sitk_threads=1,
            show_progress=False,
            abort_event=abort_event,
        )
        return frame_index, registered_da, frame_affine, frame_diag

    arr = data_moved.values
    # Aborted/un-started frames are left blank (filled with the data minimum,
    # i.e. background) rather than copying the unregistered input, so the partial
    # result visibly shows which frames were skipped.
    output = np.full_like(arr, arr.min())
    affines: list[npt.NDArray[np.floating]] = [
        aborted_affine.copy() for _ in range(n_frames)
    ]
    final_metric_values = [float("nan")] * n_frames
    n_iterations_per_frame = [0] * n_frames
    statuses = ["aborted"] * n_frames
    diagnostics = [aborted_diagnostics] * n_frames

    try:
        with progress_context:
            results = Parallel(return_as="generator_unordered", **parallel_kwargs)(
                delayed(_register_one)(t, volume)
                for t, volume in enumerate(data_moved)
                if abort_event is None or not abort_event.is_set()
            )
            for t, registered_da, frame_affine, frame_diag in results:
                skipped = (
                    frame_diag.status == "aborted" and frame_diag.n_iterations == 0
                )
                if not skipped:
                    output[t] = registered_da.values
                affines[t] = frame_affine
                final_metric_values[t] = frame_diag.final_metric_value
                n_iterations_per_frame[t] = frame_diag.n_iterations
                statuses[t] = frame_diag.status
                diagnostics[t] = frame_diag
                if progress_reporter is not None:
                    progress_reporter.frame_completed(t, registered_da, frame_diag)
    finally:
        if progress_reporter is not None:
            progress_reporter.close()

    time_coords = (
        data_moved.coords["time"].values if "time" in data_moved.coords else None
    )
    motion_df = create_motion_dataframe(
        affines=affines, reference=ref_da, time_coords=time_coords
    )

    # Per-frame summary columns are cheap (one float / int each) and useful
    # for spotting frames that failed to converge, so we always keep them.
    motion_df["final_metric_value"] = final_metric_values
    motion_df["n_iterations"] = n_iterations_per_frame
    motion_df["status"] = statuses

    result = xr.DataArray(
        output,
        coords=data_moved.coords,
        dims=data_moved.dims,
        attrs=data.attrs.copy(),
    )

    result.attrs["reference_time"] = reference_time
    result.attrs["motion_params"] = motion_df
    if keep_diagnostics:
        # The full diagnostics list carries every frame's optimizer metric
        # trace, which adds up over long recordings, so it is gated behind the flag.
        result.attrs["registration_diagnostics"] = list(diagnostics)

    return result.transpose(*data.dims)
