"""Volumewise registration for fUSI data."""

import logging
import os
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from distributed import Client, as_completed, get_client
from rich.progress import Progress

from confusius._utils.io import is_h5py_backed
from confusius.registration._utils import validate_intensity_scaling
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.motion import create_motion_dataframe
from confusius.registration.volume import register_volume
from confusius.validation import ensure_voxeldata

if TYPE_CHECKING:
    from collections.abc import Callable

    from distributed import Event

    from confusius.registration.volumewise_progress import VolumewiseProgressReporter


def _default_worker_count() -> int:
    """Return a portable default worker count for an auto-created local Client.

    Returns
    -------
    int
        `len(os.sched_getaffinity(0))` where available (respects CPU affinity/cgroup
        limits, e.g. containers), otherwise `os.cpu_count()`. Always at least 1.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


@contextmanager
def _volumewise_client():
    """Yield a `submit` callable bound to an active `distributed.Client`.

    Yields
    ------
    Callable
        `submit(fn, *args)`, returning a `distributed.Future`.

    Notes
    -----
    Reuses the ambient client (`distributed.get_client()`) if one is active,
    otherwise creates a local one for the duration of this call, with one worker
    process per CPU.
    """

    def _submit_impure(client: Client, fn: "Callable[..., object]", /, *args: object):
        # pure=False: each call is a distinct piece of work (register_volumewise has
        # no caching/memoization concept), so never let distributed deduplicate two
        # submissions that happen to hash the same.
        return client.submit(fn, *args, pure=False)

    try:
        client = get_client()
        yield partial(_submit_impure, client)
        return
    except ValueError:
        pass

    # silence_logs: an auto-created local cluster torn down right after this call
    # can race a scheduled worker heartbeat, logging a benign but alarming-looking
    # CommClosedError traceback. Not relevant to the caller. dashboard_address=":0":
    # an ephemeral port, so this doesn't collide with another dashboard (this
    # session's own, or a concurrent call's) bound to the default :8787.
    with Client(
        n_workers=_default_worker_count(),
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.ERROR,
        dashboard_address=":0",
    ) as client:
        yield partial(_submit_impure, client)


def register_volumewise(
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
    show_progress : bool, default: True
        Whether to display a progress bar while registering volumes.
    progress_reporter : VolumewiseProgressReporter, optional
        Thread-safe reporter notified whenever one frame completes. Useful for GUI
        progress bars or progressively filling an output layer while frames finish.
    abort_event : distributed.Event, optional
        Cooperative cancellation flag shared across frames. Must be a `distributed.
        Event` (not a `threading.Event`) so that a live update is visible from a
        frame already running on a separate worker process -- see Notes. If set
        before or during execution, frames not yet dispatched are skipped, and
        in-flight frames stop at the next optimiser iteration boundary; this
        function then returns the partial dataset collected so far. Frames that
        were not started are left blank (filled with the data minimum), and
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
        and added motion metadata in `attrs["reference_time"]` and
        `attrs["motion_params"]`. `motion_params` always carries per-frame
        `final_metric_value`, `n_iterations`, and `status` columns. When
        `keep_diagnostics=True`, `attrs["registration_diagnostics"]` also
        carries a list of
        [`RegistrationDiagnostics`][confusius.registration.RegistrationDiagnostics]
        (one entry per frame) with the full per-iteration metric trace.

    Raises
    ------
    TypeError
        If `data` is backed by an h5py dataset, which cannot be pickled to send it
        to a Dask worker. See Notes.

    Notes
    -----
    Frames are registered in parallel through a Dask `distributed.Client`: if one
    is already active (`distributed.get_client()`), its workers are used as-is;
    otherwise a local `Client` is created for the duration of this call, using one
    worker process per CPU. To control parallelism explicitly -- e.g. to size a
    cluster, or to reuse one `Client` across several calls -- create and activate a
    `distributed.Client` yourself before calling this function.

    `abort_event` must be a `distributed.Event` rather than a `threading.Event`
    because `Client.submit` always pickles the submitted call to send it across the
    scheduler's message-passing protocol, even for an in-process/thread-based local
    cluster -- a live `threading.Event` cannot be pickled at all (nor can it stay
    live once pickled), whereas `distributed.Event` is itself backed by the
    scheduler and stays live across worker processes. Constructing one
    (`distributed.Event()`) requires an active `distributed.Client`, so create one
    yourself before calling this function if you need `abort_event`.

    Each frame is read via lazy `data.isel(time=t)` indexing, so a
    Dask-backed `data` is only pulled into memory frame-by-frame as each worker
    executes its task, not all at once before registration starts.
    Gzip-compressed NIfTI input is a notable exception: gzip has no random
    access, so every independent per-frame read against an unpersisted
    gzip-backed array re-decompresses from the start of the file. For that
    backing specifically, materializing once with `.compute()` (or persisting
    with `.persist()`) before calling this function is faster in practice --
    see confusius-tools/confusius#439 for measurements.

    SCAN files are HDF5 files loaded lazily via h5py. h5py datasets cannot be
    pickled, so a `distributed.Client` can never send one to a worker process.
    Materialize the data before calling this function:

    ```python
    import confusius as cf

    fusi = cf.load("recording.scan").compute()  # load into memory first
    fusi = cf.registration.register_volumewise(fusi)
    ```
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
        volume: xr.DataArray,
    ) -> tuple[xr.DataArray, npt.NDArray[np.floating], RegistrationDiagnostics]:
        # Once aborted, skip cheaply: building SimpleITK images and resampling is
        # pure-Python/GIL-bound work that, multiplied across many workers, starves the
        # GUI thread. Return the original frame with a zero-iteration "aborted"
        # diagnostic instead. Only observes a live abort_event update when this task
        # runs on a thread-based worker -- see register_volumewise's Notes.
        if abort_event is not None and abort_event.is_set():
            return volume, aborted_affine.copy(), aborted_diagnostics

        return register_volume(
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
            # Restrict SimpleITK to 1 thread per frame: parallelism comes from running
            # many frames concurrently across Dask workers, not from SimpleITK itself.
            sitk_threads=1,
            show_progress=False,
            abort_event=abort_event,
        )

    # Aborted/un-started frames are left blank (filled with the data minimum,
    # i.e. background) rather than copying the unregistered input, so the partial
    # result visibly shows which frames were skipped.
    output = np.full(data_moved.shape, float(ref_da.min()), dtype=data_moved.dtype)
    affines: list[npt.NDArray[np.floating]] = [
        aborted_affine.copy() for _ in range(n_frames)
    ]
    final_metric_values = [float("nan")] * n_frames
    n_iterations_per_frame = [0] * n_frames
    statuses = ["aborted"] * n_frames
    diagnostics: list[RegistrationDiagnostics] = [aborted_diagnostics] * n_frames

    if is_h5py_backed(data):
        raise TypeError(
            "Data is backed by an h5py dataset, which cannot be pickled to send to "
            "a Dask distributed.Client worker. Call .compute() to materialize the "
            "data into memory before calling register_volumewise."
        )

    with _volumewise_client() as submit:
        futures = {}
        for t in range(n_frames):
            if abort_event is not None and abort_event.is_set():
                continue
            futures[submit(_register_one, data_moved.isel(time=t))] = t

        progress_ctx: Progress | nullcontext[None] = (
            Progress() if show_progress else nullcontext()
        )
        try:
            with progress_ctx as progress:
                task_id = None
                if progress is not None:
                    task_id = progress.add_task(
                        "Registering volumes...", total=n_frames
                    )
                    if skipped_at_start := n_frames - len(futures):
                        progress.update(task_id, advance=skipped_at_start)

                for future in as_completed(futures):
                    t = futures[future]
                    registered_da, frame_affine, frame_diag = future.result()
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
                    if progress is not None and task_id is not None:
                        progress.update(task_id, advance=1)
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
        # trace, which adds up over long recordings — gated behind the flag.
        result.attrs["registration_diagnostics"] = list(diagnostics)

    return result.transpose(*data.dims)
