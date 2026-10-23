"""Volume-to-volume registration for fUSI data."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.itk import (
    RegistrationMetric,
    RegistrationOptimizer,
    resolve_itk_threads,
    voxeldata_to_itk_image,
)
from confusius.registration._utils import (
    abort_on_sigint,
    expand_thin_dims,
    replace_affines_attr,
    set_itk_thread_count,
    validate_fill_value,
    validate_intensity_scaling,
)
from confusius.registration.affines import (
    affine_to_itk_linear_transform,
    itk_linear_transform_to_affine,
)
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.validation import validate_matching_spatial_units

if TYPE_CHECKING:
    from threading import Event

    from confusius.registration.progress import RegistrationProgress


def _validate_register_volume_inputs(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    fixed_mask: xr.DataArray | None,
    moving_mask: xr.DataArray | None,
    transform_type: Literal["translation", "rigid", "affine", "bspline", "syn"],
    metric: RegistrationMetric,
    fixed_intensity_scaling: Literal["none", "db", "sqrt"] | float,
    moving_intensity_scaling: Literal["none", "db", "sqrt"] | float,
    number_of_histogram_bins: int,
    metric_sampling_percentage: float | None,
    metric_sampling_seed: int | None,
    learning_rate: float | Literal["auto"],
    number_of_iterations: int,
    convergence_window_size: int,
    initialization: Literal["center_geometry", "center_moments"]
    | npt.NDArray[np.floating]
    | None,
    shrink_factors: Sequence[int],
    smoothing_sigmas: Sequence[int],
    resample_interpolation: Literal["linear", "bspline"],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None, xr.DataArray | None]:
    """Validate all inputs to `register_volume` before any computation.

    Parameters
    ----------
    moving : xarray.DataArray
        Spatial-only VoxelData array to register.
    fixed : xarray.DataArray
        Spatial-only VoxelData array used as the reference volume.
    fixed_mask : xarray.DataArray or None
        Mask for fixed image. If provided, must have same dimensions as fixed.
    moving_mask : xarray.DataArray or None
        Mask for moving image. If provided, must have same dimensions as moving.
    transform_type : {"translation", "rigid", "affine", "bspline", "syn"}
        Transform model name.
    metric : {"correlation", "mattes_mi", "ants_cc", "mean_squares", "joint_mi"}
        Similarity metric name.
    fixed_intensity_scaling : {"none", "db", "sqrt"} or float
        Intensity transform applied to `fixed` before optimization.
    moving_intensity_scaling : {"none", "db", "sqrt"} or float
        Intensity transform applied to `moving` before optimization.
    number_of_histogram_bins : int
        Number of histogram bins for Mattes or joint-histogram mutual information.
    metric_sampling_percentage : float, optional
        Random metric sampling percentage in `(0, 1]`. If not provided, all voxels
        are used without random sampling.
    metric_sampling_seed : int, optional
        Random metric sampling seed.
    learning_rate : float or "auto"
        Optimizer step size or `"auto"`.
    number_of_iterations : int
        Maximum number of optimizer iterations.
    convergence_window_size : int
        Window size for convergence checking.
    initialization : {"center_geometry", "center_moments"} or (N+1, N+1) numpy.ndarray
        Transform initialization mode or precomputed affine transform.
    shrink_factors : sequence of int
        Downsampling factors per pyramid level.
    smoothing_sigmas : sequence of int
        Smoothing sigmas per pyramid level.
    resample_interpolation : {"linear", "bspline"}
        Interpolator name used when resampling.

    Returns
    -------
    moving : xarray.DataArray
        Canonicalized moving volume.
    fixed : xarray.DataArray
        Canonicalized fixed volume.
    fixed_mask : xarray.DataArray or None
        `fixed_mask` coerced to boolean dtype, or None if not provided.
    moving_mask : xarray.DataArray or None
        `moving_mask` coerced to boolean dtype, or None if not provided.

    Raises
    ------
    ValueError
        For any invalid combination of inputs.
    """
    # --- DataArray dims and ndim ---
    if "time" in fixed.dims or "time" in moving.dims:
        raise ValueError(
            "register_volume expects spatial-only VoxelData arrays. "
            "For volume-wise registration, use register_volumewise."
        )

    from confusius.validation import ensure_voxeldata

    moving = ensure_voxeldata(
        moving, require_time=False, allow_pose=False, allow_extra_dims=False
    )
    fixed = ensure_voxeldata(
        fixed, require_time=False, allow_pose=False, allow_extra_dims=False
    )
    validate_matching_spatial_units((("moving", moving), ("fixed", fixed)))

    # --- NaN checks ---
    if np.any(np.isnan(moving.values)):
        raise ValueError(
            "'moving' contains NaN values. ITK treats NaN as a regular float, "
            "which corrupts the similarity metric. Replace NaN values before "
            "registering (e.g. fill with zero or a background value)."
        )
    if np.any(np.isnan(fixed.values)):
        raise ValueError(
            "'fixed' contains NaN values. ITK treats NaN as a regular float, "
            "which corrupts the similarity metric. Replace NaN values before "
            "registering (e.g. fill with zero or a background value)."
        )

    # --- Literal-valued parameters ---
    valid_transform_types = {"translation", "rigid", "affine", "bspline", "syn"}
    if transform_type not in valid_transform_types:
        raise ValueError(
            f"Invalid transform {transform_type!r}. "
            f"Expected one of {sorted(valid_transform_types)}."
        )

    valid_metrics = {"correlation", "mattes_mi", "ants_cc", "mean_squares", "joint_mi"}
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric {metric!r}. Expected one of {sorted(valid_metrics)}."
        )

    validate_intensity_scaling(fixed_intensity_scaling, "fixed_intensity_scaling")
    validate_intensity_scaling(moving_intensity_scaling, "moving_intensity_scaling")

    valid_initializations = {"center_geometry", "center_moments"}
    if (
        initialization is not None
        and not isinstance(initialization, np.ndarray)
        and not (
            isinstance(initialization, str) and initialization in valid_initializations
        )
    ):
        raise ValueError(
            f"Invalid initialization {initialization!r}. "
            f"Expected one of {sorted(valid_initializations)}, None, or a "
            "homogeneous affine matrix."
        )

    valid_interpolations = {"linear", "bspline"}
    if resample_interpolation not in valid_interpolations:
        raise ValueError(
            f"Invalid resample_interpolation {resample_interpolation!r}. "
            f"Expected one of {sorted(valid_interpolations)}."
        )

    # --- Numeric parameters ---
    if learning_rate != "auto" and (
        not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            f"learning_rate must be a positive finite float or 'auto'; "
            f"got {learning_rate!r}."
        )

    if not isinstance(number_of_iterations, int) or number_of_iterations < 1:
        raise ValueError(
            f"number_of_iterations must be a positive integer; "
            f"got {number_of_iterations!r}."
        )

    if not isinstance(convergence_window_size, int) or convergence_window_size < 1:
        raise ValueError(
            f"convergence_window_size must be a positive integer; "
            f"got {convergence_window_size!r}."
        )

    if not isinstance(number_of_histogram_bins, int) or number_of_histogram_bins < 1:
        raise ValueError(
            f"number_of_histogram_bins must be a positive integer; "
            f"got {number_of_histogram_bins!r}."
        )

    if metric_sampling_percentage is not None and (
        not isinstance(metric_sampling_percentage, (int, float))
        or not np.isfinite(metric_sampling_percentage)
        or not 0 < metric_sampling_percentage <= 1
    ):
        raise ValueError(
            "metric_sampling_percentage must be a positive finite float no larger "
            f"than 1; got {metric_sampling_percentage!r}."
        )

    if metric_sampling_seed is not None and (
        metric_sampling_percentage is None or metric_sampling_percentage >= 1
    ):
        raise ValueError(
            "metric_sampling_seed requires metric_sampling_percentage below 1."
        )

    if metric_sampling_seed is not None and (
        not isinstance(metric_sampling_seed, int) or metric_sampling_seed < 0
    ):
        raise ValueError(
            f"metric_sampling_seed must be a non-negative integer; got "
            f"{metric_sampling_seed!r}."
        )

    # --- Mask validation ---
    from confusius.validation import ensure_mask

    if fixed_mask is not None:
        fixed_mask = ensure_mask(fixed_mask, fixed, "fixed_mask")
    if moving_mask is not None:
        moving_mask = ensure_mask(moving_mask, moving, "moving_mask")

    # --- Multi-resolution consistency ---
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError(
            f"shrink_factors and smoothing_sigmas must have the same length; "
            f"got {len(shrink_factors)} and {len(smoothing_sigmas)}."
        )

    return moving, fixed, fixed_mask, moving_mask


def _apply_intensity_scaling(
    data: xr.DataArray, intensity_scaling: Literal["none", "db", "sqrt"] | float
) -> xr.DataArray:
    """Apply optional intensity scaling for optimizer inputs.

    Parameters
    ----------
    data : xarray.DataArray
        Input VoxelData array.
    intensity_scaling : {"none", "db", "sqrt"} or float
        Scaling mode to apply. Assumed already validated by
        `validate_intensity_scaling`. Floats apply power scaling with that
        exponent.

    Returns
    -------
    xarray.DataArray
        Scaled data for optimization, or `data` unchanged.
    """
    if intensity_scaling == "none":
        return data
    from confusius.xarray.scale import db_scale, power_scale

    if intensity_scaling == "db":
        if bool((data < 0).any()):
            raise ValueError(
                "Decibel registration scaling requires nonnegative values."
            )
        # Zero power must remain finite in optimizer images, not become -inf.
        floor = np.finfo(np.result_type(data.dtype, np.float32)).tiny
        return db_scale(data.clip(min=floor))
    if intensity_scaling == "sqrt":
        return power_scale(data, exponent=0.5)
    return power_scale(data, exponent=float(intensity_scaling))


def _translate_registration_runtime_error(
    exc: RuntimeError,
    *,
    transform_type: Literal["translation", "rigid", "affine", "bspline", "syn"],
    learning_rate: float | Literal["auto"],
) -> RuntimeError:
    """Return a clearer registration error for known ITK failures.

    Parameters
    ----------
    exc : RuntimeError
        Exception raised by ITK during optimizer execution.
    transform_type : {"translation", "rigid", "affine", "bspline", "syn"}
        Registration model used for the failed run.
    learning_rate : float or "auto"
        User-requested learning rate mode.

    Returns
    -------
    RuntimeError
        Translated exception when the failure mode is recognized, otherwise `exc`.
    """
    message = str(exc)
    if "m_Scales values must be > epsilon" not in message:
        return exc

    parts = [
        "ITK could not compute valid optimizer scales for this registration.",
        "Some transform parameters have near-zero world effect, so the gradient-descent optimizer cannot choose a stable step size.",
    ]
    if transform_type == "bspline":
        parts.append(
            "This is most common for `transform_type='bspline'`, especially when the control-point grid is too fine for the image extent or overlap."
        )
    if learning_rate == "auto":
        parts.append(
            'Retry with a fixed `learning_rate` such as `0.1` or `0.01` instead of `"auto"`.'
        )
    else:
        parts.append(
            "Changing `learning_rate` alone may not help because this failure happens before optimisation starts."
        )
    if transform_type == "bspline":
        parts.append(
            "If that still fails, use a coarser `mesh_size` or run affine/rigid registration first and pass the result as `initialization`."
        )

    return RuntimeError(" ".join(parts))


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = ...,
    moving_mask: xr.DataArray | None = ...,
    transform_type: Literal["translation", "rigid", "affine"],
    optimizer: RegistrationOptimizer = ...,
    optimizer_seed: int = ...,
    correlation_radius: int = ...,
    update_field_variance: float = ...,
    total_field_variance: float = ...,
    metric: RegistrationMetric = ...,
    fixed_intensity_scaling: Literal["none", "db", "sqrt"] | float = ...,
    moving_intensity_scaling: Literal["none", "db", "sqrt"] | float = ...,
    number_of_histogram_bins: int = ...,
    metric_sampling_percentage: float | None = ...,
    metric_sampling_seed: int | None = ...,
    learning_rate: float | Literal["auto"] = ...,
    number_of_iterations: int = ...,
    convergence_minimum_value: float = ...,
    convergence_window_size: int = ...,
    initialization: Literal["center_geometry", "center_moments"]
    | npt.NDArray[np.floating]
    | None = ...,
    optimizer_weights: list[float] | None = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    fill_value: float | None = ...,
    itk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
    max_composite_slices: int | None = ...,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating], RegistrationDiagnostics]":
    """Overload for linear transforms (translation/rigid/affine)."""


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = ...,
    moving_mask: xr.DataArray | None = ...,
    transform_type: Literal["bspline", "syn"],
    optimizer: RegistrationOptimizer = ...,
    optimizer_seed: int = ...,
    correlation_radius: int = ...,
    update_field_variance: float = ...,
    total_field_variance: float = ...,
    metric: RegistrationMetric = ...,
    fixed_intensity_scaling: Literal["none", "db", "sqrt"] | float = ...,
    moving_intensity_scaling: Literal["none", "db", "sqrt"] | float = ...,
    number_of_histogram_bins: int = ...,
    metric_sampling_percentage: float | None = ...,
    metric_sampling_seed: int | None = ...,
    learning_rate: float | Literal["auto"] = ...,
    number_of_iterations: int = ...,
    convergence_minimum_value: float = ...,
    convergence_window_size: int = ...,
    initialization: Literal["center_geometry", "center_moments"]
    | npt.NDArray[np.floating]
    | None = ...,
    optimizer_weights: list[float] | None = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    fill_value: float | None = ...,
    itk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
    max_composite_slices: int | None = ...,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = ...,
) -> "tuple[xr.DataArray, xr.DataArray, RegistrationDiagnostics]":
    """Overload for nonlinear transforms (returns DataArray transform)."""


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = ...,
    moving_mask: xr.DataArray | None = ...,
    optimizer: RegistrationOptimizer = ...,
    optimizer_seed: int = ...,
    correlation_radius: int = ...,
    update_field_variance: float = ...,
    total_field_variance: float = ...,
    metric: RegistrationMetric = ...,
    fixed_intensity_scaling: Literal["none", "db", "sqrt"] | float = ...,
    moving_intensity_scaling: Literal["none", "db", "sqrt"] | float = ...,
    number_of_histogram_bins: int = ...,
    metric_sampling_percentage: float | None = ...,
    metric_sampling_seed: int | None = ...,
    learning_rate: float | Literal["auto"] = ...,
    number_of_iterations: int = ...,
    convergence_minimum_value: float = ...,
    convergence_window_size: int = ...,
    initialization: Literal["center_geometry", "center_moments"]
    | npt.NDArray[np.floating]
    | None = ...,
    optimizer_weights: list[float] | None = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    fill_value: float | None = ...,
    itk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
    max_composite_slices: int | None = ...,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating], RegistrationDiagnostics]":
    """Overload for default transform (rigid, returns affine)."""


def register_volume(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = None,
    moving_mask: xr.DataArray | None = None,
    transform_type: Literal[
        "translation", "rigid", "affine", "bspline", "syn"
    ] = "rigid",
    correlation_radius: int = 3,
    update_field_variance: float = 3.0,
    total_field_variance: float = 1.0,
    metric: RegistrationMetric = "correlation",
    optimizer: RegistrationOptimizer = "gradient_descent",
    optimizer_seed: int = 0,
    fixed_intensity_scaling: Literal["none", "db", "sqrt"] | float = "none",
    moving_intensity_scaling: Literal["none", "db", "sqrt"] | float = "none",
    number_of_histogram_bins: int = 50,
    metric_sampling_percentage: float | None = None,
    metric_sampling_seed: int | None = None,
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
    resample: bool = True,
    resample_interpolation: Literal["linear", "bspline"] = "linear",
    fill_value: float | None = None,
    itk_threads: int = -1,
    show_progress: bool = False,
    plot_metric: bool = True,
    plot_composite: bool = True,
    max_composite_slices: int | None = 9,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = None,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating] | xr.DataArray, RegistrationDiagnostics]":
    """Register a single 3D volume to a fixed reference.

    Voxel spacing and origin are automatically extracted from the DataArray coordinates.
    Both inputs must be spatial-only (no `time` dimension). Single-slice recordings are
    supported as 3D volumes with a singleton `k` axis.

    Parameters
    ----------
    moving : xarray.DataArray
        Volume to register to `fixed`. Must be a 3D volume with dimensions `k`, `j`, `i`
        (single-slice recordings use a singleton `k` axis).
    fixed : xarray.DataArray
        Reference volume. Must be a 3D volume with dimensions `k`, `j`, `i`. Need not
        have the same shape as `moving`.
        When spatial coordinate `units` metadata is present on both `moving` and
        `fixed`, it must match.
    fixed_mask : xarray.DataArray, optional
        Mask for the fixed image. Must have boolean dtype and match the shape
        and coordinates of `fixed`. When provided, only voxels where the mask
        is True are used for computing the similarity metric. This is useful
        when the fixed image contains NaN values or regions that should be
        excluded from registration.
    moving_mask : xarray.DataArray, optional
        Mask for the moving image. Must have boolean dtype and match the shape
        and coordinates of `moving`. When provided, only voxels where the mask
        is True are used for computing the similarity metric. This is useful
        when the moving image contains NaN values or regions that should be
        excluded from registration.
    transform_type : {"translation", "rigid", "affine", "bspline", "syn"}, default: "rigid"
        Transform model to use during registration. `"translation"` allows
        only shifts. `"rigid"` adds rotation. `"affine"` adds scaling and
        shearing. `"bspline"` fits a non-linear deformable transform (see
        `mesh_size`). `"syn"` uses symmetric diffeomorphic normalization and
        returns a dense displacement field on the fixed grid.
    correlation_radius : int, default: 3
        Neighborhood radius in voxels along each native axis for `metric="ants_cc"`.
        Must be a positive integer.
    update_field_variance : float, default: 3.0
        Gaussian smoothing variance of SyN update fields, in squared voxel units.
        Must be finite and nonnegative. Only used for `transform_type="syn"`.
    total_field_variance : float, default: 1.0
        Gaussian smoothing variance of accumulated SyN fields, in squared voxel units.
        Must be finite and nonnegative. Only used for `transform_type="syn"`.
    metric : {"correlation", "mattes_mi", "ants_cc", "mean_squares", "joint_mi"}, default: "correlation"
        Similarity metric. `"correlation"` (normalized cross-correlation) is
        appropriate for same-modality registration. `"mattes_mi"` (Mattes
        mutual information) is better suited for multi-modal registration or
        when the intensity relationship between images is non-linear.
        `"ants_cc"` uses ANTs neighborhood correlation, recommended for same-modality
        SyN. `"mean_squares"` minimizes squared intensity differences for similarly
        scaled images. `"joint_mi"` uses joint-histogram mutual information.
    optimizer : {"gradient_descent", "lbfgs", "one_plus_one"}, default: "gradient_descent"
        Solver for parametric transforms. L-BFGS uses a limited-memory quasi-Newton
        method and supports linear and B-spline transforms. 1+1 evolutionary search
        supports linear transforms only; its covariance storage is quadratic in
        parameter count. SyN retains its own solver and rejects non-default choices.
        Only gradient descent supports `optimizer_weights`.
    optimizer_seed : int, default: 0
        Reproducible 1+1 evolutionary random seed in `[0, 2147483647]`, independent
        of metric sampling. Only used by `optimizer="one_plus_one"`.
    fixed_intensity_scaling : {"none", "db", "sqrt"} or float, default: "none"
        Intensity transform applied to `fixed` only for the registration optimizer.
        Floats apply power scaling with that exponent; `"sqrt"` is an alias for `0.5`.
    moving_intensity_scaling : {"none", "db", "sqrt"} or float, default: "none"
        Intensity transform applied to `moving` only for the registration optimizer,
        with the same options as `fixed_intensity_scaling`. Returned/resampled data
        keeps the original input intensities.
    number_of_histogram_bins : int, default: 50
        Number of histogram bins for `"mattes_mi"` and `"joint_mi"`.
    metric_sampling_percentage : float, optional
        Percentage of voxels randomly sampled when computing the metric, in `(0, 1]`.
        If not provided, all voxels are used without random sampling.
    metric_sampling_seed : int, optional
        Seed for random metric sampling. Only used when `metric_sampling_percentage < 1`.
        If not provided, ITK's default wall-clock seed is used.
    learning_rate : float or "auto", default: "auto"
        Optimizer step size in normalized units. `"auto"` re-estimates the rate at
        every iteration. A float uses that value directly; if registration diverges or
        fails to converge, reduce it. For SyN, this is the maximum update length in
        voxel units; `"auto"` selects a fixed rate of `0.1`, not automatic estimation.
        For L-BFGS, a float is the initial line-search step length; for 1+1 it is
        the initial search radius in scaled parameter space. Both use `1.0` for `"auto"`.
    number_of_iterations : int, default: 100
        Maximum number of optimizer iterations.
    convergence_minimum_value : float, default: 1e-6
        Value used for convergence checking in conjunction with the energy profile of
        the similarity metric that is estimated in the given window size. For
        L-BFGS, the gradient convergence tolerance; for 1+1, the minimum search
        covariance Frobenius norm. The convergence window applies only to gradient
        descent and SyN.
    convergence_window_size : int, default: 10
        Number of values of the similarity metric which are used to estimate the energy
        profile of the similarity metric.
    initialization : {"center_geometry", "center_moments"} or (N+1, N+1) numpy.ndarray, default: "center_geometry"
        Initial transform mapping `fixed` to `moving` coordinates, applied before
        optimization:

        - `"center_geometry"`: aligns image centers.
        - `"center_moments"`: aligns centers of mass.
        - `(N+1, N+1)` homogeneous affine matrix: uses a precomputed affine
          transform.
        - `None`: uses the identity transform.

        For `transform_type="bspline"`, centering modes are ignored but affine
        initialization is supported.
    optimizer_weights : list of float, optional
        Per-parameter weights applied on top of the auto-estimated world shift
        scales. `None` uses identity weights (all ones). A list is passed directly to
        ITK's `SetWeights`; its length must match the number of
        transform parameters (6 for rigid, 12 for affine). The weight for each
        parameter is multiplied into the effective step
        size: `0` freezes a parameter entirely, values in `(0, 1)` slow it down, and
        `1` leaves it unchanged. For the 3D Euler transform the parameter order is
        `[angleX, angleY, angleZ, tx, ty, tz]`; to disable rotations around x and y
        set weights to `[0, 0, 1, 1, 1, 1]`. Not supported for SyN.
    mesh_size : tuple of int, default: (10, 10, 10)
        Number of B-spline mesh nodes along each spatial dimension. Only used when
        `transform_type="bspline"`.
    use_multi_resolution : bool, default: False
        Whether to use a multi-resolution pyramid during registration. When `True`,
        registration proceeds from a coarse downsampled version of the images to the
        full resolution, which improves convergence for large displacements and reduces
        the risk of local minima. SyN currently requires `False` because ITK wheels
        do not wrap the displacement-field parameter adaptors needed for its pyramid.
    shrink_factors : sequence of int, default: (6, 2, 1)
        Downsampling factor at each pyramid level, from coarsest to finest. Must have
        the same length as `smoothing_sigmas`. Only used when
        `use_multi_resolution=True`.
    smoothing_sigmas : sequence of int, default: (6, 2, 1)
        Gaussian smoothing sigma (in voxels) applied at each pyramid level, from
        coarsest to finest. Must have the same length as `shrink_factors`. Only used
        when `use_multi_resolution=True`.
    resample : bool, default: True
        Whether to resample the moving volume onto the fixed grid after estimating the
        transform. When `True`, the output is resampled onto the fixed grid and its
        coordinates match `fixed`. When `False`, only the transform is computed and the
        moving volume is returned unchanged with its original coordinates.
    resample_interpolation : {"linear", "bspline"}, default: "linear"
        Interpolator used when resampling the moving volume onto the fixed grid.
        `"linear"` is fast and appropriate for most cases. `"bspline"` (3rd-order
        B-spline) produces smoother results and reduces ringing, useful for atlas
        registration. Only used when `resample=True`.
    fill_value : float, optional
        Fill value for voxels that fall outside the moving image's field of view after
        resampling. Applied to both the final registered output (when `resample=True`)
        and the progress composite overlay (when `show_progress=True` and
        `plot_composite=True`). If not provided, defaults to the minimum value of
        `moving`, which renders out-of-FOV regions as background regardless of intensity
        scale (important for dB data where 0 is maximum intensity). Must be representable
        in `moving`'s dtype: to fill an integer volume with NaN, cast it to float first.
    itk_threads : int, default: -1
        Number of threads ITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + itk_threads)`, so `-1` means all CPUs, `-2`
        means all minus one, and so on. You may want to set this to a lower value or
        `1` when running multiple registrations in parallel (e.g. with joblib) to
        avoid over-subscribing the CPU.
    show_progress : bool, default: False
        Whether to display a live progress plot during registration. The plot is shown
        in a Jupyter notebook or in an interactive matplotlib window depending on the
        active backend.
    plot_metric : bool, default: True
        Whether to include the optimizer metric curve in the progress plot. Ignored when
        `show_progress=False`.
    plot_composite : bool, default: True
        Whether to include a fixed/moving composite overlay in the progress plot.
        Requires resampling the moving image at every iteration. Ignored when
        `show_progress=False`.
    max_composite_slices : int or None, default: 9
        Maximum number of slices to draw in the composite overlay for 3D volumes,
        evenly spaced along the volume's `k` axis. Keeps the mosaic grid readable and
        each iteration's render cheap for volumes with many slices. Use `None` to
        plot every slice. Ignored when `show_progress=False`, `plot_composite=False`,
        or the volume is 2D.
    progress_plotter : callable, optional
        Factory that builds the progress reporter, called inside `register_volume` as
        `progress_plotter(registration_method, fixed_img, moving_img, *, plot_metric,
        plot_composite, resample_kwargs, max_composite_slices)`. Here `resample_kwargs`
        carries `interpolation`, `fill_value`, and `itk_threads`. The returned object
        must implement the
        [`RegistrationProgress`][confusius.registration.RegistrationProgress] protocol
        (`update()` / `close()`). If not provided, the default
        [`MatplotlibRegistrationProgressPlotter`][confusius.registration.MatplotlibRegistrationProgressPlotter]
        is used. Ignored when `show_progress=False`. Custom factories are expected to
        be safe to call from a non-GUI thread; GUI side effects must be marshalled via
        thread-safe primitives such as Qt signals.
    abort_event : threading.Event, optional
        Cooperative cancellation flag. If set before or during optimisation, the
        registration stops at the next ITK iteration boundary and returns
        the current intermediate result with `diagnostics.status="aborted"`.

    Returns
    -------
    registered : xarray.DataArray
        When `resample=True`, the moving volume resampled onto the fixed grid with
        coordinates matching `fixed` and world-space affines inherited from `fixed`.
        When `resample=False`, the original moving volume with its original coordinates
        and attributes.
    transform : (4, 4) numpy.ndarray or xarray.DataArray or None
        Estimated registration transform. For linear transforms (`"translation"`,
        `"rigid"`, `"affine"`), returns a homogeneous `(4, 4)` affine matrix in world
        space. Follows ITK's pull/inverse convention: the matrix maps fixed-space
        coordinates to moving-space coordinates. For `transform_type="bspline"`,
        returns an `xarray.DataArray` containing the B-spline control-point grid, not a
        dense deformation field. The first dimension is `component` with length 3,
        followed by spatial dimensions in ConfUSIus order (`("k", "j", "i")`). The
        coordinate values along each spatial axis are
        the world positions of the control points. Attributes include `type =
        "bspline_transform"`, the spline `order`, and the control-grid `direction`
        matrix. When an affine `initialization` was also supplied, the DataArray also
        includes `attrs["affines"]["bspline_initialization"]` so that the full
        composite transform (pre-affine + B-spline) can be reconstructed for later
        resampling. For `transform_type="syn"`, returns a VoxelData displacement
        field with dimensions `(component, k, j, i)` and
        `attrs['type'] = 'displacement_field_transform'`. Components are world
        `z/y/x` displacements in the input spatial units, labeled `k/j/i` as for
        existing displacement fields. The fixed-grid geometry is preserved and
        affine initialization is already included in the field. Reuse it with
        [resample_like][confusius.registration.resample_like].
    diagnostics : confusius.registration.RegistrationDiagnostics
        Per-iteration metric values, final metric value, iteration count, and the
        optimizer stop condition. Useful for plotting convergence curves, comparing
        runs, and detecting registrations that did not converge.

    Raises
    ------
    ValueError
        If either input contains a `time` dimension or does not contain the spatial
        dimensions `k`, `j`, and `i`.
    ValueError
        If `moving` or `fixed` contains NaN values.
    ValueError
        If `transform_type`, `metric`, `fixed_intensity_scaling`,
        `moving_intensity_scaling`, `initialization`, or `resample_interpolation`
        is not a recognised value, or the optimizer selection/seed is invalid or
        incompatible with the transform or optimizer weights.
    ValueError
        If `learning_rate` is not a positive finite float or `"auto"`.
    ValueError
        If `number_of_iterations`, `convergence_window_size`, or
        `number_of_histogram_bins` is not a positive integer.
    ValueError
        If `metric_sampling_percentage` is outside `(0, 1]`, or if
        `metric_sampling_seed` is negative or provided without random sampling.
    ValueError
        If `shrink_factors` and `smoothing_sigmas` have different lengths.
    ValueError
        If an affine `initialization` is provided and its shape does not match the
        image dimensionality.
    ValueError
        If the neighborhood radius or SyN variances are invalid, or if SyN is
        requested with multi-resolution registration or optimizer weights.
    TypeError
        If `fixed_mask` or `moving_mask` is not a boolean DataArray.
    ValueError
        If `fixed_mask` shape does not match `fixed` or `moving_mask` shape does
        not match `moving`.
    """
    import itk

    moving, fixed, fixed_mask, moving_mask = _validate_register_volume_inputs(
        moving=moving,
        fixed=fixed,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        transform_type=transform_type,
        metric=metric,
        fixed_intensity_scaling=fixed_intensity_scaling,
        moving_intensity_scaling=moving_intensity_scaling,
        number_of_histogram_bins=number_of_histogram_bins,
        metric_sampling_percentage=metric_sampling_percentage,
        metric_sampling_seed=metric_sampling_seed,
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
        convergence_window_size=convergence_window_size,
        initialization=initialization,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        resample_interpolation=resample_interpolation,
    )

    if optimizer not in ("gradient_descent", "lbfgs", "one_plus_one"):
        raise ValueError(f"Invalid optimizer: {optimizer!r}.")
    if not isinstance(optimizer_seed, int) or not 0 <= optimizer_seed <= 2**31 - 1:
        raise ValueError("optimizer_seed must be an integer in [0, 2147483647].")
    if transform_type == "syn" and optimizer != "gradient_descent":
        raise ValueError(
            "SyN uses its own solver; optimizer selection is not supported."
        )
    if optimizer == "one_plus_one" and transform_type == "bspline":
        raise ValueError(
            "one_plus_one supports only linear transforms: its covariance matrix "
            "grows quadratically with the number of parameters."
        )
    if optimizer != "gradient_descent" and optimizer_weights is not None:
        raise ValueError("optimizer_weights are supported only by gradient_descent.")
    if metric == "ants_cc" and (
        not isinstance(correlation_radius, int) or correlation_radius < 1
    ):
        raise ValueError("correlation_radius must be a positive integer.")
    if transform_type == "syn":
        if use_multi_resolution:
            raise ValueError(
                "SyN currently supports single-resolution registration only: "
                "ITK Python wheels do not wrap displacement-field parameter adaptors."
            )
        if optimizer_weights is not None:
            raise ValueError(
                "optimizer_weights are not supported for SyN registration."
            )
        for name, variance in (
            ("update_field_variance", update_field_variance),
            ("total_field_variance", total_field_variance),
        ):
            if not np.isfinite(variance) or variance < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    fixed_itk = voxeldata_to_itk_image(fixed)
    moving_itk = voxeldata_to_itk_image(moving)
    fixed_reg = expand_thin_dims(
        voxeldata_to_itk_image(_apply_intensity_scaling(fixed, fixed_intensity_scaling))
    )
    moving_reg = expand_thin_dims(
        voxeldata_to_itk_image(
            _apply_intensity_scaling(moving, moving_intensity_scaling)
        )
    )
    image_type = type(fixed_reg)
    if type(moving_reg) is not image_type:
        moving_reg = itk.cast_image_filter(
            moving_reg, ttype=(type(moving_reg), image_type)
        )
    ndim = 3
    initialization_mode = initialization if isinstance(initialization, str) else None
    affine_initialization = (
        initialization if isinstance(initialization, np.ndarray) else None
    )
    if affine_initialization is not None and affine_initialization.shape != (4, 4):
        raise ValueError(
            f"initialization shape {affine_initialization.shape} does not match "
            "image dimensionality 3D (expected (4, 4))."
        )

    if transform_type == "bspline":
        initial = itk.BSplineTransform[itk.D, ndim, 3].New()
        initializer = itk.BSplineTransformInitializer[type(initial), image_type].New(
            Transform=initial, Image=fixed_reg, TransformDomainMeshSize=list(mesh_size)
        )
        initializer.InitializeTransform()
    else:
        transform_class = {
            "translation": itk.TranslationTransform[itk.D, ndim],
            "rigid": itk.Euler3DTransform[itk.D],
            "affine": itk.AffineTransform[itk.D, ndim],
            "syn": itk.AffineTransform[itk.D, ndim],
        }[transform_type]
        initial = transform_class.New()
        if (
            initialization_mode in ("center_geometry", "center_moments")
            and transform_type != "translation"
        ):
            # The wheel wraps 3D centering only for versor transforms; copy its pivot and offset.
            centered = itk.VersorRigid3DTransform[itk.D].New()
            initializer = itk.CenteredTransformInitializer[
                type(centered), image_type, image_type
            ].New(Transform=centered, FixedImage=fixed_reg, MovingImage=moving_reg)
            if initialization_mode == "center_geometry":
                initializer.GeometryOn()
            else:
                initializer.MomentsOn()
            initializer.InitializeTransform()
            initial.SetCenter(centered.GetCenter())
            initial.SetTranslation(centered.GetTranslation())
    if affine_initialization is not None:
        pre_transform = affine_to_itk_linear_transform(affine_initialization)
        if transform_type == "bspline":
            composite = itk.CompositeTransform[itk.D, ndim].New()
            composite.AddTransform(pre_transform)
            composite.AddTransform(initial)
            composite.SetOnlyMostRecentTransformToOptimizeOn()
            initial = composite
        else:
            initial = pre_transform

    # FixedImage/MovingImage do not drive template inference for metrics or registration.
    if metric == "correlation":
        metric_object = itk.CorrelationImageToImageMetricv4[
            image_type, image_type
        ].New()
    elif metric == "ants_cc":
        metric_object = itk.ANTSNeighborhoodCorrelationImageToImageMetricv4[
            image_type, image_type
        ].New(Radius=[correlation_radius] * ndim)
    elif metric == "mean_squares":
        metric_object = itk.MeanSquaresImageToImageMetricv4[
            image_type, image_type
        ].New()
    elif metric == "joint_mi":
        metric_object = itk.JointHistogramMutualInformationImageToImageMetricv4[
            image_type, image_type
        ].New(NumberOfHistogramBins=number_of_histogram_bins)
    else:
        metric_object = itk.MattesMutualInformationImageToImageMetricv4[
            image_type, image_type
        ].New(NumberOfHistogramBins=number_of_histogram_bins)
    metric_object.SetMaximumNumberOfWorkUnits(resolve_itk_threads(itk_threads))
    for mask, setter in (
        (fixed_mask, metric_object.SetFixedImageMask),
        (moving_mask, metric_object.SetMovingImageMask),
    ):
        if mask is not None:
            mask_image = expand_thin_dims(voxeldata_to_itk_image(mask))
            mask_image = itk.cast_image_filter(
                mask_image, ttype=(type(mask_image), itk.Image[itk.UC, ndim])
            )
            setter(itk.ImageMaskSpatialObject[ndim].New(Image=mask_image))

    if transform_type == "syn":
        registration = itk.SyNImageRegistrationMethod[image_type, image_type].New(
            FixedImage=fixed_reg,
            MovingImage=moving_reg,
            Metric=metric_object,
            MovingInitialTransform=initial,
            LearningRate=0.1 if learning_rate == "auto" else learning_rate,
            ConvergenceThreshold=convergence_minimum_value,
            ConvergenceWindowSize=convergence_window_size,
            GaussianSmoothingVarianceForTheUpdateField=update_field_variance,
            GaussianSmoothingVarianceForTheTotalField=total_field_variance,
        )
        # New() unpacks singleton lists, which is unsafe for ITK's array-valued setter.
        registration.SetNumberOfIterationsPerLevel([number_of_iterations])
        iteration_source = registration
    else:
        if optimizer == "lbfgs":
            optimizer_object = itk.LBFGSOptimizerv4.New(
                NumberOfIterations=number_of_iterations,
                GradientConvergenceTolerance=convergence_minimum_value,
                DefaultStepLength=1.0 if learning_rate == "auto" else learning_rate,
            )
        elif optimizer == "one_plus_one":
            generator = itk.NormalVariateGenerator.New()
            generator.Initialize(optimizer_seed)
            optimizer_object = itk.OnePlusOneEvolutionaryOptimizerv4.New(
                MaximumIteration=number_of_iterations,
                Epsilon=convergence_minimum_value,
                NormalVariateGenerator=generator,
            )
            optimizer_object.Initialize(
                1.0 if learning_rate == "auto" else learning_rate
            )
        else:
            optimizer_object = itk.GradientDescentOptimizerv4.New(
                LearningRate=1.0 if learning_rate == "auto" else learning_rate,
                NumberOfIterations=number_of_iterations,
                MinimumConvergenceValue=convergence_minimum_value,
                ConvergenceWindowSize=convergence_window_size,
                DoEstimateLearningRateAtEachIteration=learning_rate == "auto",
                DoEstimateLearningRateOnce=False,
            )
        scales = itk.RegistrationParameterScalesFromPhysicalShift[
            type(metric_object)
        ].New(Metric=metric_object, TransformForward=True)
        optimizer_object.SetScalesEstimator(scales)
        if optimizer_weights is not None:
            optimizer_object.SetWeights(
                itk.OptimizerParameters[itk.D](
                    np.asarray(optimizer_weights, dtype=np.float64)
                )
            )
        registration = itk.ImageRegistrationMethodv4[image_type, image_type].New(
            FixedImage=fixed_reg,
            MovingImage=moving_reg,
            Metric=metric_object,
            Optimizer=optimizer_object,
            InitialTransform=initial,
            InPlace=True,
        )
        iteration_source = optimizer_object
    registration.SetNumberOfWorkUnits(resolve_itk_threads(itk_threads))
    registration.SetNumberOfLevels(len(shrink_factors) if use_multi_resolution else 1)
    registration.SetShrinkFactorsPerLevel(
        list(shrink_factors) if use_multi_resolution else [1]
    )
    registration.SetSmoothingSigmasPerLevel(
        list(smoothing_sigmas) if use_multi_resolution else [0]
    )
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    if metric_sampling_percentage is not None and metric_sampling_percentage < 1:
        registration.SetMetricSamplingStrategy(
            itk.ImageRegistrationMethodv4Enums.MetricSamplingStrategy_RANDOM
        )
        registration.SetMetricSamplingPercentage(metric_sampling_percentage)
        if metric_sampling_seed is None:
            registration.MetricSamplingReinitializeSeed()
        else:
            registration.MetricSamplingReinitializeSeed(metric_sampling_seed)

    metric_values: list[float] = []
    needs_fill_value = resample or (show_progress and plot_composite)
    resolved_fill = (
        fill_value
        if fill_value is not None
        else (float(moving.min()) if needs_fill_value else None)
    )
    if resolved_fill is not None:
        validate_fill_value(resolved_fill, moving.dtype)

    with abort_on_sigint(abort_event) as effective_abort_event:
        cost_event = False

        def _ignore_cost_event() -> None:
            """Mark L-BFGS cost evaluations, which inherit the iteration event type."""
            nonlocal cost_event
            cost_event = True

        def _record_iteration() -> None:
            """Record the optimizer metric and honor cancellation at iteration boundaries."""
            nonlocal cost_event
            if cost_event:
                cost_event = False
                return
            # SyN also emits a level-initialization event before its first iteration.
            if transform_type == "syn" and registration.GetCurrentIteration() == 0:
                return
            metric_values.append(
                float(
                    registration.GetCurrentMetricValue()
                    if transform_type == "syn"
                    else iteration_source.GetValue()
                )
            )
            if (
                transform_type == "syn" or optimizer == "lbfgs"
            ) and plotter is not None:
                plotter.update()
            if effective_abort_event.is_set():
                if transform_type == "syn":
                    registration.SetNumberOfIterationsPerLevel(
                        [registration.GetCurrentIteration()]
                    )
                elif optimizer == "lbfgs":
                    # ITK's L-BFGS report_iter checks this limit after invoking observers.
                    # https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/Numerics/Optimizersv4/include/itkLBFGSOptimizerBasev4.hxx.
                    iteration_source.SetNumberOfIterations(0)
                else:
                    iteration_source.StopOptimization()
                # ITK's outer loop otherwise starts the optimizer again at the next pyramid level.
                registration.SetNumberOfLevels(registration.GetCurrentLevel() + 1)

        # Register filters first so cost events are marked before the parent observer.
        observer_tags = []
        if optimizer == "lbfgs":
            observer_tags.extend(
                iteration_source.AddObserver(event(), _ignore_cost_event)
                for event in (
                    itk.FunctionEvaluationIterationEvent,
                    itk.GradientEvaluationIterationEvent,
                    itk.FunctionAndGradientEvaluationIterationEvent,
                )
            )
        observer_tags.append(
            iteration_source.AddObserver(itk.IterationEvent(), _record_iteration)
        )
        plotter = None
        executed = False
        try:
            if show_progress:
                from confusius.registration.progress import (
                    MatplotlibRegistrationProgressPlotter,
                )

                factory = progress_plotter or MatplotlibRegistrationProgressPlotter
                plotter = factory(
                    registration,
                    fixed_itk,
                    moving_itk,
                    plot_metric=plot_metric,
                    plot_composite=plot_composite,
                    resample_kwargs={
                        "interpolation": resample_interpolation,
                        "fill_value": resolved_fill,
                        "itk_threads": itk_threads,
                    },
                    max_composite_slices=max_composite_slices,
                )
                if transform_type != "syn" and optimizer != "lbfgs":
                    observer_tags.append(
                        iteration_source.AddObserver(
                            itk.IterationEvent(), plotter.update
                        )
                    )

            if effective_abort_event.is_set():
                if transform_type != "bspline":
                    initial = (
                        affine_to_itk_linear_transform(affine_initialization)
                        if affine_initialization is not None
                        else itk.TranslationTransform[itk.D, ndim].New()
                    )
                aborted = True
                stop_condition = "Registration aborted before optimisation started."
            else:
                try:
                    with set_itk_thread_count(itk_threads):
                        registration.Update()
                except RuntimeError as exc:
                    raise _translate_registration_runtime_error(
                        exc, transform_type=transform_type, learning_rate=learning_rate
                    ) from exc
                executed = True
                aborted = effective_abort_event.is_set()
                if transform_type == "syn":
                    stop_condition = (
                        "SyN registration aborted."
                        if aborted
                        else "SyN convergence threshold reached."
                        if registration.GetCurrentConvergenceValue()
                        < convergence_minimum_value
                        else "SyN maximum number of iterations reached."
                    )
                else:
                    stop_condition = iteration_source.GetStopConditionDescription()
        finally:
            # C++ observers holding Python closures otherwise retain optimizers and images per frame.
            for tag in observer_tags:
                iteration_source.RemoveObserver(tag)
            if plotter is not None:
                plotter.close()

    if transform_type == "syn":
        from confusius._utils.geometry import VOXEL_DIMS, WORLD_DIMS
        from confusius.registration.resampling import resample_like
        from confusius.xarray import create_voxeldata

        pre_affine = itk_linear_transform_to_affine(initial)
        displacement = np.zeros((3, *fixed.shape), dtype=np.float64)
        if executed:
            native_field = registration.GetTransform().GetDisplacementField()
            field = create_voxeldata(
                itk.array_from_image(native_field).transpose(3, 2, 1, 0),
                dims=("component", *VOXEL_DIMS),
                extra_coords={"component": list(VOXEL_DIMS)},
                spacing=list(native_field.GetSpacing()),
                origin=list(native_field.GetOrigin()),
                direction=itk.array_from_matrix(native_field.GetDirection()),
                units=fixed.coords["z"].attrs["units"],
                attrs={"type": "displacement_field_transform"},
            )
            displacement = resample_like(
                field,
                fixed,
                np.eye(4),
                interpolation="linear",
                fill_value=0,
                itk_threads=itk_threads,
            ).values
            composite = itk.CompositeTransform[itk.D, ndim].New()
            composite.AddTransform(initial)
            composite.AddTransform(registration.GetTransform())
            initial = composite
        world = np.stack([fixed.coords[dim].values for dim in WORLD_DIMS])
        displacement = (
            np.einsum("ij,j...->i...", pre_affine[:3, :3], world + displacement)
            + pre_affine[:3, 3, None, None, None]
            - world
        )
        optimized_transform = xr.DataArray(
            displacement,
            coords=fixed.coords,
            dims=("component", *VOXEL_DIMS),
            attrs={"type": "displacement_field_transform"},
        ).assign_coords(component=list(VOXEL_DIMS))

    if resample:
        from confusius._utils.itk import cast_resampled_values, resample_itk_image

        assert resolved_fill is not None
        image = resample_itk_image(
            moving_itk,
            fixed_itk,
            initial,
            interpolation=resample_interpolation,
            fill_value=resolved_fill,
            itk_threads=itk_threads,
        )
        registered_arr = cast_resampled_values(
            itk.array_from_image(image).T, moving.dtype
        )
        reference = fixed
    else:
        registered_arr = moving.values
        reference = moving
    result = xr.DataArray(
        registered_arr,
        coords=reference.coords,
        dims=reference.dims,
        attrs=moving.attrs.copy(),
    )
    if resample:
        replace_affines_attr(result, fixed)
    if transform_type == "bspline":
        from confusius.registration.bspline import itk_bspline_to_dataarray

        optimized_transform = itk_bspline_to_dataarray(
            initial, pre_affine=affine_initialization
        )
    elif transform_type != "syn":
        optimized_transform = itk_linear_transform_to_affine(initial)
    final_metric = (
        float(metric_values[-1])
        if metric_values
        else float(
            registration.GetCurrentMetricValue()
            if transform_type == "syn"
            else iteration_source.GetValue()
        )
        if executed
        else float("nan")
    )
    diagnostics = RegistrationDiagnostics(
        metric=metric,
        metric_values=np.asarray(metric_values, dtype=float),
        final_metric_value=final_metric,
        n_iterations=len(metric_values),
        stop_condition=stop_condition,
        status="aborted" if aborted else "completed",
    )
    return result, optimized_transform, diagnostics
