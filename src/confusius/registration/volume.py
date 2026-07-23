"""Volume-to-volume registration for fUSI data."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration._utils import (
    abort_on_sigint,
    dataarray_to_sitk_image,
    expand_thin_dims,
    replace_affines_attr,
    set_sitk_thread_count,
)
from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    sitk_linear_transform_to_affine,
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
    transform_type: Literal["translation", "rigid", "affine", "bspline"],
    metric: Literal["correlation", "mattes_mi"],
    number_of_histogram_bins: int,
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
        Volume to register.
    fixed : xarray.DataArray
        Reference volume.
    fixed_mask : xarray.DataArray or None
        Mask for fixed image. If provided, must have same dimensions as fixed.
    moving_mask : xarray.DataArray or None
        Mask for moving image. If provided, must have same dimensions as moving.
    transform_type : {"translation", "rigid", "affine", "bspline"}
        Transform model name.
    metric : {"correlation", "mattes_mi"}
        Similarity metric name.
    number_of_histogram_bins : int
        Number of histogram bins for Mattes mutual information.
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
            "register_volume expects spatial-only DataArrays. "
            "For volume-wise registration, use register_volumewise."
        )

    from confusius.validation import ensure_fusi_dataarray

    moving = ensure_fusi_dataarray(
        moving,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )
    fixed = ensure_fusi_dataarray(
        fixed,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
    )
    validate_matching_spatial_units((("moving", moving), ("fixed", fixed)))

    # --- NaN checks ---
    if np.any(np.isnan(moving.values)):
        raise ValueError(
            "'moving' contains NaN values. SimpleITK treats NaN as a regular float, "
            "which corrupts the similarity metric. Replace NaN values before "
            "registering (e.g. fill with zero or a background value)."
        )
    if np.any(np.isnan(fixed.values)):
        raise ValueError(
            "'fixed' contains NaN values. SimpleITK treats NaN as a regular float, "
            "which corrupts the similarity metric. Replace NaN values before "
            "registering (e.g. fill with zero or a background value)."
        )

    # --- Literal-valued parameters ---
    valid_transform_types = {"translation", "rigid", "affine", "bspline"}
    if transform_type not in valid_transform_types:
        raise ValueError(
            f"Invalid transform {transform_type!r}. "
            f"Expected one of {sorted(valid_transform_types)}."
        )

    valid_metrics = {"correlation", "mattes_mi"}
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric {metric!r}. Expected one of {sorted(valid_metrics)}."
        )

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
    if learning_rate != "auto":
        if (
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

    # --- Mask validation ---
    from confusius.validation import validate_mask

    if fixed_mask is not None:
        fixed_mask = validate_mask(fixed_mask, fixed, "fixed_mask")
    if moving_mask is not None:
        moving_mask = validate_mask(moving_mask, moving, "moving_mask")

    # --- Multi-resolution consistency ---
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError(
            f"shrink_factors and smoothing_sigmas must have the same length; "
            f"got {len(shrink_factors)} and {len(smoothing_sigmas)}."
        )

    return moving, fixed, fixed_mask, moving_mask


def _translate_registration_runtime_error(
    exc: RuntimeError,
    *,
    transform_type: Literal["translation", "rigid", "affine", "bspline"],
    learning_rate: float | Literal["auto"],
) -> RuntimeError:
    """Return a clearer registration error for known SimpleITK failures.

    Parameters
    ----------
    exc : RuntimeError
        Exception raised by SimpleITK during optimizer execution.
    transform_type : {"translation", "rigid", "affine", "bspline"}
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
        "SimpleITK could not compute valid optimizer scales for this registration.",
        "Some transform parameters have near-zero physical effect, so the gradient-descent optimizer cannot choose a stable step size.",
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
    metric: Literal["correlation", "mattes_mi"] = ...,
    number_of_histogram_bins: int = ...,
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
    sitk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating], RegistrationDiagnostics]":
    """Overload for linear transforms (translation/rigid/affine)."""
    ...


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = ...,
    moving_mask: xr.DataArray | None = ...,
    transform_type: Literal["bspline"],
    metric: Literal["correlation", "mattes_mi"] = ...,
    number_of_histogram_bins: int = ...,
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
    sitk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = ...,
) -> "tuple[xr.DataArray, xr.DataArray, RegistrationDiagnostics]":
    """Overload for bspline transform (returns DataArray transform)."""
    ...


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = ...,
    moving_mask: xr.DataArray | None = ...,
    metric: Literal["correlation", "mattes_mi"] = ...,
    number_of_histogram_bins: int = ...,
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
    sitk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating], RegistrationDiagnostics]":
    """Overload for default transform (rigid, returns affine)."""
    ...


def register_volume(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    fixed_mask: xr.DataArray | None = None,
    moving_mask: xr.DataArray | None = None,
    transform_type: Literal["translation", "rigid", "affine", "bspline"] = "rigid",
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
    resample: bool = True,
    resample_interpolation: Literal["linear", "bspline"] = "linear",
    fill_value: float | None = None,
    sitk_threads: int = -1,
    show_progress: bool = False,
    plot_metric: bool = True,
    plot_composite: bool = True,
    progress_plotter: "Callable[..., RegistrationProgress] | None" = None,
    abort_event: "Event | None" = None,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating] | xr.DataArray, RegistrationDiagnostics]":  # noqa: E501
    """Register a single 3D volume to a fixed reference.

    Voxel spacing and origin are automatically extracted from the DataArray coordinates.
    Both inputs must be spatial-only (no `time` dimension). Single-slice recordings are
    supported as 3D volumes with a singleton `z` axis.

    Parameters
    ----------
    moving : xarray.DataArray
        Volume to register to `fixed`. Must be a 3D volume with dimensions `z`, `y`, `x`
        (single-slice recordings use a singleton `z` axis).
    fixed : xarray.DataArray
        Reference volume. Must be a 3D volume with dimensions `z`, `y`, `x`. Need not
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
    transform_type : {"translation", "rigid", "affine", "bspline"}, default: "rigid"
        Transform model to use during registration. `"translation"` allows
        only shifts. `"rigid"` adds rotation. `"affine"` adds scaling and
        shearing. `"bspline"` fits a non-linear deformable transform (see
        `mesh_size`).
    metric : {"correlation", "mattes_mi"}, default: "correlation"
        Similarity metric. `"correlation"` (normalized cross-correlation) is
        appropriate for same-modality registration. `"mattes_mi"` (Mattes
        mutual information) is better suited for multi-modal registration or
        when the intensity relationship between images is non-linear.
    number_of_histogram_bins : int, default: 50
        Number of histogram bins used by Mattes mutual information. Only
        relevant when using `"mattes_mi"` metric.
    learning_rate : float or "auto", default: "auto"
        Optimizer step size in normalized units. `"auto"` re-estimates the rate at
        every iteration. A float uses that value directly; if registration diverges or
        fails to converge, reduce it.
    number_of_iterations : int, default: 100
        Maximum number of optimizer iterations.
    convergence_minimum_value : float, default: 1e-6
        Value used for convergence checking in conjunction with the energy profile of
        the similarity metric that is estimated in the given window size.
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
        Per-parameter weights applied on top of the auto-estimated physical shift
        scales. `None` uses identity weights (all ones). A list is passed directly to
        SimpleITK's `SetOptimizerWeights`; its length must match the number of
        transform parameters (6 for rigid, 12 for affine). The weight for each
        parameter is multiplied into the effective step
        size: `0` freezes a parameter entirely, values in `(0, 1)` slow it down, and
        `1` leaves it unchanged. For the 3D Euler transform the parameter order is
        `[angleX, angleY, angleZ, tx, ty, tz]`; to disable rotations around x and y
        set weights to `[0, 0, 1, 1, 1, 1]`.
    mesh_size : tuple of int, default: (10, 10, 10)
        Number of B-spline mesh nodes along each spatial dimension. Only used when
        `transform_type="bspline"`.
    use_multi_resolution : bool, default: False
        Whether to use a multi-resolution pyramid during registration. When `True`,
        registration proceeds from a coarse downsampled version of the images to the
        full resolution, which improves convergence for large displacements and reduces
        the risk of local minima.
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
        scale (important for dB data where 0 is maximum intensity).
    sitk_threads : int, default: -1
        Number of threads SimpleITK may use internally. Negative values resolve to
        `max(1, os.cpu_count() + 1 + sitk_threads)`, so `-1` means all CPUs, `-2`
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
    progress_plotter : callable, optional
        Factory that builds the progress reporter, called inside `register_volume` as
        `progress_plotter(registration_method, fixed_img, moving_img, *, plot_metric,
        plot_composite, resample_kwargs)`. Here `resample_kwargs` carries
        `interpolation`, `fill_value`, and `sitk_threads`. The returned object must
        implement the
        [`RegistrationProgress`][confusius.registration.RegistrationProgress] protocol
        (`update()` / `close()`). If not provided, the default
        [`MatplotlibRegistrationProgressPlotter`][confusius.registration.MatplotlibRegistrationProgressPlotter]
        is used. Ignored when `show_progress=False`. Custom factories are expected to
        be safe to call from a non-GUI thread; GUI side effects must be marshalled via
        thread-safe primitives such as Qt signals.
    abort_event : threading.Event, optional
        Cooperative cancellation flag. If set before or during optimisation, the
        registration stops at the next SimpleITK iteration boundary and returns
        the current intermediate result with `diagnostics.status="aborted"`.

    Returns
    -------
    registered : xarray.DataArray
        When `resample=True`, the moving volume resampled onto the fixed grid with
        coordinates matching `fixed` and physical-space affines inherited from `fixed`.
        When `resample=False`, the original moving volume with its original coordinates
        and attributes.
    transform : (4, 4) numpy.ndarray or xarray.DataArray or None
        Estimated registration transform. For linear transforms (`"translation"`,
        `"rigid"`, `"affine"`), returns a homogeneous `(4, 4)` affine matrix in physical
        space. Follows SimpleITK's pull/inverse convention: the matrix maps fixed-space
        coordinates to moving-space coordinates. For `transform_type="bspline"`,
        returns an `xarray.DataArray` containing the B-spline control-point grid, not a
        dense deformation field. The first dimension is `component` with length 3,
        followed by spatial dimensions in ConfUSIus order (`("z", "y", "x")`). The
        coordinate values along each spatial axis are
        the physical positions of the control points. Attributes include `type =
        "bspline_transform"`, the spline `order`, and the control-grid `direction`
        matrix. When an affine `initialization` was also supplied, the DataArray also
        includes `attrs["affines"]["bspline_initialization"]` so that the full
        composite transform (pre-affine + B-spline) can be reconstructed for later
        resampling.
    diagnostics : confusius.registration.RegistrationDiagnostics
        Per-iteration metric values, final metric value, iteration count, and the
        optimizer stop condition. Useful for plotting convergence curves, comparing
        runs, and detecting registrations that did not converge.

    Raises
    ------
    ValueError
        If either input contains a `time` dimension or does not contain the spatial
        dimensions `z`, `y`, and `x`.
    ValueError
        If `moving` or `fixed` contains NaN values.
    ValueError
        If `transform_type`, `metric`, `initialization`, or
        `resample_interpolation` is not a recognised value.
    ValueError
        If `learning_rate` is not a positive finite float or `"auto"`.
    ValueError
        If `number_of_iterations`, `convergence_window_size`, or
        `number_of_histogram_bins` is not a positive integer.
    ValueError
        If `shrink_factors` and `smoothing_sigmas` have different lengths.
    ValueError
        If an affine `initialization` is provided and its shape does not match the
        image dimensionality.
    TypeError
        If `fixed_mask` or `moving_mask` is not a boolean DataArray.
    ValueError
        If `fixed_mask` shape does not match `fixed` or `moving_mask` shape does
        not match `moving`.
    """
    import SimpleITK as sitk

    moving, fixed, fixed_mask, moving_mask = _validate_register_volume_inputs(
        moving=moving,
        fixed=fixed,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        transform_type=transform_type,
        metric=metric,
        number_of_histogram_bins=number_of_histogram_bins,
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
        convergence_window_size=convergence_window_size,
        initialization=initialization,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        resample_interpolation=resample_interpolation,
    )

    fixed_sitk = dataarray_to_sitk_image(fixed)
    moving_sitk = dataarray_to_sitk_image(moving)

    # SimpleITK's multi-resolution pyramid and interpolation fail when any spatial
    # dimension is smaller than 4 voxels (common for single-slice fUSI recordings with
    # a 1-voxel depth). We thus expand thin dimensions before registration; the
    # originals are kept as the resample source/reference so the output grid is never
    # affected.
    fixed_reg = expand_thin_dims(fixed_sitk)
    moving_reg = expand_thin_dims(moving_sitk)

    # CenteredTransformInitializer (and the registration method) require both images to
    # have the same pixel type. Cast moving to fixed's type when they differ.
    if moving_reg.GetPixelID() != fixed_reg.GetPixelID():
        moving_reg = sitk.Cast(moving_reg, fixed_reg.GetPixelID())

    ndim = fixed_reg.GetDimension()

    # Validate affine initialization shape now that ndim is known.
    initialization_mode = initialization if isinstance(initialization, str) else None
    affine_initialization = (
        initialization if isinstance(initialization, np.ndarray) else None
    )
    if affine_initialization is not None:
        expected_shape = (ndim + 1, ndim + 1)
        if affine_initialization.shape != expected_shape:
            raise ValueError(
                f"initialization shape {affine_initialization.shape} does not match "
                f"image dimensionality {ndim}D (expected {expected_shape})."
            )

    requested_learning_rate = learning_rate

    registration = sitk.ImageRegistrationMethod()

    # --- Metric ---
    if metric == "correlation":
        registration.SetMetricAsCorrelation()
    else:
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=number_of_histogram_bins
        )

    registration.SetInterpolator(sitk.sitkLinear)

    # --- Masks ---
    if fixed_mask is not None:
        # Convert boolean mask to uint8 for SimpleITK
        fixed_mask_uint8 = fixed_mask.astype(np.uint8)
        fixed_mask_sitk = dataarray_to_sitk_image(fixed_mask_uint8)
        # Expand mask if image was expanded
        fixed_mask_sitk = expand_thin_dims(fixed_mask_sitk)
        registration.SetMetricFixedMask(fixed_mask_sitk)
    if moving_mask is not None:
        # Convert boolean mask to uint8 for SimpleITK
        moving_mask_uint8 = moving_mask.astype(np.uint8)
        moving_mask_sitk = dataarray_to_sitk_image(moving_mask_uint8)
        # Expand mask if image was expanded
        moving_mask_sitk = expand_thin_dims(moving_mask_sitk)
        registration.SetMetricMovingMask(moving_mask_sitk)

    # --- Optimizer ---
    estimate_learning_rate = registration.Never
    if learning_rate == "auto":
        learning_rate = 1.0
        estimate_learning_rate = registration.EachIteration

    registration.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=convergence_minimum_value,
        convergenceWindowSize=convergence_window_size,
        estimateLearningRate=estimate_learning_rate,
    )

    # Normalise parameter scales so that a unit step in each parameter produces the same
    # physical displacement. This is always applied regardless of learning_rate, so a
    # user-supplied float is interpreted in these normalised units. If registration
    # diverges, reduce learning_rate accordingly.
    registration.SetOptimizerScalesFromPhysicalShift()

    if optimizer_weights is not None:
        registration.SetOptimizerWeights(optimizer_weights)

    # --- Multi-resolution pyramid ---
    if use_multi_resolution:
        registration.SetShrinkFactorsPerLevel(shrinkFactors=list(shrink_factors))
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smoothing_sigmas))
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    else:
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    # --- Transform and centering initialization ---
    if transform_type == "bspline":
        sitk_centering_transform: sitk.Transform = sitk.BSplineTransformInitializer(
            fixed_reg, transformDomainMeshSize=list(mesh_size)
        )
    else:
        if transform_type == "translation":
            sitk_centering_transform: sitk.Transform = sitk.TranslationTransform(ndim)
        elif transform_type == "rigid":
            sitk_centering_transform = sitk.Euler3DTransform()
        else:
            sitk_centering_transform = sitk.AffineTransform(ndim)

        # CenteredTransformInitializer requires a transform with a center parameter
        # (e.g. Euler, Affine). TranslationTransform has no center, so centering
        # initialization is always skipped for translation.
        if initialization_mode == "center_geometry" and transform_type != "translation":
            sitk_centering_transform = sitk.CenteredTransformInitializer(
                fixed_reg,
                moving_reg,
                sitk_centering_transform,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif (
            initialization_mode == "center_moments" and transform_type != "translation"
        ):
            sitk_centering_transform = sitk.CenteredTransformInitializer(
                fixed_reg,
                moving_reg,
                sitk_centering_transform,
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )

    if affine_initialization is not None:
        pre_tx = affine_to_sitk_linear_transform(affine_initialization)
        if transform_type == "bspline":
            sitk_initial_transform = sitk.CompositeTransform(ndim)
            sitk_initial_transform.AddTransform(pre_tx)
            sitk_initial_transform.AddTransform(sitk_centering_transform)
        else:
            sitk_initial_transform = pre_tx
    else:
        sitk_initial_transform = sitk_centering_transform

    registration.SetInitialTransform(sitk_initial_transform, inPlace=True)

    # Always collect per-iteration metric values so callers get convergence
    # diagnostics regardless of whether the live progress plot is enabled.
    metric_values: list[float] = []

    def _record_iteration() -> None:
        metric_value = float(registration.GetMetricValue())
        metric_values.append(metric_value)

    needs_fill_value = resample or (show_progress and plot_composite)
    _fill_value = (
        fill_value
        if fill_value is not None
        else (float(moving.min()) if needs_fill_value else None)
    )

    with abort_on_sigint(abort_event) as effective_abort_event:
        registration.AddCommand(sitk.sitkIterationEvent, _record_iteration)
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: (
                registration.StopRegistration()
                if effective_abort_event.is_set()
                else None
            ),
        )

        if show_progress:
            from confusius.registration.progress import (
                MatplotlibRegistrationProgressPlotter,
            )

            resample_kwargs: dict[str, object] = {
                "interpolation": resample_interpolation,
                "fill_value": _fill_value,
                "sitk_threads": sitk_threads,
            }

            plotter_factory = progress_plotter or MatplotlibRegistrationProgressPlotter
            plotter = plotter_factory(
                registration,
                fixed_sitk,
                moving_sitk,
                plot_metric=plot_metric,
                plot_composite=plot_composite,
                resample_kwargs=resample_kwargs,
            )
            registration.AddCommand(sitk.sitkIterationEvent, plotter.update)
            registration.AddCommand(sitk.sitkEndEvent, plotter.close)

        executed = False
        if effective_abort_event.is_set():
            if transform_type == "bspline":
                sitk_optimized_transform = sitk_initial_transform
            elif affine_initialization is not None:
                sitk_optimized_transform = affine_to_sitk_linear_transform(
                    affine_initialization
                )
            else:
                sitk_optimized_transform = sitk.TranslationTransform(ndim)
            aborted = True
            stop_condition = "Registration aborted before optimisation started."
        else:
            try:
                with set_sitk_thread_count(sitk_threads):
                    sitk_optimized_transform = registration.Execute(
                        fixed_reg, moving_reg
                    )
            except RuntimeError as exc:
                raise _translate_registration_runtime_error(
                    exc,
                    transform_type=transform_type,
                    learning_rate=requested_learning_rate,
                ) from exc
            executed = True
            aborted = effective_abort_event.is_set()
            stop_condition = registration.GetOptimizerStopConditionDescription()
            if aborted and not stop_condition.strip():
                stop_condition = "Registration aborted."

    # When resampling, the output lives on the fixed grid; otherwise the moving volume
    # is returned unchanged and its own coordinates are preserved.
    if resample:
        interp = (
            sitk.sitkLinear if resample_interpolation == "linear" else sitk.sitkBSpline
        )
        assert _fill_value is not None
        with set_sitk_thread_count(sitk_threads):
            # .T restores numpy axis order, inverse of the .T used to build the SITK
            # image.
            registered_arr = sitk.GetArrayFromImage(
                sitk.Resample(
                    moving_sitk,
                    fixed_sitk,
                    sitk_optimized_transform,
                    interp,
                    _fill_value,
                    moving_sitk.GetPixelID(),
                )
            ).T
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
        from confusius.registration.bspline import sitk_bspline_to_dataarray

        optimized_transform = sitk_bspline_to_dataarray(
            sitk_optimized_transform, pre_affine=affine_initialization
        )
    else:
        optimized_transform = sitk_linear_transform_to_affine(sitk_optimized_transform)

    final_metric_value: float
    if metric_values:
        final_metric_value = float(metric_values[-1])
    elif executed:
        final_metric_value = float(registration.GetMetricValue())
    else:
        final_metric_value = float("nan")
    diagnostics = RegistrationDiagnostics(
        metric=metric,
        metric_values=np.asarray(metric_values, dtype=float),
        final_metric_value=final_metric_value,
        n_iterations=len(metric_values),
        stop_condition=stop_condition,
        status="aborted" if aborted else "completed",
    )

    return result, optimized_transform, diagnostics
