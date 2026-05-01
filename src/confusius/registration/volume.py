"""Volume-to-volume registration for fUSI data."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.registration._utils import (
    dataarray_to_sitk_image,
    replace_affines_attr,
    set_sitk_thread_count,
)
from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    sitk_linear_transform_to_affine,
)

if TYPE_CHECKING:
    import SimpleITK as sitk


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
    centering_initialization: Literal["geometry", "moments", "none"],
    shrink_factors: Sequence[int],
    smoothing_sigmas: Sequence[int],
    resample_interpolation: Literal["linear", "bspline"],
) -> None:
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
    centering_initialization : {"geometry", "moments", "none"}
        Transform initializer name.
    shrink_factors : sequence of int
        Downsampling factors per pyramid level.
    smoothing_sigmas : sequence of int
        Smoothing sigmas per pyramid level.
    resample_interpolation : {"linear", "bspline"}
        Interpolator name used when resampling.

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
    if moving.ndim not in (2, 3):
        raise ValueError(
            f"register_volume expects 2D or 3D inputs, 'moving' is {moving.ndim}D."
        )
    if fixed.ndim not in (2, 3):
        raise ValueError(
            f"register_volume expects 2D or 3D inputs, 'fixed' is {fixed.ndim}D."
        )

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

    valid_initializations = {"geometry", "moments", "none"}
    if centering_initialization not in valid_initializations:
        raise ValueError(
            f"Invalid initialization {centering_initialization!r}. "
            f"Expected one of {sorted(valid_initializations)}."
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
        validate_mask(fixed_mask, fixed, "fixed_mask")
    if moving_mask is not None:
        validate_mask(moving_mask, moving, "moving_mask")

    # --- Multi-resolution consistency ---
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError(
            f"shrink_factors and smoothing_sigmas must have the same length; "
            f"got {len(shrink_factors)} and {len(smoothing_sigmas)}."
        )


def _expand_thin_dims(img: "sitk.Image", min_size: int = 4) -> "sitk.Image":
    """Expand any image dimension smaller than `min_size` by replication.

    SimpleITK's registration and multi-resolution pyramid fail when any spatial
    dimension is smaller than 4 voxels. This helper replicates thin dimensions so that
    the image is safe to register, while preserving the physical extent (spacing is
    divided by the expansion factor, keeping `size * spacing` constant).

    Parameters
    ----------
    img : SimpleITK.Image
        Input image. May be 2D or 3D.
    min_size : int, default: 4
        Minimum acceptable size along each dimension.

    Returns
    -------
    SimpleITK.Image
        Image with all dimensions >= `min_size`. Returns `img` unchanged if no
        dimension is too small.
    """
    import SimpleITK as sitk

    size = np.array(img.GetSize())
    factors = np.ones(len(size), dtype=int)
    thin = size < min_size
    if not thin.any():
        return img

    factors[thin] = np.ceil(min_size / size[thin]).astype(int)

    # sitk.Expand replicates voxels and halves spacing proportionally.
    return sitk.Expand(img, factors.tolist())


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
    centering_initialization: Literal["geometry", "moments", "none"] = ...,
    optimizer_weights: list[float] | None = ...,
    initial_transform: "npt.NDArray[np.floating] | None" = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    sitk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating]]":
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
    centering_initialization: Literal["geometry", "moments", "none"] = ...,
    optimizer_weights: list[float] | None = ...,
    initial_transform: "npt.NDArray[np.floating] | None" = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    sitk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
) -> "tuple[xr.DataArray, xr.DataArray]":
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
    centering_initialization: Literal["geometry", "moments", "none"] = ...,
    optimizer_weights: list[float] | None = ...,
    initial_transform: "npt.NDArray[np.floating] | None" = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    sitk_threads: int = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating]]":
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
    centering_initialization: Literal["geometry", "moments", "none"] = "geometry",
    optimizer_weights: list[float] | None = None,
    initial_transform: "npt.NDArray[np.floating] | None" = None,
    mesh_size: tuple[int, int, int] = (10, 10, 10),
    use_multi_resolution: bool = False,
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    resample: bool = True,
    resample_interpolation: Literal["linear", "bspline"] = "linear",
    sitk_threads: int = -1,
    show_progress: bool = False,
    plot_metric: bool = True,
    plot_composite: bool = True,
) -> "tuple[xr.DataArray, npt.NDArray[np.floating] | xr.DataArray]":
    """Register a single 2D or 3D volume to a fixed reference.

    Voxel spacing and origin are automatically extracted from the DataArray coordinates.
    Both inputs must be spatial-only (no `time` dimension).

    Parameters
    ----------
    moving : xarray.DataArray
        Volume to register to `fixed`. Must be 2D or 3D.
    fixed : xarray.DataArray
        Reference volume. Must be 2D or 3D. Need not have the same shape as
        `moving`.
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
    centering_initialization : {"geometry", "moments", "none"}, default: "geometry"
        Transform initializer applied before optimization. `"geometry"` aligns the
        image centers (safe default, no assumptions about content). `"moments"` aligns
        centers of mass (better when images are offset but share the same content).
        `"none"` uses the identity transform. Ignored for `transform_type="bspline"`.
    optimizer_weights : list of float, optional
        Per-parameter weights applied on top of the auto-estimated physical shift
        scales. `None` uses identity weights (all ones). A list is passed directly to
        SimpleITK's `SetOptimizerWeights`; its length must match the number of
        transform parameters (3 for 2D rigid, 6 for 3D rigid, 6 for 2D affine, 12 for 3D
        affine). The weight for each parameter is multiplied into the effective step
        size: `0` freezes a parameter entirely, values in `(0, 1)` slow it down, and
        `1` leaves it unchanged. For the 3D Euler transform the parameter order is
        `[angleX, angleY, angleZ, tx, ty, tz]`; to disable rotations around x and y
        set weights to `[0, 0, 1, 1, 1, 1]`.
    initial_transform : (N+1, N+1) numpy.ndarray, optional
        Pre-computed affine matrix (pull/inverse convention, as returned by a previous
        call to `register_volume`) used as a warm-start before optimisation. When
        provided the optimized transform (`transform_type`) is composed on top of this
        pre-alignment. Primarily useful for the affine → B-spline composition
        workflow: run an affine registration first, then pass its result here together
        with `transform_type="bspline"` to refine the deformation.  When not provided
        the existing `centering_initialization` behaviour is unchanged.
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

    Returns
    -------
    registered : xarray.DataArray
        When `resample=True`, the moving volume resampled onto the fixed grid with
        coordinates matching `fixed` and physical-space affines inherited from `fixed`.
        When `resample=False`, the original moving volume with its original coordinates
        and attributes.
    transform : (N+1, N+1) numpy.ndarray or xarray.DataArray or None
        Estimated registration transform. For linear transforms (`"translation"`,
        `"rigid"`, `"affine"`), returns a homogeneous affine matrix of shape `(N+1,
        N+1)` in physical space, where `N` is the spatial dimensionality (2 or 3).
        Follows SimpleITK's pull/inverse convention: the matrix maps fixed-space
        coordinates to moving-space coordinates. For `transform_type="bspline"`,
        returns a DataArray encoding the B-spline control-point grid (see
        [`confusius.registration.bspline`][confusius.registration.bspline] for the
        DataArray schema). When `initial_transform` was also supplied, the DataArray
        includes `attrs["affines"]["bspline_initialization"]` so that the full composite
        (pre-affine + B-spline) can be reconstructed for resampling.

    Raises
    ------
    ValueError
        If either input contains a `time` dimension or is not 2D or 3D.
    ValueError
        If `moving` or `fixed` contains NaN values.
    ValueError
        If `transform_type`, `metric`, `centering_initialization`, or
        `resample_interpolation` is not a recognised value.
    ValueError
        If `learning_rate` is not a positive finite float or `"auto"`.
    ValueError
        If `number_of_iterations`, `convergence_window_size`, or
        `number_of_histogram_bins` is not a positive integer.
    ValueError
        If `shrink_factors` and `smoothing_sigmas` have different lengths.
    ValueError
        If `initial_transform` is provided and its shape does not match the image
        dimensionality.
    TypeError
        If `fixed_mask` or `moving_mask` is not a boolean DataArray.
    ValueError
        If `fixed_mask` shape does not match `fixed` or `moving_mask` shape does
        not match `moving`.
    """
    import SimpleITK as sitk

    _validate_register_volume_inputs(
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
        centering_initialization=centering_initialization,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        resample_interpolation=resample_interpolation,
    )

    fixed_sitk = dataarray_to_sitk_image(fixed)
    moving_sitk = dataarray_to_sitk_image(moving)

    # SimpleITK's multi-resolution pyramid and interpolation fail when any spatial
    # dimension is smaller than 4 voxels (common for 2D+t fUSI recordings with a 1-voxel
    # depth). We thus expand thin dimensions before registration; the originals are kept
    # as the resample source/reference so the output grid is never affected.
    fixed_reg = _expand_thin_dims(fixed_sitk)
    moving_reg = _expand_thin_dims(moving_sitk)

    # CenteredTransformInitializer (and the registration method) require both images to
    # have the same pixel type. Cast to Float64 to normalise across dtype combinations.
    fixed_reg = sitk.Cast(fixed_reg, sitk.sitkFloat64)
    moving_reg = sitk.Cast(moving_reg, sitk.sitkFloat64)

    ndim = fixed_reg.GetDimension()

    # Validate initial_transform shape now that ndim is known.
    if initial_transform is not None:
        expected_shape = (ndim + 1, ndim + 1)
        if initial_transform.shape != expected_shape:
            raise ValueError(
                f"initial_transform shape {initial_transform.shape} does not match "
                f"image dimensionality {ndim}D (expected {expected_shape})."
            )

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
        fixed_mask_sitk = _expand_thin_dims(fixed_mask_sitk)
        registration.SetMetricFixedMask(fixed_mask_sitk)
    if moving_mask is not None:
        # Convert boolean mask to uint8 for SimpleITK
        moving_mask_uint8 = moving_mask.astype(np.uint8)
        moving_mask_sitk = dataarray_to_sitk_image(moving_mask_uint8)
        # Expand mask if image was expanded
        moving_mask_sitk = _expand_thin_dims(moving_mask_sitk)
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
            sitk_centering_transform = (
                sitk.Euler2DTransform() if ndim == 2 else sitk.Euler3DTransform()
            )
        else:
            sitk_centering_transform = sitk.AffineTransform(ndim)

        # CenteredTransformInitializer requires a transform with a center parameter
        # (e.g. Euler, Affine). TranslationTransform has no center, so centering
        # initialization is always skipped for translation.
        if centering_initialization == "geometry" and transform_type != "translation":
            sitk_centering_transform = sitk.CenteredTransformInitializer(
                fixed_reg,
                moving_reg,
                sitk_centering_transform,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif centering_initialization == "moments" and transform_type != "translation":
            sitk_centering_transform = sitk.CenteredTransformInitializer(
                fixed_reg,
                moving_reg,
                sitk_centering_transform,
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        else:
            sitk_centering_transform = sitk_centering_transform

    if initial_transform is not None:
        pre_tx = affine_to_sitk_linear_transform(initial_transform)

        sitk_initial_transform: sitk.Transform = sitk.CompositeTransform(ndim)
        sitk_initial_transform.AddTransform(pre_tx)
        sitk_initial_transform.AddTransform(sitk_centering_transform)
    else:
        sitk_initial_transform = sitk_centering_transform

    registration.SetInitialTransform(sitk_initial_transform, inPlace=True)

    if show_progress:
        from confusius.registration._progress import RegistrationProgressPlotter

        plotter = RegistrationProgressPlotter(
            registration,
            fixed_sitk,
            moving_sitk,
            plot_metric=plot_metric,
            plot_composite=plot_composite,
        )
        registration.AddCommand(sitk.sitkIterationEvent, plotter.update)
        registration.AddCommand(sitk.sitkEndEvent, plotter.close)

    with set_sitk_thread_count(sitk_threads):
        sitk_optimized_transform = registration.Execute(fixed_reg, moving_reg)

    # When resampling, the output lives on the fixed grid; otherwise the moving volume
    # is returned unchanged and its own coordinates are preserved.
    if resample:
        interp = (
            sitk.sitkLinear if resample_interpolation == "linear" else sitk.sitkBSpline
        )
        with set_sitk_thread_count(sitk_threads):
            # .T restores numpy axis order, inverse of the .T used to build the SITK
            # image.
            registered_arr = sitk.GetArrayFromImage(
                sitk.Resample(
                    moving_sitk,
                    fixed_sitk,
                    sitk_optimized_transform,
                    interp,
                    0.0,
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
            sitk_optimized_transform, pre_affine=initial_transform
        )
    else:
        optimized_transform = sitk_linear_transform_to_affine(sitk_optimized_transform)

    return result, optimized_transform
