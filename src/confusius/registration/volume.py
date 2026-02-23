"""Volume-to-volume registration for fUSI data."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import _compute_origin, _compute_spacing
from confusius.registration.affines import sitk_transform_to_affine

if TYPE_CHECKING:
    import SimpleITK as sitk


def _dataarray_to_sitk(da: xr.DataArray) -> "sitk.Image":
    """Convert a spatial DataArray to a SimpleITK image.

    Uses the transpose convention: ``da.values.T`` is passed to ``GetImageFromArray``,
    so that the first DataArray axis maps to SimpleITK's physical x-axis. The DataArray
    must be spatial-only (no ``time`` dimension).

    Parameters
    ----------
    da : xarray.DataArray
        2D or 3D spatial DataArray. Spacing and origin are derived from its coordinates;
        missing coordinates warn and fall back to spacing ``1.0`` and origin ``0.0``.

    Returns
    -------
    SimpleITK.Image
        SimpleITK image with spacing and origin set from the DataArray coordinates.
    """
    import SimpleITK as sitk

    spacing = tuple(s if s is not None else 1.0 for s in _compute_spacing(da).values())
    origin = tuple(_compute_origin(da).values())
    image = sitk.GetImageFromArray(da.values.T)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def _validate_register_volume_inputs(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    transform: Literal["translation", "rigid", "affine", "bspline"],
    metric: Literal["correlation", "mattes_mi"],
    number_of_histogram_bins: int,
    learning_rate: float | Literal["auto"],
    number_of_iterations: int,
    convergence_window_size: int,
    initialization: Literal["geometry", "moments", "none"],
    shrink_factors: Sequence[int],
    smoothing_sigmas: Sequence[int],
    resample_interpolation: Literal["linear", "bspline"],
) -> None:
    """Validate all inputs to :func:`register_volume` before any computation.

    Parameters
    ----------
    moving : xarray.DataArray
        Volume to register.
    fixed : xarray.DataArray
        Reference volume.
    transform : {"translation", "rigid", "affine", "bspline"}
        Transform model name.
    metric : {"correlation", "mattes_mi"}
        Similarity metric name.
    number_of_histogram_bins : int
        Number of histogram bins for Mattes mutual information.
    learning_rate : float or "auto"
        Optimizer step size or ``"auto"``.
    number_of_iterations : int
        Maximum number of optimizer iterations.
    convergence_window_size : int
        Window size for convergence checking.
    initialization : {"geometry", "moments", "none"}
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

    # --- Literal-valued parameters ---
    valid_transforms = {"translation", "rigid", "affine", "bspline"}
    if transform not in valid_transforms:
        raise ValueError(
            f"Invalid transform {transform!r}. "
            f"Expected one of {sorted(valid_transforms)}."
        )

    valid_metrics = {"correlation", "mattes_mi"}
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric {metric!r}. Expected one of {sorted(valid_metrics)}."
        )

    valid_initializations = {"geometry", "moments", "none"}
    if initialization not in valid_initializations:
        raise ValueError(
            f"Invalid initialization {initialization!r}. "
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

    # --- Multi-resolution consistency ---
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError(
            f"shrink_factors and smoothing_sigmas must have the same length; "
            f"got {len(shrink_factors)} and {len(smoothing_sigmas)}."
        )


def _expand_thin_dims(img: "sitk.Image", min_size: int = 4) -> "sitk.Image":
    """Expand any image dimension smaller than ``min_size`` by replication.

    SimpleITK's registration and multi-resolution pyramid fail when any spatial
    dimension is smaller than 4 voxels. This helper replicates thin dimensions so that
    the image is safe to register, while preserving the physical extent (spacing is
    divided by the expansion factor, keeping ``size * spacing`` constant).

    Parameters
    ----------
    img : SimpleITK.Image
        Input image. May be 2D or 3D.
    min_size : int, default: 4
        Minimum acceptable size along each dimension.

    Returns
    -------
    SimpleITK.Image
        Image with all dimensions >= ``min_size``. Returns `img` unchanged if no
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
    transform: Literal["translation", "rigid", "affine"],
    metric: Literal["correlation", "mattes_mi"] = ...,
    number_of_histogram_bins: int = ...,
    learning_rate: float | Literal["auto"] = ...,
    number_of_iterations: int = ...,
    convergence_minimum_value: float = ...,
    convergence_window_size: int = ...,
    initialization: Literal["geometry", "moments", "none"] = ...,
    optimizer_weights: list[float] | None = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.float64]]":
    """Overload for linear transforms (translation/rigid/affine)."""
    ...


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    transform: Literal["bspline"],
    metric: Literal["correlation", "mattes_mi"] = ...,
    number_of_histogram_bins: int = ...,
    learning_rate: float | Literal["auto"] = ...,
    number_of_iterations: int = ...,
    convergence_minimum_value: float = ...,
    convergence_window_size: int = ...,
    initialization: Literal["geometry", "moments", "none"] = ...,
    optimizer_weights: list[float] | None = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
) -> "tuple[xr.DataArray, None]":
    """Overload for bspline transform (returns None affine)."""
    ...


@overload
def register_volume(  # numpydoc ignore=GL08,PR01,RT01
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    metric: Literal["correlation", "mattes_mi"] = ...,
    number_of_histogram_bins: int = ...,
    learning_rate: float | Literal["auto"] = ...,
    number_of_iterations: int = ...,
    convergence_minimum_value: float = ...,
    convergence_window_size: int = ...,
    initialization: Literal["geometry", "moments", "none"] = ...,
    optimizer_weights: list[float] | None = ...,
    mesh_size: tuple[int, int, int] = ...,
    use_multi_resolution: bool = ...,
    shrink_factors: Sequence[int] = ...,
    smoothing_sigmas: Sequence[int] = ...,
    resample: bool = ...,
    resample_interpolation: Literal["linear", "bspline"] = ...,
    show_progress: bool = ...,
    plot_metric: bool = ...,
    plot_composite: bool = ...,
) -> "tuple[xr.DataArray, npt.NDArray[np.float64]]":
    """Overload for default transform (rigid, returns affine)."""
    ...


def register_volume(
    moving: xr.DataArray,
    fixed: xr.DataArray,
    *,
    transform: Literal["translation", "rigid", "affine", "bspline"] = "rigid",
    metric: Literal["correlation", "mattes_mi"] = "correlation",
    number_of_histogram_bins: int = 50,
    learning_rate: float | Literal["auto"] = "auto",
    number_of_iterations: int = 100,
    convergence_minimum_value: float = 1e-6,
    convergence_window_size: int = 10,
    initialization: Literal["geometry", "moments", "none"] = "geometry",
    optimizer_weights: list[float] | None = None,
    mesh_size: tuple[int, int, int] = (10, 10, 10),
    use_multi_resolution: bool = False,
    shrink_factors: Sequence[int] = (6, 2, 1),
    smoothing_sigmas: Sequence[int] = (6, 2, 1),
    resample: bool = False,
    resample_interpolation: Literal["linear", "bspline"] = "linear",
    show_progress: bool = False,
    plot_metric: bool = True,
    plot_composite: bool = True,
) -> "tuple[xr.DataArray, npt.NDArray[np.float64] | None]":
    """Register a single 2D or 3D volume to a fixed reference.

    Voxel spacing and origin are automatically extracted from the DataArray coordinates.
    Both inputs must be spatial-only (no ``time`` dimension).

    Parameters
    ----------
    moving : xarray.DataArray
        Volume to register to `fixed`. Must be 2D or 3D.
    fixed : xarray.DataArray
        Reference volume. Must be 2D or 3D. Need not have the same shape as
        `moving`.
    transform : {"translation", "rigid", "affine", "bspline"}, default: "rigid"
        Transform model to use during registration. ``"translation"`` allows
        only shifts. ``"rigid"`` adds rotation. ``"affine"`` adds scaling and
        shearing. ``"bspline"`` fits a non-linear deformable transform (see
        ``mesh_size``).
    metric : {"correlation", "mattes_mi"}, default: "correlation"
        Similarity metric. ``"correlation"`` (normalized cross-correlation) is
        appropriate for same-modality registration. ``"mattes_mi"`` (Mattes
        mutual information) is better suited for multi-modal registration or
        when the intensity relationship between images is non-linear.
    number_of_histogram_bins : int, default: 50
        Number of histogram bins used by Mattes mutual information. Only
        relevant when using ``"mattes_mi"`` metric.
    learning_rate : float or "auto", default: "auto"
        Optimizer step size in normalized units. ``"auto"`` re-estimates the rate at
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
    initialization : {"geometry", "moments", "none"}, default: "geometry"
        Transform initializer applied before optimization. ``"geometry"`` aligns the
        image centers (safe default, no assumptions about content). ``"moments"`` aligns
        centers of mass (better when images are offset but share the same content).
        ``"none"`` uses the identity transform. Ignored for ``transform="bspline"``.
    optimizer_weights : list of float or None, default: None
        Per-parameter weights applied on top of the auto-estimated physical shift
        scales. ``None`` uses identity weights (all ones). A list is passed directly to
        SimpleITK's ``SetOptimizerWeights``; its length must match the number of
        transform parameters (3 for 2D rigid, 6 for 3D rigid, 6 for 2D affine, 12 for 3D
        affine). The weight for each parameter is multiplied into the effective step
        size: ``0`` freezes a parameter entirely, values in ``(0, 1)`` slow it down, and
        ``1`` leaves it unchanged. For the 3D Euler transform the parameter order is
        ``[angleX, angleY, angleZ, tx, ty, tz]``; to disable rotations around x and y
        set weights to ``[0, 0, 1, 1, 1, 1]``.
    mesh_size : tuple of int, default: (10, 10, 10)
        Number of B-spline mesh nodes along each spatial dimension. Only used when
        ``transform="bspline"``.
    use_multi_resolution : bool, default: False
        Whether to use a multi-resolution pyramid during registration. When ``True``,
        registration proceeds from a coarse downsampled version of the images to the
        full resolution, which improves convergence for large displacements and reduces
        the risk of local minima.
    shrink_factors : sequence of int, default: (6, 2, 1)
        Downsampling factor at each pyramid level, from coarsest to finest. Must have
        the same length as ``smoothing_sigmas``. Only used when
        ``use_multi_resolution=True``.
    smoothing_sigmas : sequence of int, default: (6, 2, 1)
        Gaussian smoothing sigma (in voxels) applied at each pyramid level, from
        coarsest to finest. Must have the same length as ``shrink_factors``. Only used
        when ``use_multi_resolution=True``.
    resample : bool, default: False
        Whether to resample the moving volume onto the fixed grid after estimating the
        transform. When ``True``, the output is resampled onto the fixed grid and its
        coordinates match ``fixed``. When ``False`` (the default), only the transform is
        computed and the moving volume is returned unchanged with its original
        coordinates.
    resample_interpolation : {"linear", "bspline"}, default: "linear"
        Interpolator used when resampling the moving volume onto the fixed grid.
        ``"linear"`` is fast and appropriate for most cases. ``"bspline"`` (3rd-order
        B-spline) produces smoother results and reduces ringing, useful for atlas
        registration. Only used when ``resample=True``.
    show_progress : bool, default: False
        Whether to display a live progress plot during registration. The plot is shown
        in a Jupyter notebook or in an interactive matplotlib window depending on the
        active backend.
    plot_metric : bool, default: True
        Whether to include the optimizer metric curve in the progress plot. Ignored when
        ``show_progress=False``.
    plot_composite : bool, default: True
        Whether to include a fixed/moving composite overlay in the progress plot.
        Requires resampling the moving image at every iteration. Ignored when
        ``show_progress=False``.

    Returns
    -------
    registered : xarray.DataArray
        When ``resample=True``, the moving volume resampled onto the fixed grid with
        coordinates matching `fixed`. When ``resample=False``, the original moving
        volume with a `registration` attribute added to its metadata.
    affine : (N+1, N+1) numpy.ndarray or None
        Estimated registration transform as a homogeneous affine matrix in
        physical space, where ``N`` is the spatial dimensionality (2 or 3).
        Follows SimpleITK's pull/inverse convention: the matrix maps
        fixed-space coordinates to moving-space coordinates, i.e. for each
        output (fixed-grid) point it gives the corresponding location in the
        moving image. For non-linear transforms (``transform="bspline"``),
        returns ``None`` because a B-spline deformation field cannot be
        represented as a finite affine matrix.

    Raises
    ------
    ValueError
        If either input contains a ``time`` dimension or is not 2D or 3D.
    ValueError
        If ``transform``, ``metric``, ``initialization``, or
        ``resample_interpolation`` is not a recognised value.
    ValueError
        If ``learning_rate`` is not a positive finite float or ``"auto"``.
    ValueError
        If ``number_of_iterations``, ``convergence_window_size``, or
        ``number_of_histogram_bins`` is not a positive integer.
    ValueError
        If ``shrink_factors`` and ``smoothing_sigmas`` have different lengths.
    """
    import SimpleITK as sitk

    _validate_register_volume_inputs(
        moving=moving,
        fixed=fixed,
        transform=transform,
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

    fixed_sitk = _dataarray_to_sitk(fixed)
    moving_sitk = _dataarray_to_sitk(moving)

    # SimpleITK's multi-resolution pyramid and interpolation fail when any spatial
    # dimension is smaller than 4 voxels (common for 2D+t fUSI recordings with a
    # 1-voxel depth). We thus expand thin dimensions before registration; the originals
    # are kept as the resample source/reference so the output grid is never affected.
    fixed_reg = _expand_thin_dims(fixed_sitk)
    moving_reg = _expand_thin_dims(moving_sitk)

    ndim = fixed_reg.GetDimension()

    registration = sitk.ImageRegistrationMethod()

    # --- Metric ---
    if metric == "correlation":
        registration.SetMetricAsCorrelation()
    else:
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=number_of_histogram_bins
        )

    registration.SetInterpolator(sitk.sitkLinear)

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

    # --- Transform and initialization ---
    if transform == "bspline":
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_reg, transformDomainMeshSize=list(mesh_size)
        )
    else:
        if transform == "translation":
            tx = sitk.TranslationTransform(ndim)
        elif transform == "rigid":
            tx = sitk.Euler2DTransform() if ndim == 2 else sitk.Euler3DTransform()
        else:
            tx = sitk.AffineTransform(ndim)

        # CenteredTransformInitializer requires a transform with a center parameter
        # (e.g. Euler, Affine). TranslationTransform has no center, so initialization is
        # always skipped for translation.
        if initialization == "geometry" and transform != "translation":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_reg,
                moving_reg,
                tx,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif initialization == "moments" and transform != "translation":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_reg,
                moving_reg,
                tx,
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
        else:
            initial_transform = tx

    registration.SetInitialTransform(initial_transform, inPlace=True)

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

    result_transform = registration.Execute(fixed_reg, moving_reg)

    # When resampling, the output lives on the fixed grid; otherwise the moving volume
    # is returned unchanged and its own coordinates are preserved.
    if resample:
        interp = (
            sitk.sitkLinear if resample_interpolation == "linear" else sitk.sitkBSpline
        )
        # .T restores numpy axis order (inverse of the .T applied in _dataarray_to_sitk).
        registered_arr = sitk.GetArrayFromImage(
            sitk.Resample(
                moving_sitk,
                fixed_sitk,
                result_transform,
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
    result.attrs["registration"] = "volume"

    return result, sitk_transform_to_affine(result_transform)
