"""Coordinate spacing and origin helpers shared across modules."""

import warnings
from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.stack import find_stack_level


def get_representative_step(
    values: npt.NDArray[np.floating], uniformity_tolerance: float = 1e-2
) -> tuple[float | None, bool]:
    """Return a representative step size for a 1D coordinate array.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional coordinate values.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed per-interval relative deviation from the median consecutive
        difference. Each interval must satisfy
        `|interval - median| <= uniformity_tolerance * |median|` for the coordinate
        to be considered uniform. Bounds the worst single drift, which is what
        matters for downstream frequency-domain operations (filtering, FFT,
        HRF convolution).

    Returns
    -------
    step : float or None
        Exact step size when sampling is uniform, the median consecutive difference when
        sampling is non-uniform, or `None` when fewer than two values are provided.
    approximate : bool
        Whether the returned step is a median approximation derived from non-uniform
        spacing.
    """
    if len(values) < 2:
        return None, False

    diffs = np.diff(values)
    median = float(np.median(diffs))
    if np.isclose(median, 0.0):
        is_uniform = np.isclose(np.max(diffs), np.min(diffs))
    else:
        is_uniform = bool(
            np.allclose(diffs, median, rtol=uniformity_tolerance, atol=0.0)
        )

    if is_uniform:
        return median, False
    return median, True


class CoordinateSpacingInfo:
    """Result of per-dimension spacing analysis.

    Parameters
    ----------
    value : float or None
        Exact spacing when the coordinate is uniform. `None` otherwise, including for
        single-point coordinates.
    median : float or None
        Median consecutive difference for numeric coordinates with two or more points.
        Available for both uniform and non-uniform coordinates; `None` for missing,
        non-numeric, or single-point coordinates.
    warn_msg : str or None
        Warning message to emit, or `None` if no warning is needed.
    """

    __slots__ = ("median", "value", "warn_msg")

    def __init__(
        self,
        value: float | None,
        median: float | None,
        warn_msg: str | None,
    ) -> None:
        self.value = value
        self.median = median
        self.warn_msg = warn_msg


def get_coordinate_spacing_info(
    dim: str,
    data: xr.DataArray,
    uniformity_tolerance: float,
) -> CoordinateSpacingInfo:
    """Compute coordinate spacing information for a single dimension.

    Shared implementation used by
    [`get_coordinate_spacings_best_effort`][confusius._utils.coordinates.get_coordinate_spacings_best_effort]
    and other coordinate-spacing consumers to avoid duplicating the uniformity check
    and median computation.

    Parameters
    ----------
    dim : str
        Dimension name.
    data : xarray.DataArray
        DataArray whose coordinate to inspect.
    uniformity_tolerance : float
        Maximum allowed per-interval relative deviation from the median consecutive
        difference (see
        [`get_representative_step`][confusius._utils.coordinates.get_representative_step]).

    Returns
    -------
    CoordinateSpacingInfo
        Spacing result for the dimension.
    """
    if dim not in data.coords:
        return CoordinateSpacingInfo(
            value=None,
            median=None,
            warn_msg=f"Dimension '{dim}' has no coordinate; spacing is undefined.",
        )

    coord = data.coords[dim]
    # Bare int/float map to the exact int64/float64 dtypes, silently excluding
    # e.g. float32 coordinates; use the abstract numpy supertypes instead.
    if not np.issubdtype(coord.dtype, np.integer) and not np.issubdtype(
        coord.dtype, np.floating
    ):
        return CoordinateSpacingInfo(value=None, median=None, warn_msg=None)

    # A coordinate named after its dim is not necessarily 1D (e.g. a pose-dependent
    # array's "time" coordinate can be genuinely (time, pose)-shaped, holding each
    # pose's own real timestamps -- see confusius.multipose.stack_poses). Compute the
    # step along `dim` independently for every combination of the other dims'
    # values (flattened into columns) rather than trusting just one slice: an
    # answer this function reports as "the" spacing for `dim` must actually hold
    # for all of them, the same way equal spatial scale across poses is an
    # enforced invariant for voxel-to-world affines, not an assumption.
    other_dims = [d for d in coord.dims if d != dim]
    dim_axis = coord.dims.index(dim)

    if coord.shape[dim_axis] < 2:
        return CoordinateSpacingInfo(
            value=None,
            median=None,
            warn_msg=(
                f"Dimension '{dim}' has a single coordinate point; spacing is "
                "undefined."
            ),
        )

    values = np.moveaxis(coord.values, dim_axis, 0).reshape(coord.shape[dim_axis], -1)

    steps: list[float] = []
    for column in range(values.shape[1]):
        step, is_approximate = get_representative_step(
            values[:, column], uniformity_tolerance=uniformity_tolerance
        )
        assert step is not None  # values.shape[0] >= 2, checked above.
        if is_approximate:
            return CoordinateSpacingInfo(
                value=None,
                median=float(np.median(steps + [step])),
                warn_msg=(
                    f"Coordinate '{dim}' has non-uniform sampling; spacing is "
                    "undefined."
                ),
            )
        steps.append(step)

    if other_dims and not np.allclose(
        steps, steps[0], rtol=uniformity_tolerance, atol=0.0
    ):
        return CoordinateSpacingInfo(
            value=None,
            median=float(np.median(steps)),
            warn_msg=(
                f"Coordinate '{dim}' spacing along '{dim}' differs across "
                f"{other_dims!r}; spacing is undefined."
            ),
        )

    return CoordinateSpacingInfo(value=steps[0], median=steps[0], warn_msg=None)


def get_coordinate_spacings_best_effort(
    da: xr.DataArray, uniformity_tolerance: float = 1e-2
) -> tuple[dict[str, float], list[str]]:
    """Compute coordinate spacing, falling back to median diff for non-uniform dims.

    For each dimension, returns the median step size if the coordinate has two or
    more points and is uniformly sampled. Otherwise it returns the median
    consecutive difference as a best-effort approximation rather than `None`. This
    is appropriate when a single representative spacing is required (e.g. for
    napari's `scale` parameter) even though the coordinate is not perfectly
    uniform. No warnings are emitted; the caller is responsible for issuing
    context-appropriate messages for the dims listed in `non_uniform`.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate spacing to compute.
    uniformity_tolerance : float, default: 1e-2
        Passed through to the uniformity check (see
        [`get_coordinate_spacings`][confusius._utils.coordinates.get_coordinate_spacings]).

    Returns
    -------
    spacing : dict[str, float]
        Spacing per dimension. Always a `float`; never `None`. Falls back to
        `1.0` for dimensions with missing, non-numeric, or single-point coordinates.
    non_uniform : list[str]
        Names of dimensions whose coordinates were non-uniform. The median diff
        was used as the spacing for these dims.
    """
    spacing: dict[str, float] = {}
    non_uniform: list[str] = []
    for dim in (str(d) for d in da.dims):
        r = get_coordinate_spacing_info(dim, da, uniformity_tolerance)
        if r.value is not None:
            spacing[dim] = r.value
        elif r.median is not None:
            spacing[dim] = r.median
            non_uniform.append(dim)
        else:
            spacing[dim] = 1.0
    return spacing, non_uniform


def get_coordinate_origins(data: xr.DataArray) -> dict[str, float]:
    """Return the world origin (first coordinate value) for each dimension.

    For each dimension, returns the first coordinate value. If a coordinate is missing,
    falls back to `0.0` with a warning. If a coordinate is non-numeric (e.g.
    string-based), falls back to `0.0` without a warning.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray whose coordinate origins to compute.

    Returns
    -------
    dict[str, float]
        Origin per dimension in DataArray dimension order.
    """
    result: dict[str, float] = {}
    for dim in (str(d) for d in data.dims):
        if dim not in data.coords:
            warnings.warn(
                f"Dimension '{dim}' has no coordinate; origin defaults to 0.0.",
                stacklevel=find_stack_level(),
            )
            result[dim] = 0.0
        else:
            coord = data.coords[dim]
            if not np.issubdtype(coord.dtype, np.integer) and not np.issubdtype(
                coord.dtype, np.floating
            ):
                # Non-numeric coordinate (e.g., strings); fall back to 0.0.
                result[dim] = 0.0
            else:
                result[dim] = float(coord.values.flat[0])
    return result


def get_dim_keyed_origin(data: xr.DataArray) -> dict[str, float]:
    """Return `data.fusi.origin`, re-keyed by dimension name for voxel-to-world data.

    `data.fusi.origin` keys voxel-to-world spatial dims by their world coordinate
    name (e.g. `"z"`), not by the native voxel dimension name (e.g. `"k"`) used
    elsewhere (e.g. `data.fusi.spacing`, `data.dims`). This aligns the two
    conventions, keeping any non-spatial dims (e.g. `time`) as `data.fusi.origin`
    already keys them.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.

    Returns
    -------
    dict[str, float]
        Origin keyed by `data`'s own dimension names.

    Raises
    ------
    ValueError
        If `data` does not carry a voxel-to-world index.
    """
    from confusius._utils.geometry import (
        get_voxel_to_world_coord_names,
        get_voxel_to_world_spatial_dims,
    )

    origin = data.fusi.origin
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    world_names = get_voxel_to_world_coord_names(data)
    return {
        **origin,
        **{
            voxel_dim: origin[world_name]
            for voxel_dim, world_name in zip(voxel_dims, world_names, strict=True)
        },
    }


class GridKwargs(TypedDict):
    """Output grid specification for SimpleITK-based resampling.

    Bundles the keyword arguments (`shape`, `spacing`, `origin`, `dims`) that
    resampling helpers accept, each a list in DataArray dimension order.
    """

    shape: list[int]
    spacing: list[float]
    origin: list[float]
    dims: list[str]


def get_grid_info_from_dataarray(
    data: xr.DataArray,
    dims: Sequence[str] | None = None,
    *,
    error_prefix: str = "Cannot build grid kwargs because spacing is undefined",
) -> GridKwargs:
    """Return the resampling grid specification extracted from a VoxelData array.

    Bundles the `shape`, `spacing`, `origin`, and `dims` that a VoxelData array
    defines into one `dims`-ordered specification, reconciling
    [`data.fusi.spacing`][confusius.xarray.accessors.FUSIAccessor.spacing] (keyed by
    native voxel dimension name) with
    [`data.fusi.origin`][confusius.xarray.accessors.FUSIAccessor.origin] (keyed by
    world coordinate name for spatial dimensions, dimension name otherwise) into a
    single triple per dimension. Non-spatial origins fall back to
    [`get_coordinate_origins`][confusius._utils.coordinates.get_coordinate_origins].
    Each requested dimension must have defined spacing; a singleton non-spatial
    dimension has no defined spacing.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array.
    dims : sequence[str], optional
        Dimensions to extract, in the desired output order. Must be a subset of
        `data`'s dimensions. If not provided, all of `data`'s dimensions are used.
    error_prefix : str, default: "Cannot build grid kwargs because spacing is undefined"
        Start of the error message raised when spacing is undefined for any of the
        requested dimensions, stating what could not be built.

    Returns
    -------
    GridKwargs
        Dictionary with `shape`, `spacing`, `origin`, and `dims` keys, each a list in
        the requested dimension order.

    Raises
    ------
    ValueError
        If `data` is not a valid VoxelData array, or if spacing is undefined for any
        requested dimension.
    """
    from confusius.validation.voxeldata import ensure_voxeldata

    data = ensure_voxeldata(data)
    dims = [str(dim) for dim in data.dims] if dims is None else list(dims)

    spacings = data.fusi.spacing
    origin = get_dim_keyed_origin(data)
    missing_spacing = [dim for dim in dims if spacings[dim] is None]
    if missing_spacing:
        raise ValueError(
            f"{error_prefix} for dimensions {missing_spacing!r}. Provide regular "
            "(2+ point, uniformly sampled) coordinates for these dimensions."
        )
    return {
        "shape": [int(data.sizes[dim]) for dim in dims],
        "spacing": [float(spacings[dim]) for dim in dims],
        "origin": [float(origin[dim]) for dim in dims],
        "dims": dims,
    }


def get_axis_aligned_affine(
    translation: npt.NDArray[np.floating],
    zoom: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Build the axis-aligned homogeneous affine `[[diag(zoom), translation]]`.

    Parameters
    ----------
    translation : (3,) numpy.ndarray
        Translation vector placed in the last column.
    zoom : (3,) numpy.ndarray
        Per-axis scaling.

    Returns
    -------
    (4, 4) numpy.ndarray
        Homogeneous affine with `diag(zoom)` and `translation` as its last column.
    """
    affine = np.eye(4)
    affine[:3, :3] = np.diag(np.asarray(zoom, dtype=np.float64))
    affine[:3, 3] = np.asarray(translation, dtype=np.float64)
    return affine


def get_probe_surface_origin(
    sizes: Mapping[str, int], spacing: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the default world origin for ConfUSIus probe geometry.

    Elevation (`k`) and lateral (`i`) are centered on the probe; depth (`j`) is
    referenced to the probe surface, starting half a voxel below it.

    Parameters
    ----------
    sizes : mapping[str, int]
        Spatial voxel sizes keyed by native `k`/`j`/`i` dimension name.
    spacing : tuple[float, float, float]
        World spacing in `z`/`y`/`x` order.

    Returns
    -------
    tuple[float, float, float]
        Default origin in `z`/`y`/`x` order.
    """
    return (
        -spacing[0] * (sizes["k"] - 1) / 2,
        spacing[1] / 2,
        -spacing[2] * (sizes["i"] - 1) / 2,
    )
