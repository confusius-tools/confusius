"""Coordinate spacing and origin helpers shared across modules."""

import warnings
from collections.abc import Hashable, Sequence

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
        Exact spacing when the coordinate is uniform or a single-point coord with a
        `voxdim` attribute. `None` otherwise.
    median : float or None
        Median consecutive difference for numeric coordinates with two or more points.
        Available for both uniform and non-uniform coordinates; `None` for missing,
        non-numeric, or single-point coordinates.
    warn_msg : str or None
        Warning message to emit, or `None` if no warning is needed.
    """

    __slots__ = ("value", "median", "warn_msg")

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
    [`get_coordinate_spacings`][confusius._utils.coordinates.get_coordinate_spacings]
    and
    [`get_coordinate_spacings_best_effort`][confusius._utils.coordinates.get_coordinate_spacings_best_effort]
    to avoid duplicating the uniformity check and median computation.

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
    if not np.issubdtype(coord.dtype, int) and not np.issubdtype(coord.dtype, float):
        return CoordinateSpacingInfo(value=None, median=None, warn_msg=None)

    if len(coord) < 2:
        if "voxdim" in coord.attrs:
            return CoordinateSpacingInfo(
                value=float(coord.attrs["voxdim"]), median=None, warn_msg=None
            )
        return CoordinateSpacingInfo(
            value=None,
            median=None,
            warn_msg=(
                f"Dimension '{dim}' has a single coordinate point and no "
                "'voxdim' attribute; spacing is undefined."
            ),
        )

    representative_step, is_approximate = get_representative_step(
        coord.values, uniformity_tolerance=uniformity_tolerance
    )

    if not is_approximate:
        return CoordinateSpacingInfo(
            value=representative_step, median=representative_step, warn_msg=None
        )
    return CoordinateSpacingInfo(
        value=None,
        median=representative_step,
        warn_msg=f"Coordinate '{dim}' has non-uniform sampling; spacing is undefined.",
    )


def get_coordinate_spacings(
    data: xr.DataArray, uniformity_tolerance: float = 1e-2
) -> dict[str, float | None]:
    """Compute coordinate spacing for all dimensions of a DataArray.

    For each dimension:

    - If the coordinate has two or more points and is uniformly sampled, returns the
      median step size.
    - If the coordinate has a single point, returns the `voxdim` coordinate attribute
      if present, otherwise `None` with a warning.
    - If the coordinate is missing or has non-uniform spacing, returns `None` with a
      warning.
    - If the coordinate doesn't have int or float dtype, returns `None` without a
      warning.

    Uniformity is assessed per-interval: each consecutive difference must satisfy
    `|diff - median| <= uniformity_tolerance * |median|`.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray whose coordinate spacing to compute.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed per-interval relative deviation from the median consecutive
        difference. Coordinates with any interval exceeding this threshold are
        considered non-uniform.

    Returns
    -------
    dict[str, float | None]
        Spacing per dimension in DataArray dimension order. `None` indicates that
        spacing is undefined for that dimension.
    """
    result: dict[str, float | None] = {}
    for dim in (str(d) for d in data.dims):
        r = get_coordinate_spacing_info(dim, data, uniformity_tolerance)
        if r.warn_msg is not None:
            warnings.warn(r.warn_msg, stacklevel=find_stack_level())
        result[dim] = r.value
    return result


def get_coordinate_spacings_best_effort(
    da: xr.DataArray, uniformity_tolerance: float = 1e-2
) -> tuple[dict[str, float], list[str]]:
    """Compute coordinate spacing, falling back to median diff for non-uniform dims.

    Like
    [`get_coordinate_spacings`][confusius._utils.coordinates.get_coordinate_spacings]
    but instead of returning `None` for non-uniform coordinates it returns the median
    consecutive difference as a best-effort approximation. This is appropriate when a
    single representative spacing is required (e.g. for napari's `scale` parameter) even
    though the coordinate is not perfectly uniform. No warnings are emitted; the caller
    is responsible for issuing context-appropriate messages for the dims listed in
    `non_uniform`.

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
        `1.0` only for dimensions with truly missing, non-numeric, or
        single-point-without-voxdim coordinates.
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
    """Return the physical origin (first coordinate value) for each dimension.

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


def sort_coords_into_increasing_order(
    data: xr.DataArray,
    dims: Sequence[Hashable],
) -> xr.DataArray:
    """Sort coordinate axes into increasing order.

    Importantly used in two cases throughout the package:

        - Any plotted coordinate axis that is not already monotonic increasing,
          including monotonic-decreasing axes, is sorted to avoid ambiguous geometry in
          plotting backends that assume ordered coordinates (e.g. `pcolormesh` edge
          construction, contour interpolation, and napari array indexing with
          scale/translate).

        - During resampling/registration steps, `spacing` in spatial dimensions is
          expected to be positive. Any monotonic-decreasing coordinate is flipped to
          ensure positive spacing.

    Parameters
    ----------
    data : xarray.DataArray
        Input DataArray whose plotted coordinate axes should be sorted.
    dims : sequence of hashable
        Dimensions whose coordinates to consider for sorting.

    Returns
    -------
    xarray.DataArray
        The input with every non-monotonic-increasing coordinate among `dims`
        sorted into ascending order.
    """
    sorted_data = data
    for dim in dims:
        if dim not in sorted_data.coords:
            continue
        if not sorted_data.get_index(dim).is_monotonic_increasing:
            sorted_data = sorted_data.sortby(dim)
    return sorted_data
