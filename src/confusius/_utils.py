"""Package-level utilities for confusius."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr


_TIME_UNIT_TO_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "usec": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
}


def find_stack_level() -> int:
    """Find the first place in the stack that is not inside confusius.

    Adapted from
    [pandas](https://github.com/pandas-dev/pandas/tree/main/pandas/util/_exceptions.py#L37)
    and
    [Nilearn](https://github.com/nilearn/nilearn/blob/2d1a2c6d901ef4aba2737ed84e08ad1956afd123/nilearn/_utils/logger.py#L150).

    Returns
    -------
    int
        Stack level pointing to the first frame outside the confusius package.
    """
    import confusius

    pkg_dir = Path(confusius.__file__).parent

    frame = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if not filename.startswith(str(pkg_dir)):
                break
            frame = frame.f_back
            n += 1
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


def _time_coord_values_in_seconds(
    data: xr.DataArray, dim: str = "time"
) -> npt.NDArray[np.float64]:
    """Return a time coordinate converted to seconds.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the time coordinate.
    dim : str, default: "time"
        Name of the coordinate dimension to convert.

    Returns
    -------
    numpy.ndarray
        Time coordinate values converted to seconds.

    Warns
    -----
    UserWarning
        If the coordinate has missing or unknown units. In that case, seconds are
        assumed.
    """
    values = np.asarray(data.coords[dim].values, dtype=np.float64)
    units = data.coords[dim].attrs.get("units")
    if units is None:
        warnings.warn(
            "Time coordinate has no `units` attribute. Assuming seconds.",
            stacklevel=find_stack_level(),
        )
        return values

    unit = str(units).lower()
    if unit not in _TIME_UNIT_TO_SECONDS:
        warnings.warn(
            f"Time coordinate uses unknown units {units!r}. Assuming seconds.",
            stacklevel=find_stack_level(),
        )
        return values

    return values * _TIME_UNIT_TO_SECONDS[unit]


def _representative_step(
    values: npt.NDArray[np.float64], uniformity_tolerance: float = 1e-5
) -> tuple[float | None, bool]:
    """Return a representative step size for a 1D coordinate array.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional coordinate values.
    uniformity_tolerance : float, default: 1e-5
        Maximum allowed relative range `(max_diff - min_diff) / median_diff` for the
        coordinate to be considered uniform.

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
        is_uniform = (np.max(diffs) - np.min(diffs)) / abs(
            median
        ) <= uniformity_tolerance

    if is_uniform:
        return median, False
    return median, True


def _representative_time_step_seconds(
    data: xr.DataArray, dim: str = "time", uniformity_tolerance: float = 1e-5
) -> tuple[float | None, bool]:
    """Return a representative time step in seconds.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the time coordinate.
    dim : str, default: "time"
        Name of the time coordinate dimension.
    uniformity_tolerance : float, default: 1e-5
        Maximum allowed relative range `(max_diff - min_diff) / median_diff` for the
        time coordinate to be considered uniform.

    Returns
    -------
    step : float or None
        Representative time step in seconds, or `None` if fewer than two time points
        are available.
    approximate : bool
        Whether the returned step is a median approximation derived from non-uniform
        sampling.
    """
    return _representative_step(
        _time_coord_values_in_seconds(data, dim),
        uniformity_tolerance=uniformity_tolerance,
    )


class _DimSpacing:
    """Result of per-dimension spacing analysis.

    Attributes
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


def _spacing_for_dim(
    dim: str,
    data: xr.DataArray,
    uniformity_tolerance: float,
) -> _DimSpacing:
    """Compute spacing information for a single dimension.

    Shared implementation used by
    [`_compute_spacing`][confusius._utils._compute_spacing] and
    [`_compute_spacing_best_effort`][confusius._utils._compute_spacing_best_effort] to
    avoid duplicating the uniformity check and median computation.

    Parameters
    ----------
    dim : str
        Dimension name.
    data : xarray.DataArray
        DataArray whose coordinate to inspect.
    uniformity_tolerance : float
        Maximum allowed relative range `(max_diff - min_diff) / median_diff`.

    Returns
    -------
    _DimSpacing
        Spacing result for the dimension.
    """
    if dim not in data.coords:
        return _DimSpacing(
            value=None,
            median=None,
            warn_msg=f"Dimension '{dim}' has no coordinate; spacing is undefined.",
        )

    coord = data.coords[dim]
    if not np.issubdtype(coord.dtype, int) and not np.issubdtype(coord.dtype, float):
        return _DimSpacing(value=None, median=None, warn_msg=None)

    if len(coord) < 2:
        if "voxdim" in coord.attrs:
            return _DimSpacing(
                value=float(coord.attrs["voxdim"]), median=None, warn_msg=None
            )
        return _DimSpacing(
            value=None,
            median=None,
            warn_msg=(
                f"Dimension '{dim}' has a single coordinate point and no "
                "'voxdim' attribute; spacing is undefined."
            ),
        )

    representative_step, is_approximate = _representative_step(
        np.asarray(coord.values, dtype=np.float64),
        uniformity_tolerance=uniformity_tolerance,
    )

    if not is_approximate:
        return _DimSpacing(
            value=representative_step, median=representative_step, warn_msg=None
        )
    return _DimSpacing(
        value=None,
        median=representative_step,
        warn_msg=f"Coordinate '{dim}' has non-uniform sampling; spacing is undefined.",
    )


def _compute_spacing(
    data: xr.DataArray,
    uniformity_tolerance: float = 1e-2,
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

    Uniformity is assessed as `(max_diff - min_diff) / median_diff`, i.e. the
    relative range of consecutive differences.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray whose coordinate spacing to compute.
    uniformity_tolerance : float, default: 1e-2
        Maximum allowed relative range of consecutive differences, defined as
        `(max_diff - min_diff) / median_diff`. Coordinates whose relative range
        exceeds this threshold are considered non-uniform.

    Returns
    -------
    dict[str, float | None]
        Spacing per dimension in DataArray dimension order. `None` indicates that
        spacing is undefined for that dimension.
    """
    result: dict[str, float | None] = {}
    for dim in (str(d) for d in data.dims):
        r = _spacing_for_dim(dim, data, uniformity_tolerance)
        if r.warn_msg is not None:
            warnings.warn(r.warn_msg, stacklevel=find_stack_level())
        result[dim] = r.value
    return result


def _compute_spacing_best_effort(
    da: xr.DataArray,
    uniformity_tolerance: float = 1e-2,
) -> tuple[dict[str, float], list[str]]:
    """Compute coordinate spacing, falling back to median diff for non-uniform dims.

    Like [`_compute_spacing`][confusius._utils._compute_spacing] but instead of
    returning `None` for non-uniform coordinates it returns the median consecutive
    difference as a best-effort approximation. This is appropriate when a single
    representative spacing is required (e.g. for napari's `scale` parameter) even
    though the coordinate is not perfectly uniform. No warnings are emitted; the caller
    is responsible for issuing context-appropriate messages for the dims listed in
    `non_uniform`.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray whose coordinate spacing to compute.
    uniformity_tolerance : float, default: 1e-2
        Passed through to the uniformity check (see
        [`_compute_spacing`][confusius._utils._compute_spacing]).

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
        r = _spacing_for_dim(dim, da, uniformity_tolerance)
        if r.value is not None:
            spacing[dim] = r.value
        elif r.median is not None:
            spacing[dim] = r.median
            non_uniform.append(dim)
        else:
            spacing[dim] = 1.0
    return spacing, non_uniform


def _compute_origin(
    data: xr.DataArray,
) -> dict[str, float]:
    """Compute the physical origin (first coordinate value) for each dimension.

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


def _one_level_deeper() -> int:
    """Call `find_stack_level` one level deeper.

    Used in tests for `find_stack_level`. Must be defined in a ConfUSIus module (not a
    test file) so the stack level counter increments past this frame.

    Returns
    -------
    int
        Result of `find_stack_level` called from within ConfUSIus.
    """
    return find_stack_level()
