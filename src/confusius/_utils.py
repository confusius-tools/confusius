"""Package-level utilities for confusius."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import xarray as xr


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
        if dim not in data.coords:
            warnings.warn(
                f"Dimension '{dim}' has no coordinate; spacing is undefined.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result[dim] = None
            continue

        coord = data.coords[dim]
        if not np.issubdtype(coord.dtype, int) and not np.issubdtype(
            coord.dtype, float
        ):
            result[dim] = None
            continue

        if len(coord) < 2:
            if "voxdim" in coord.attrs:
                result[dim] = float(coord.attrs["voxdim"])
            else:
                warnings.warn(
                    f"Dimension '{dim}' has a single coordinate point and no "
                    "'voxdim' attribute; spacing is undefined.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                result[dim] = None
            continue

        diffs = np.diff(coord.values)
        median_diff = np.median(diffs)
        if (np.max(diffs) - np.min(diffs)) / median_diff > uniformity_tolerance:
            warnings.warn(
                f"Coordinate '{dim}' has non-uniform sampling; spacing is undefined.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result[dim] = None
        else:
            result[dim] = float(median_diff)
    return result


def _compute_origin(
    data: xr.DataArray,
) -> dict[str, float]:
    """Compute the physical origin (first coordinate value) for each dimension.

    For each dimension, returns the first coordinate value. If a coordinate is missing,
    warns and falls back to `0.0`.

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
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result[dim] = 0.0
        else:
            result[dim] = float(data.coords[dim].values.flat[0])
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
