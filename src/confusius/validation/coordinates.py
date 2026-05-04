"""Coordinate validation utilities."""

from collections.abc import Iterable
from typing import Hashable

import numpy as np
import xarray as xr


def _coordinates_match(
    coord_a: xr.DataArray,
    coord_b: xr.DataArray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Return whether two coordinate arrays match.

    Parameters
    ----------
    coord_a : xarray.DataArray
        First coordinate array to compare.
    coord_b : xarray.DataArray
        Second coordinate array to compare.
    rtol : float, default: 1e-5
        Relative tolerance used for numeric coordinate comparison.
    atol : float, default: 1e-8
        Absolute tolerance used for numeric coordinate comparison.

    Returns
    -------
    bool
        `True` when coordinates have the same shape and matching values.
    """
    values_a = coord_a.values
    values_b = coord_b.values

    if values_a.shape != values_b.shape:
        return False

    if values_a.dtype == object or values_b.dtype == object:
        return bool(np.array_equal(values_a, values_b))

    if np.issubdtype(values_a.dtype, np.number) and np.issubdtype(
        values_b.dtype, np.number
    ):
        return bool(np.allclose(values_a, values_b, rtol=rtol, atol=atol))

    return bool(np.array_equal(values_a, values_b))


def validate_matching_coordinates(
    left: xr.DataArray,
    right: xr.DataArray,
    coord_names: Hashable | Iterable[Hashable] | None = None,
    *,
    left_name: str = "left array",
    right_name: str = "right array",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Validate that selected coordinates match between two DataArrays.

    Comparison is performed on coordinate values rather than the full coordinate
    `DataArray`, so unrelated attached coordinates do not cause false mismatches.
    Numeric coordinates are compared with tolerance to accommodate harmless
    floating-point drift (for example after serialization and reload). Non-numeric
    coordinates are compared exactly.

    Parameters
    ----------
    left : xarray.DataArray
        First array to compare.
    right : xarray.DataArray
        Second array to compare.
    coord_names : Hashable | Iterable[Hashable] | None, default: None
        Coordinate names to compare. If not specified, all shared dimension coordinates
        are checked.
    left_name : str, default: "left array"
        Label used for `left` in error messages. Override with a context-specific name
        (e.g. `"run 0"`, `"map 0"`) for more actionable errors.
    right_name : str, default: "right array"
        Label used for `right` in error messages.
    rtol : float, default: 1e-5
        Relative tolerance used for numeric coordinate comparison.
    atol : float, default: 1e-8
        Absolute tolerance used for numeric coordinate comparison.

    Raises
    ------
    ValueError
        If a requested coordinate is missing or if coordinates do not match.
    """
    if coord_names is None:
        names = [dim for dim in left.dims if dim in left.coords and dim in right.coords]
    elif not isinstance(coord_names, Iterable) or isinstance(coord_names, str):
        names = [coord_names]
    else:
        names = list(coord_names)

    for name in names:
        if name not in left.coords:
            raise ValueError(f"Coordinate '{name}' is missing from {left_name}.")
        if name not in right.coords:
            raise ValueError(f"Coordinate '{name}' is missing from {right_name}.")
        left_coord = left.coords[name]
        right_coord = right.coords[name]
        if not _coordinates_match(left_coord, right_coord, rtol=rtol, atol=atol):
            is_numeric = np.issubdtype(left_coord.dtype, np.number) and np.issubdtype(
                right_coord.dtype, np.number
            )
            comparison = (
                f"within rtol={rtol}, atol={atol}"
                if is_numeric
                else "with exact equality"
            )
            raise ValueError(
                f"Coordinate '{name}' does not match between {left_name} and "
                f"{right_name} ({comparison}).\n"
                f"  {left_name}: {left_coord.values}\n"
                f"  {right_name}: {right_coord.values}"
            )
