"""Unit validation utilities."""

from collections.abc import Sequence

import xarray as xr


def validate_matching_spatial_units(arrays: Sequence[tuple[str, xr.DataArray]]) -> None:
    """Raise `ValueError` if spatial coordinate units disagree across DataArrays.

    Parameters
    ----------
    arrays : sequence of tuple[str, xarray.DataArray]
        Named DataArrays to compare. Spatial dimensions present in more than one array
        must carry matching `coord.attrs["units"]` values when that metadata is
        defined.

    Raises
    ------
    ValueError
        If any shared spatial dimension has conflicting `units` metadata.
    """
    spatial_dims = ("z", "y", "x")
    for dim in spatial_dims:
        seen: dict[str, str] = {}
        for name, array in arrays:
            if dim not in array.coords:
                continue
            units = array.coords[dim].attrs.get("units")
            if units is None:
                continue
            seen[name] = str(units)
        if len(set(seen.values())) > 1:
            mismatch = ", ".join(f"{name}={units!r}" for name, units in seen.items())
            raise ValueError(
                f"Spatial coordinate units for dimension {dim!r} must match across "
                f"inputs; got {mismatch}."
            )
