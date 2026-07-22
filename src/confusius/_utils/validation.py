"""Shared input-type checks used across ConfUSIus validators."""

import xarray as xr


def require_dataarray(data: object, name: str = "data") -> None:
    """Raise `TypeError` if `data` is not an `xarray.DataArray`.

    Parameters
    ----------
    data : object
        Candidate value.
    name : str, default: "data"
        Name used in the raised error message.

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"{name} must be an xarray.DataArray, got {type(data).__name__}."
        )
