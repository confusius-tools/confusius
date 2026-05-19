"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings

import numpy as np
import xarray as xr

from confusius._utils.stack import find_stack_level


def coerce_complex_to_magnitude(data: xr.DataArray, caller: str) -> xr.DataArray:
    """Convert complex-valued arrays to magnitude for plotting.

    Parameters
    ----------
    data : xarray.DataArray
        Input data to display.
    caller : str
        Name of the plotting entry point used in the warning message.

    Returns
    -------
    xarray.DataArray
        `data` unchanged for non-complex inputs, otherwise `abs(data)`.

    Warns
    -----
    UserWarning
        Raised when `data` is complex-valued to make the implicit magnitude
        conversion explicit to users.
    """
    if np.iscomplexobj(data):
        warnings.warn(
            f"Complex-valued data passed to {caller}; plotting magnitude "
            "(`abs(data)`).",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        return xr.ufuncs.abs(data)
    return data
