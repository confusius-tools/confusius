"""Helpers shared between matplotlib- and napari-based plotting code."""

import warnings
from collections.abc import Hashable, Sequence

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


def sort_coords_for_plot(
    data: xr.DataArray,
    dims: Sequence[Hashable],
) -> xr.DataArray:
    """Sort coordinate axes into increasing order before plotting.

    Any plotted coordinate axis that is not already monotonic increasing,
    including monotonic-decreasing axes, is sorted to avoid ambiguous
    geometry in plotting backends that assume ordered coordinates (e.g.
    `pcolormesh` edge construction, contour interpolation, and napari array
    indexing with scale/translate).

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
