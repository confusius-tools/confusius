"""Shared helpers for confusius.iq.process and confusius.iq.clutter_filters."""

import numpy as np
import xarray as xr

from confusius.validation import ensure_voxeldata


def ensure_iq_voxeldata(
    iq: xr.DataArray, *, require_velocity_attrs: bool = False
) -> xr.DataArray:
    """Canonicalize and validate `iq` as a complex-valued, time-resolved VoxelData array.

    Parameters
    ----------
    iq : xarray.DataArray
        Input DataArray to canonicalize and validate as IQ data.
    require_velocity_attrs : bool, default: False
        Whether to require the `transmit_frequency`/`beamforming_sound_velocity`
        attributes needed for velocity estimation.

    Returns
    -------
    xarray.DataArray
        Canonicalized `(time, k, j, i)` VoxelData array.

    Raises
    ------
    ValueError
        If `iq` is not valid canonical VoxelData, or has fewer than two timepoints.
    TypeError
        If `iq` is not complex-valued.
    """
    return ensure_voxeldata(
        iq,
        require_time=True,
        allow_pose=False,
        allow_extra_dims=False,
        require_canonical_dim_order=True,
        require_velocity_attrs=require_velocity_attrs,
        require_dtype=np.complexfloating,
    )
