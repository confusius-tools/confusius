"""Unit validation utilities."""

from collections.abc import Sequence

import xarray as xr

from confusius._utils.geometry import get_voxel_to_world_units


def validate_matching_spatial_units(arrays: Sequence[tuple[str, xr.DataArray]]) -> None:
    """Raise `ValueError` if world-space units disagree across DataArrays.

    Parameters
    ----------
    arrays : sequence of tuple[str, xarray.DataArray]
        Named DataArrays to compare. Each must carry voxel-to-world geometry.

    Raises
    ------
    ValueError
        If any input lacks voxel-to-world geometry, or if any two inputs disagree
        on their `.fusi.affine.units`.
    """
    seen = {name: get_voxel_to_world_units(array) for name, array in arrays}
    if len(set(seen.values())) > 1:
        mismatch = ", ".join(f"{name}={units!r}" for name, units in seen.items())
        raise ValueError(f"World-space units must match across inputs; got {mismatch}.")
