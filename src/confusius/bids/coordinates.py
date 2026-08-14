"""Coordinate handling for fUSI-BIDS timing metadata.

This module provides utilities for converting between fUSI-BIDS timing metadata and
ConfUSIus coordinate structures.
"""

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import VOXEL_DIMS

_SLICE_ENCODING_DIRECTIONS: Final = tuple(
    f"{dim}{suffix}" for dim in reversed(VOXEL_DIMS) for suffix in ("", "-")
)
"""fUSI-BIDS `SliceEncodingDirection` values accepted by ConfUSIus."""


def create_slice_time_coordinate_from_bids(
    volume_times: npt.ArrayLike,
    slice_timing: npt.ArrayLike,
    slice_encoding_direction: Literal["i", "j", "k", "i-", "j-", "k-"],
    units: str = "s",
) -> xr.DataArray:
    """Create a `slice_time` coordinate from fUSI-BIDS `SliceTiming` metadata.

    The `slice_time` coordinate stores the absolute acquisition time of each slice for
    each volume.

    Parameters
    ----------
    volume_times : (n_time,) array-like
        Absolute onset time of each volume, in the same units as `slice_timing`.
    slice_timing : (n_slices,) array-like
        Array of slice acquisition times relative to the onset of each volume.
    slice_encoding_direction : {"i", "j", "k", "i-", "j-", "k-"}
        Direction of slice acquisition: `"i"` → `i`, `"j"` → `j`, `"k"` → `k`. A
        trailing `-` indicates that `slice_timing` is defined in reverse order (the
        first entry corresponds to the slice with the largest index). The values are
        reversed internally so the stored coordinate is always aligned with natural
        slice index order.
    units : str
        Units of the slice timing values.

    Returns
    -------
    (n_time, n_slices) xarray.DataArray
        Absolute slice time coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> slice_timing = np.array([0.0, 0.1, 0.2, 0.3])  # 4 slices
    >>> volume_times = np.array([0.0, 1.0])
    >>> coord = create_slice_time_coordinate_from_bids(
    ...     volume_times, slice_timing, slice_encoding_direction="k"
    ... )
    >>> coord.dims
    ('time', 'k')

    >>> coord.shape
    (2, 4)
    """
    if slice_encoding_direction not in _SLICE_ENCODING_DIRECTIONS:
        raise ValueError(
            f"Invalid SliceEncodingDirection: {slice_encoding_direction}. "
            f"Must be one of: {list(_SLICE_ENCODING_DIRECTIONS)}"
        )

    dim_name = slice_encoding_direction.removesuffix("-")
    slice_timing_arr = np.asarray(slice_timing)
    volume_times_arr = np.asarray(volume_times)

    if slice_timing_arr.ndim != 1:
        raise ValueError(f"SliceTiming must be 1D, got shape {slice_timing_arr.shape}")
    if volume_times_arr.ndim != 1:
        raise ValueError(f"volume_times must be 1D, got shape {volume_times_arr.shape}")

    if slice_encoding_direction.endswith("-"):
        slice_timing_arr = slice_timing_arr[::-1]

    return xr.DataArray(
        volume_times_arr[:, np.newaxis] + slice_timing_arr[np.newaxis, :],
        dims=["time", dim_name],
        attrs={"units": units},
    )


def create_bids_slice_timing_from_coordinate(
    slice_time_coord: xr.DataArray, volume_times: npt.ArrayLike
) -> tuple[np.ndarray, str]:
    """Extract fUSI-BIDS `SliceTiming` metadata from a `slice_time` coordinate.

    Parameters
    ----------
    slice_time_coord : (time, slice_encoding_direction) xarray.DataArray
        Absolute slice time coordinate.
    volume_times : (n_time,) array-like
        Absolute onset time of each volume, in the same units as `slice_time_coord`.

    Returns
    -------
    slice_timing : numpy.ndarray
        The slice acquisition times for one volume.
    slice_encoding_direction : {"i", "j", "k"}
        The direction of slice acquisition.

    Raises
    ------
    ValueError
        If the coordinate is not 2D, has an invalid spatial dimension, or if the slice
        timing varies across volumes after converting to onset-relative offsets.

    Examples
    --------
    >>> import numpy as np
    >>> volume_times = np.array([0.0, 1.0])
    >>> slice_timing = np.array([0.0, 0.1, 0.2, 0.3])
    >>> coord = create_slice_time_coordinate_from_bids(
    ...     volume_times, slice_timing, slice_encoding_direction="k"
    ... )
    >>> slice_timing, direction = create_bids_slice_timing_from_coordinate(
    ...     coord, volume_times
    ... )
    >>> slice_timing
    array([0. , 0.1, 0.2, 0.3])
    >>> direction
    'k'
    """
    if len(slice_time_coord.dims) != 2:
        raise ValueError(
            f"slice_time coordinate must be 2D, got {len(slice_time_coord.dims)}D: "
            f"{slice_time_coord.dims}"
        )

    if "time" not in slice_time_coord.dims:
        raise ValueError(
            f"slice_time coordinate must include a 'time' dimension, got: {slice_time_coord.dims}"
        )

    spatial_dim = next(iter(set(slice_time_coord.dims) - {"time"}))
    if spatial_dim not in VOXEL_DIMS:
        raise ValueError(
            f"slice_time coordinate must have one of spatial dimensions "
            f"{list(reversed(VOXEL_DIMS))}, got: {slice_time_coord.dims}"
        )

    volume_times_arr = np.asarray(volume_times)
    if volume_times_arr.ndim != 1:
        raise ValueError(f"volume_times must be 1D, got shape {volume_times_arr.shape}")
    if volume_times_arr.shape[0] != slice_time_coord.sizes["time"]:
        raise ValueError(
            "volume_times length must match the time dimension of slice_time coordinate."
        )

    slice_time_values = np.asarray(
        slice_time_coord.transpose("time", spatial_dim).values,
        dtype=np.float64,
    )
    relative_slice_timing = slice_time_values - volume_times_arr[:, np.newaxis]
    if not np.allclose(
        relative_slice_timing, relative_slice_timing[[0]], equal_nan=True
    ):
        raise ValueError(
            "slice_time coordinate varies across time points after converting to "
            "volume-onset-relative offsets."
        )

    return relative_slice_timing[0], str(spatial_dim)
