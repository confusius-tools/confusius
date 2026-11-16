"""Shared helpers for confusius.iq.process and confusius.iq.clutter_filters."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.validation import ensure_voxeldata


def validate_filter_frequency(freq: float, nyquist: float) -> float:
    """Validate a digital-filter frequency against the Nyquist limit.

    Parameters
    ----------
    freq : float
        Filter frequency in hertz.
    nyquist : float
        Half the sampling frequency, in hertz.

    Returns
    -------
    float
        Validated input frequency.

    Raises
    ------
    ValueError
        If `freq` is not strictly between zero and `nyquist`.
    """
    if not 0 < freq < nyquist:
        raise ValueError(f"Frequency {freq} must be in range (0, {nyquist})")
    return freq


def get_iq_window_count(n_volumes: int, width: int, stride: int) -> int:
    """Count complete temporal windows after validating their sizes.

    Parameters
    ----------
    n_volumes : int
        Available input volumes.
    width : int
        Positive integral window width, not exceeding `n_volumes`.
    stride : int
        Positive integral window stride.

    Returns
    -------
    int
        Number of complete windows.

    Raises
    ------
    ValueError
        If window sizes are not positive integers or no complete window fits.
    """
    for name, value in (("window_width", width), ("window_stride", stride)):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(f"`{name}` must be a positive integer.")
    if width > n_volumes:
        raise ValueError(
            f"`window_width` ({width}) must not exceed the available volumes "
            f"({n_volumes}); at least one complete window is required."
        )
    return int((n_volumes - width) // stride + 1)


def get_clutter_filtered_dtype(
    dtype: npt.DTypeLike,
    filter_method: Literal[
        "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
    ],
    low_cutoff: float | None,
    high_cutoff: float | None,
    window_width: int,
    n_voxels: int,
) -> np.dtype:
    """Describe clutter-filter output precision without inspecting IQ samples.

    Parameters
    ----------
    dtype : numpy.typing.DTypeLike
        Input IQ dtype.
    filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"}
        Clutter filter family.
    low_cutoff, high_cutoff : int or float, optional
        Filter cutoffs. Missing bounds retain the corresponding full range.
    window_width : int
        Temporal filtering window size.
    n_voxels : int
        Number of spatial voxels used to estimate SVD components, after masking.

    Returns
    -------
    numpy.dtype
        Input dtype for statically identifiable no-op filters, otherwise the input
        dtype promoted to at least complex128.

    Raises
    ------
    ValueError
        If `filter_method` is unknown or an SVD mask selects no voxels.
    """
    if filter_method != "butterworth" and n_voxels < 1:
        raise ValueError("clutter_mask must select at least one voxel.")
    if filter_method == "svd_indices":
        max_components = min(window_width, n_voxels)
        no_op = low_cutoff in (None, 0) and high_cutoff in (None, max_components)
    elif filter_method in ("svd_energy", "svd_cumulative_energy"):
        no_op = low_cutoff in (None, 0) and high_cutoff in (None, np.inf)
    elif filter_method == "butterworth":
        no_op = low_cutoff is None and high_cutoff is None
    else:
        raise ValueError(f"Unknown clutter filter method: {filter_method}.")
    return np.dtype(dtype) if no_op else np.result_type(dtype, np.complex128)


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
        require_velocity_attrs=require_velocity_attrs,
        require_dtype=np.complexfloating,
    )
