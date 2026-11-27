"""Shared helpers for confusius.iq.process and confusius.iq.clutter_filters."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.validation import ensure_voxeldata


def validate_svd_index_cutoffs(
    low_cutoff: float | None, high_cutoff: float | None, max_components: int
) -> tuple[int, int]:
    """Validate static SVD cutoffs against the number of available components.

    Parameters
    ----------
    low_cutoff : int, optional
        Index of the first retained singular vector. If not provided, `0`.
    high_cutoff : int, optional
        Index after the last retained singular vector. If not provided,
        `max_components`.
    max_components : int
        Number of singular vectors, the smaller of the window width and the number
        of voxels used to estimate them.

    Returns
    -------
    low_cutoff : int
        Validated low cutoff.
    high_cutoff : int
        Validated high cutoff.

    Raises
    ------
    ValueError
        If a cutoff is not an integer, or the cutoffs do not satisfy
        `0 <= low_cutoff < high_cutoff <= max_components`.
    """
    if low_cutoff is None:
        low_cutoff = 0
    if high_cutoff is None:
        high_cutoff = max_components

    if not isinstance(low_cutoff, int) or not isinstance(high_cutoff, int):
        raise ValueError("Cutoffs must be integers")  # noqa: TRY004
    if not (0 <= low_cutoff < high_cutoff <= max_components):
        raise ValueError(
            f"Cutoffs must satisfy 0 <= low_cutoff ({low_cutoff}) < "
            f"high_cutoff ({high_cutoff}) <= {max_components}"
        )
    return low_cutoff, high_cutoff


def validate_energy_cutoffs(
    low_cutoff: float | None, high_cutoff: float | None
) -> tuple[float, float]:
    """Validate energy-based SVD cutoffs and fill in their defaults.

    Parameters
    ----------
    low_cutoff : int or float, optional
        Lower energy threshold. Components with energy below this value are treated as
        clutter. If not provided, `0.0`.
    high_cutoff : int or float, optional
        Upper energy threshold. Components with energy above this value are treated as
        clutter. If not provided, `numpy.inf`.

    Returns
    -------
    low_cutoff : float
        Validated lower cutoff value (non-negative).
    high_cutoff : float
        Validated upper cutoff value (greater than `low_cutoff`).

    Raises
    ------
    ValueError
        If a cutoff is negative, or `low_cutoff` is greater than or equal to
        `high_cutoff`.
    """
    if low_cutoff is not None and low_cutoff < 0:
        raise ValueError("Low cutoff must be non-negative.")
    if high_cutoff is not None and high_cutoff < 0:
        raise ValueError("High cutoff must be non-negative.")

    low_cutoff = low_cutoff if low_cutoff is not None else 0.0
    high_cutoff = high_cutoff if high_cutoff is not None else np.inf

    if low_cutoff >= high_cutoff:
        raise ValueError(
            f"Low cutoff ({low_cutoff}) must be lower than high cutoff ({high_cutoff})."
        )

    return low_cutoff, high_cutoff


def resolve_iq_window(
    n_volumes: int,
    width: int | None,
    stride: int | None,
    *,
    default_width: int | None = None,
    name_prefix: str = "",
    limit_name: str = "the available volumes",
) -> tuple[int, int, int]:
    """Resolve a temporal window and count its complete occurrences.

    Parameters
    ----------
    n_volumes : int
        Available input volumes.
    width : int, optional
        Positive integral window width, at most `n_volumes`.
    stride : int, optional
        Positive integral window stride. Defaults to the resolved width.
    default_width : int, optional
        Width used when `width` is omitted. Defaults to `n_volumes`.
    name_prefix : str, default: ""
        Prefix of the caller's `<prefix>window_width` and `<prefix>window_stride`
        parameters, so error messages name the arguments the caller received.
    limit_name : str, default: "the available volumes"
        What `n_volumes` counts, as worded in the error raised when `width` exceeds
        it.

    Returns
    -------
    width : int
        Window width as a Python integer.
    stride : int
        Window stride as a Python integer.
    count : int
        Number of complete windows.

    Raises
    ------
    ValueError
        If window sizes are not positive integers or no complete window fits.
    """
    if width is None:
        width = n_volumes if default_width is None else default_width
    if stride is None:
        stride = width
    width_name = f"{name_prefix}window_width"
    for name, value in ((width_name, width), (f"{name_prefix}window_stride", stride)):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(f"`{name}` must be a positive integer.")
    # Fixed-width NumPy scalars can overflow on recording-length arithmetic.
    width, stride = int(width), int(stride)
    if width > n_volumes:
        raise ValueError(
            f"`{width_name}` ({width}) must not exceed {limit_name} "
            f"({n_volumes}); at least one complete window is required."
        )
    return width, stride, (n_volumes - width) // stride + 1


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
        If `filter_method` is unknown, an SVD mask selects no voxels, or the SVD
        cutoffs are ones the filter would reject.
    """
    if filter_method != "butterworth" and n_voxels < 1:
        raise ValueError("clutter_mask must select at least one voxel.")
    if filter_method == "svd_indices":
        max_components = min(window_width, n_voxels)
        low_cutoff, high_cutoff = validate_svd_index_cutoffs(
            low_cutoff, high_cutoff, max_components
        )
        no_op = low_cutoff == 0 and high_cutoff == max_components
    elif filter_method in ("svd_energy", "svd_cumulative_energy"):
        low_cutoff, high_cutoff = validate_energy_cutoffs(low_cutoff, high_cutoff)
        no_op = low_cutoff == 0 and high_cutoff == np.inf
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
