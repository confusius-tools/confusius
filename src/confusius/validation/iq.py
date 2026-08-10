"""IQ data validation utilities."""

import numpy as np
import xarray as xr

from confusius._dims import TIME_DIM, VOXEL_DIMS
from confusius.validation.fusi import ensure_fusi, validate_fusi

_REQUIRED_DIMS = (TIME_DIM, *VOXEL_DIMS)
"""Required dimensions and coordinates that all IQ data must have."""

_AXIAL_VELOCITY_REQUIRED_ATTRS = (
    "transmit_frequency",
    "beamforming_sound_velocity",
)
"""Required attributes for IQ data used in axial velocity computation."""


def ensure_iq(iq: xr.DataArray, require_velocity_attrs: bool = False) -> xr.DataArray:
    """Return `iq` as a canonical validated IQ DataArray.

    Parameters
    ----------
    iq : xarray.DataArray
        Input DataArray to canonicalize and validate as IQ data.
    require_velocity_attrs : bool, default: False
        Whether to validate that all attributes required for velocity estimation are
        present in the DataArray attributes.

    Returns
    -------
    xarray.DataArray
        Canonical IQ DataArray with dimensions `(time, k, j, i)`.

    Raises
    ------
    ValueError
        If `iq` is not valid canonical fUSI IQ data or required attributes are missing.
    TypeError
        If `iq` is not complex-valued.
    """
    iq = ensure_fusi(iq, require_time=True)
    iq = iq.transpose(*_REQUIRED_DIMS)
    validate_iq(iq, require_velocity_attrs=require_velocity_attrs)
    return iq


def validate_iq(iq: xr.DataArray, require_velocity_attrs: bool = False) -> None:
    """Validate that a DataArray contains valid IQ data.

    Parameters
    ----------
    iq : xarray.DataArray
        Input DataArray to validate. Must have dimensions `(time, k, j, i)`, CTI-backed
        physical `z/y/x` coordinates, and the required structure and attributes.
    require_velocity_attrs : bool, default: False
        Whether to validate that all attributes required for velocity estimation are
        present in the DataArray attributes.

    Raises
    ------
    ValueError
        If the DataArray has invalid dimensions, coordinates, or required attributes.
    TypeError
        If the IQ data is not complex-valued.
    """
    validate_fusi(
        iq,
        require_time=True,
        allow_pose=False,
        allow_extra_dims=True,
        require_canonical_dim_order=True,
    )

    if iq.dims != _REQUIRED_DIMS:
        raise ValueError(
            f"Expected dimensions {_REQUIRED_DIMS}, got {iq.dims}. "
            "Use .transpose() to reorder dimensions if needed."
        )

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise TypeError(
            f"Expected complex-valued data, got dtype {iq.dtype}. "
            "IQ data should be complex64 or complex128."
        )

    if require_velocity_attrs:
        missing_attrs = set(_AXIAL_VELOCITY_REQUIRED_ATTRS) - set(iq.attrs.keys())
        if missing_attrs:
            raise ValueError(
                f"Missing required DataArray attributes: {missing_attrs}. "
                "Axial velocity computation requires attributes: "
                f"{_AXIAL_VELOCITY_REQUIRED_ATTRS}."
            )


def validate_iq_dataarray(iq: xr.DataArray, require_attrs: bool = False) -> None:
    """Validate IQ data with the legacy `require_attrs` keyword.

    Parameters
    ----------
    iq : xarray.DataArray
        Input DataArray to validate.
    require_attrs : bool, default: False
        Whether to validate velocity metadata attributes.

    Raises
    ------
    ValueError
        If IQ validation fails.
    TypeError
        If `iq` is not complex-valued.
    """
    validate_iq(iq, require_velocity_attrs=require_attrs)
