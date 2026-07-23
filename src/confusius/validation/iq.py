"""IQ data validation utilities."""

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS, TIME_DIM
from confusius.validation.fusi import validate_fusi_dataarray

_REQUIRED_DIMS = (TIME_DIM, *SPATIAL_DIMS)
"""Required dimensions and coordinates that all IQ data must have."""

_AXIAL_VELOCITY_REQUIRED_ATTRS = (
    "transmit_frequency",
    "beamforming_sound_velocity",
)
"""Required attributes for IQ data used in axial velocity computation."""


def validate_iq_dataarray(iq: xr.DataArray, require_attrs: bool = False) -> None:
    """Validate that a DataArray contains valid IQ data.

    This function performs validation of an IQ DataArray to ensure it meets all
    requirements for processing with ConfUSIus functions. Validation checks include:

    1. **Dimensions**: The IQ DataArray must have exactly 4 dimensions in the
       order: `(time, z, y, x)`.
    2. **Coordinates**: All dimensions must have corresponding coordinates.
    3. **Data type**: The data must be complex-valued (`complex64` or `complex128`).
    4. **Attributes** (optional): If `require_attrs` is `True`, the DataArray must have
       the following attributes needed for axial velocity computation:

       - `transmit_frequency`: Ultrasound probe central frequency in Hz.
       - `beamforming_sound_velocity`: Speed of sound assumed during beamforming in
         meters per second.

    Parameters
    ----------
    iq : xarray.DataArray
        Input DataArray to validate. Must have dimensions `(time, z, y, x)` and the
        required structure and attributes.
    require_attrs : bool, default: False
        Whether to validate that all required attributes (`transmit_frequency`,
        `beamforming_sound_velocity`) are present in the DataArray attributes.

    Raises
    ------
    ValueError
        If the DataArray does not have dimensions `(time, z, y, x)`, if required
        coordinates are missing, or if required attributes are missing when
        `require_attrs=True`.
    TypeError
        If the IQ data is not complex-valued.

    Examples
    --------
    Validate a properly formatted IQ DataArray:

    >>> import numpy as np
    >>> import xarray as xr
    >>> iq = xr.DataArray(
    ...     np.ones((10, 4, 6, 8), dtype=np.complex64),
    ...     dims=("time", "z", "y", "x"),
    ...     coords={
    ...         "time": np.arange(10),
    ...         "z": np.arange(4),
    ...         "y": np.arange(6),
    ...         "x": np.arange(8),
    ...     },
    ...     attrs={
    ...         "transmit_frequency": 15e6,
    ...         "beamforming_sound_velocity": 1540.0,
    ...     },
    ... )
    >>> validate_iq_dataarray(iq, require_attrs=True)

    Skip attribute validation for intermediate processing:

    >>> iq_no_attrs = xr.DataArray(
    ...     np.ones((10, 4, 6, 8), dtype=np.complex64),
    ...     dims=("time", "z", "y", "x"),
    ...     coords={
    ...         "time": np.arange(10),
    ...         "z": np.arange(4),
    ...         "y": np.arange(6),
    ...         "x": np.arange(8),
    ...     },
    ... )
    >>> validate_iq_dataarray(iq_no_attrs, require_attrs=False)
    """
    validate_fusi_dataarray(
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

    if require_attrs:
        missing_attrs = set(_AXIAL_VELOCITY_REQUIRED_ATTRS) - set(iq.attrs.keys())
        if missing_attrs:
            raise ValueError(
                f"Missing required DataArray attributes: {missing_attrs}. "
                "Axial velocity computation requires attributes: "
                f"{_AXIAL_VELOCITY_REQUIRED_ATTRS}."
            )
