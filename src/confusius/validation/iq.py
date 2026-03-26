"""IQ data validation utilities."""

import numpy as np
import xarray as xr

from confusius._dims import SPATIAL_DIMS, TIME_DIM

_REQUIRED_DIMS = (TIME_DIM, *SPATIAL_DIMS)
"""Required dimensions and coordinates that all IQ data must have."""

_REQUIRED_ATTRS = (
    "compound_sampling_frequency",
    "transmit_frequency",
    "beamforming_sound_velocity",
)
"""Required attributes that all IQ data must have."""


def validate_iq(iq: xr.DataArray, require_attrs: bool = True) -> None:
    """Validate that a DataArray contains valid IQ data.

    This function performs validation of an IQ DataArray to ensure it meets all
    requirements for processing with confusius functions. Validation checks include:

    1. **Dimensions**: The IQ DataArray must have exactly 4 dimensions in the
       order: `(time, z, y, x)`.
    2. **Coordinates**: All dimensions must have corresponding coordinates.
    3. **Data type**: The data must be complex-valued (`complex64` or `complex128`).
    4. **Attributes** (optional): If `require_attrs` is `True`, the DataArray must have
       the following attributes:

       - `compound_sampling_frequency`: Volume acquisition rate in Hz.
       - `transmit_frequency`: Ultrasound probe central frequency in Hz.
       - `beamforming_sound_velocity`: Speed of sound assumed during beamforming in
         meters per second.

    Parameters
    ----------
    iq : xarray.DataArray
        Input DataArray to validate. Must have dimensions `(time, z, y, x)` and
        the required structure and attributes.
    require_attrs : bool, default: True
        Whether to validate that all required attributes
        (`compound_sampling_frequency`, `transmit_frequency`,
        `beamforming_sound_velocity`) are present in the DataArray attributes.

    Raises
    ------
    ValueError
        If the DataArray does not have dimensions ("time", "z", "y", "x") or
        corresponding coordinates, or if required attributes are missing (when
        `require_attrs=True`).

    TypeError
        If the IQ data is not complex-valued.

    Examples
    --------
    Validate a properly formatted IQ DataArray:

    >>> import xarray as xr
    >>> import numpy as np
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
    ...         "compound_sampling_frequency": 1000.0,
    ...         "transmit_frequency": 15e6,
    ...         "beamforming_sound_velocity": 1540.0,
    ...     },
    ... )
    >>> validate_iq(iq)

    Skip attribute validation for intermediate processing:

    >>> # DataArray missing attributes
    >>> iq_no_attrs = xr.DataArray(
    ...     np.ones((10, 4, 6, 8), dtype=np.complex64),
    ...     dims=("time", "z", "y", "x"),
    ...     coords={"time": np.arange(10), "z": np.arange(4),
    ...             "y": np.arange(6), "x": np.arange(8)},
    ... )
    >>> validate_iq(iq_no_attrs, require_attrs=False)
    """
    if iq.dims != _REQUIRED_DIMS:
        raise ValueError(
            f"Expected dimensions {_REQUIRED_DIMS}, got {iq.dims}. "
            "Use .transpose() to reorder dimensions if needed."
        )

    missing_coords = set(_REQUIRED_DIMS) - set(iq.coords.keys())
    if missing_coords:
        raise ValueError(f"Missing required coordinates: {missing_coords}.")

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise TypeError(
            f"Expected complex-valued data, got dtype {iq.dtype}. "
            "IQ data should be complex64 or complex128."
        )

    if require_attrs:
        missing_attrs = set(_REQUIRED_ATTRS) - set(iq.attrs.keys())
        if missing_attrs:
            raise ValueError(
                f"Missing required DataArray attributes: {missing_attrs}. "
                f"All IQ data must have attributes: {_REQUIRED_ATTRS}."
            )
