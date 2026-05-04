"""Xarray accessor for fUSI-specific operations."""

from pathlib import Path
from typing import Any

import xarray as xr

from confusius._utils import get_coordinate_origins, get_coordinate_spacings
from confusius.xarray.affine import FUSIAffineAccessor
from confusius.xarray.connectivity import FUSIConnectivityAccessor
from confusius.xarray.extract import FUSIExtractAccessor
from confusius.xarray.iq import FUSIIQAccessor
from confusius.xarray.plotting import FUSIPlotAccessor
from confusius.xarray.registration import FUSIRegistrationAccessor
from confusius.xarray.scale import FUSIScaleAccessor


@xr.register_dataarray_accessor("fusi")
class FUSIAccessor:
    """Xarray accessor for fUSI-specific operations.

    Provides convenient methods for functional ultrasound imaging data analysis.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The `DataArray` to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius import xarray as cxr  # Registers the accessor
    >>> data = xr.DataArray([1, 10, 100, 1000])
    >>> data.fusi.scale.db(factor=20)
    <xarray.DataArray (dim_0: 4)>
    array([-60., -40., -20.,   0.])
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def connectivity(self) -> FUSIConnectivityAccessor:
        """Access connectivity analysis operations.

        Returns
        -------
        FUSIConnectivityAccessor
            Accessor for seed-based functional connectivity maps.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.open_zarr("recording.zarr")["power_doppler"]
        >>> seed_masks = xr.open_zarr("seed_masks.zarr")["masks"]
        >>> mapper = data.fusi.connectivity.seed_map(seed_masks=seed_masks)
        """
        return FUSIConnectivityAccessor(self._obj)

    @property
    def scale(self) -> FUSIScaleAccessor:
        """Access scaling operations.

        Returns
        -------
        FUSIScaleAccessor
            Accessor for scaling transformations.

        Examples
        --------
        >>> data = xr.DataArray([1, 10, 100, 1000])
        >>> data.fusi.scale.db(factor=10)
        <xarray.DataArray (dim_0: 4)>
        array([-30., -20., -10.,   0.])
        """
        return FUSIScaleAccessor(self._obj)

    @property
    def plot(self) -> FUSIPlotAccessor:
        """Access plotting operations.

        Returns
        -------
        FUSIPlotAccessor
            Accessor for plotting methods.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> viewer, layer = data.fusi.plot.napari()
        """
        return FUSIPlotAccessor(self._obj)

    @property
    def register(self) -> FUSIRegistrationAccessor:
        """Access registration operations.

        Returns
        -------
        FUSIRegistrationAccessor
            Accessor for registration methods.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["power_doppler"]
        >>> registered = data.fusi.register.volumewise(reference_time=0)
        """
        return FUSIRegistrationAccessor(self._obj)

    @property
    def iq(self) -> FUSIIQAccessor:
        """Access IQ processing operations.

        Returns
        -------
        FUSIIQAccessor
            Accessor for IQ processing methods.

        Examples
        --------
        >>> import xarray as xr
        >>> ds = xr.open_zarr("output.zarr")
        >>> iq = ds["iq"]
        >>> pwd = iq.fusi.iq.process_to_power_doppler()
        >>> velocity = iq.fusi.iq.process_to_axial_velocity()
        """
        return FUSIIQAccessor(self._obj)

    @property
    def extract(self) -> FUSIExtractAccessor:
        """Access signal extraction operations.

        Returns
        -------
        FUSIExtractAccessor
            Accessor for extracting signals from fUSI data and reconstructing fUSI data
            from processed signals.

        Examples
        --------
        >>> import xarray as xr
        >>> data = xr.open_zarr("output.zarr")["power_doppler"]
        >>> mask = xr.open_zarr("brain_mask.zarr")["mask"]
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> # ... process signals ...
        >>> restored = signals.fusi.extract.unmask(mask)
        """
        return FUSIExtractAccessor(self._obj)

    @property
    def spacing(self) -> dict[str, float | None]:
        """Coordinate spacing for all dimensions.

        Spacing is computed as the median of consecutive coordinate differences.
        A coordinate is considered uniform if every interval is within 1% of the
        median interval (per-interval `|diff - median| <= 0.01 * |median|`).

        Returns
        -------
        dict[str, float | None]
            Spacing per dimension, in DataArray dimension order. Returns `None` for
            dimensions with non-uniform or undefined spacing, with a warning.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((10, 20)),
        ...     dims=["y", "x"],
        ...     coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        ... )
        >>> data.fusi.spacing
        {'y': 0.2, 'x': 0.1}
        """
        return get_coordinate_spacings(self._obj)

    @property
    def origin(self) -> dict[str, float]:
        """Physical origin (first coordinate value) for all dimensions.

        For each dimension, returns the first coordinate value. If a coordinate is
        missing, warns and falls back to `0.0`.

        Returns
        -------
        dict[str, float]
            Origin per dimension, in DataArray dimension order.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((10, 20)),
        ...     dims=["y", "x"],
        ...     coords={"y": np.arange(10) * 0.2, "x": np.arange(20) * 0.1},
        ... )
        >>> data.fusi.origin
        {'y': 0.0, 'x': 0.0}
        """
        return get_coordinate_origins(self._obj)

    @property
    def affine(self) -> FUSIAffineAccessor:
        """Access affine transform operations.

        Returns
        -------
        FUSIAffineAccessor
            Accessor for computing relative transforms between scans and for
            applying axis-aligned affines to spatial coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import confusius  # noqa: F401
        >>> eye = np.eye(4)
        >>> a = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"to_world": eye}})
        >>> b = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"to_world": eye}})
        >>> np.allclose(a.fusi.affine.to(b, via="to_world"), np.eye(4))
        True
        """
        return FUSIAffineAccessor(self._obj)

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Save the DataArray to file, dispatching by extension.

        Supported formats:

        - **NIfTI** (`.nii`, `.nii.gz`): saved via
          [`save_nifti`][confusius.io.save_nifti].
        - **Zarr** (`.zarr`): saved via
          [`xarray.DataArray.to_zarr`][xarray.DataArray.to_zarr].

        Parameters
        ----------
        path : str or pathlib.Path
            Output path. The extension determines the format.
        **kwargs
            Additional keyword arguments forwarded to the underlying saver.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((10, 32, 1, 64)), dims=["time", "z", "y", "x"]
        ... )
        >>> data.fusi.save("recording.nii.gz")
        """
        from confusius.io.loadsave import save

        save(self._obj, path, **kwargs)
