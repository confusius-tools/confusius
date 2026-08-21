"""Xarray accessor for fUSI-specific operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
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
        from confusius.xarray.connectivity import FUSIConnectivityAccessor

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
        from confusius.xarray.scale import FUSIScaleAccessor

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
        from confusius.xarray.plotting import FUSIPlotAccessor

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
        from confusius.xarray.registration import FUSIRegistrationAccessor

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
        from confusius.xarray.iq import FUSIIQAccessor

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
        from confusius.xarray.extract import FUSIExtractAccessor

        return FUSIExtractAccessor(self._obj)

    @property
    def spacing(self) -> dict[str, float | None]:
        """Coordinate spacing for all dimensions.

        Spacing is reported in DataArray dimension order. For native voxel dimensions
        `k/j/i`, each voxel-space dimension receives its world step length derived
        from the voxel-to-world affine column norm and the 1D voxel-coordinate
        step. A coordinate is considered uniform if every interval is within 1% of the
        median interval (per-interval `|diff - median| <= 0.01 * |median|`). `time`
        has no such fallback: `volume_acquisition_duration` is the time to acquire one
        volume, not the step between volumes, so a singleton or non-uniform `time`
        reports `None` here like any other dimension without a defined step.

        Returns
        -------
        dict[str, float | None]
            Spacing per dimension. Returns `None` for dimensions with non-uniform or
            undefined spacing, with a warning.

        Raises
        ------
        ValueError
            If `self` does not carry a voxel-to-world index.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((3, 10, 20)),
        ...     dims=["k", "j", "i"],
        ...     coords={"k": np.arange(3), "j": np.arange(10), "i": np.arange(20)},
        ... )
        >>> data = data.fusi.affine.set_voxel_to_world(
        ...     np.diag([0.2, 0.1, 0.05, 1.0])
        ... )
        >>> data.fusi.spacing
        {'k': 0.2, 'j': 0.1, 'i': 0.05}
        """
        from confusius._utils.coordinates import get_coordinate_spacing_info
        from confusius._utils.geometry import (
            get_voxel_to_world_index_spacing,
            has_voxel_to_world_index,
        )

        if not has_voxel_to_world_index(self._obj):
            raise ValueError("DataArray must have a voxel-to-world index.")

        voxel_spacing = get_voxel_to_world_index_spacing(self._obj)
        missing_dims = [
            str(dim) for dim in self._obj.dims if str(dim) not in voxel_spacing
        ]
        regular_spacing = {
            dim: get_coordinate_spacing_info(dim, self._obj, 1e-2).value
            for dim in missing_dims
        }
        return {
            dim_str: voxel_spacing[dim_str]
            if dim_str in voxel_spacing
            else regular_spacing[dim_str]
            for dim_str in (str(dim) for dim in self._obj.dims)
        }

    @property
    def origin(self) -> dict[str, float]:
        """World origin metadata for the DataArray.

        Non-spatial dimensions use their first coordinate value. Spatial origin is
        returned in world coordinate order as the world location of the first
        sampled voxel under the DataArray's voxel-to-world affine.

        Returns
        -------
        dict[str, float]
            Origin metadata for the DataArray.

        Raises
        ------
        ValueError
            If `self` does not carry a voxel-to-world index.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> data = xr.DataArray(
        ...     np.zeros((3, 10, 20)),
        ...     dims=["k", "j", "i"],
        ...     coords={"k": np.arange(3), "j": np.arange(10), "i": np.arange(20)},
        ... )
        >>> voxel_to_world = np.array(
        ...     [[0.2, 0.0, 0.0, 1.0], [0.0, 0.1, 0.0, 2.0], [0.0, 0.0, 0.05, 3.0], [0.0, 0.0, 0.0, 1.0]]
        ... )
        >>> data = data.fusi.affine.set_voxel_to_world(voxel_to_world)
        >>> data.fusi.origin
        {'z': 1.0, 'y': 2.0, 'x': 3.0}
        """
        from confusius._utils.coordinates import get_coordinate_origins
        from confusius._utils.geometry import (
            get_voxel_to_world_index_origin,
            get_voxel_to_world_spatial_dims,
            has_voxel_to_world_index,
        )

        if not has_voxel_to_world_index(self._obj):
            raise ValueError("DataArray must have a voxel-to-world index.")

        voxel_dims = set(get_voxel_to_world_spatial_dims(self._obj))
        regular_origin = get_coordinate_origins(self._obj)
        return {
            **{
                dim_str: regular_origin[dim_str]
                for dim_str in (str(dim) for dim in self._obj.dims)
                if dim_str not in voxel_dims
            },
            **get_voxel_to_world_index_origin(self._obj),
        }

    @property
    def direction(self):
        """World-space direction matrix for the present spatial geometry.

        Returns
        -------
        numpy.ndarray
            Identity for axis-aligned data. For oblique data, the columns are the
            unit world-space directions of the voxel axes.

        Raises
        ------
        ValueError
            If `self` does not carry a voxel-to-world index.
        """
        from confusius._utils.geometry import (
            get_voxel_to_world_direction_matrix,
            has_voxel_to_world_index,
        )

        if not has_voxel_to_world_index(self._obj):
            raise ValueError("DataArray must have a voxel-to-world index.")
        return get_voxel_to_world_direction_matrix(self._obj)

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
        from confusius.xarray.affine import FUSIAffineAccessor

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
        ...     np.zeros((10, 32, 1, 64)), dims=["time", "k", "j", "i"]
        ... )
        >>> data.fusi.save("recording.nii.gz")
        """
        from confusius.io.loadsave import save

        save(self._obj, path, **kwargs)
