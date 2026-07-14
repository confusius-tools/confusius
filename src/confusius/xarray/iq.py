"""Xarray accessor for IQ processing."""

from typing import Literal

import xarray as xr

from confusius.iq.process import (
    process_iq_to_axial_velocity,
    process_iq_to_bmode,
    process_iq_to_power_doppler,
)


class FUSIIQAccessor:
    """Accessor for IQ processing operations on fUSI data.

    This accessor provides methods to process beamformed IQ data into
    derived quantities such as power Doppler and axial velocity.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap. Must contain complex beamformed IQ data
        with dimensions `(time, z, y, x)`.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_zarr("iq_data.zarr")
    >>> iq = ds["iq"]
    >>> pwd = iq.fusi.iq.process_to_power_doppler(low_cutoff=40)
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def process_to_power_doppler(
        self,
        clutter_window_width: int | None = None,
        clutter_window_stride: int | None = None,
        filter_method: Literal[
            "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
        ] = "svd_indices",
        clutter_mask: xr.DataArray | None = None,
        low_cutoff: int | float | None = None,
        high_cutoff: int | float | None = None,
        butterworth_order: int = 4,
        doppler_window_width: int | None = None,
        doppler_window_stride: int | None = None,
    ) -> xr.DataArray:
        """Process beamformed IQ into power Doppler volumes.

        This method computes power Doppler volumes from beamformed IQ data using
        nested sliding windows. A first sliding window is used for clutter filtering.
        Inside each clutter-filtered window, power Doppler volumes are computed using
        a second sliding window.

        Parameters
        ----------
        clutter_window_width : int, optional
            Width of the sliding temporal window for clutter filtering, in volumes.
            If not provided, uses the chunk size of the IQ data along the temporal
            dimension.
        clutter_window_stride : int, optional
            Stride of the sliding temporal window for clutter filtering, in volumes.
            If not provided, equals `clutter_window_width`.
        filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", \
                "butterworth"}, default: "svd_indices"
            Clutter filtering method to apply before power Doppler computation.

            - `"svd_indices"`: Static SVD filter using singular vector indices.
            - `"svd_energy"`: Adaptive SVD filter using singular vector energies.
            - `"svd_cumulative_energy"`: Adaptive SVD filter using cumulative energies.
            - `"butterworth"`: Butterworth frequency-domain filter.
        clutter_mask : (z, y, x) xarray.DataArray, optional
            Boolean mask to define clutter regions. Only used by SVD-based clutter
            filters to compute clutter vectors from masked voxels. If not provided,
            all voxels are used. The mask spatial coordinates (z, y, x) must match
            the IQ data coordinates.
        low_cutoff : int or float, optional
            Low cutoff for clutter filtering. Interpretation depends on `filter_method`.
            If not provided, uses method-specific defaults.
        high_cutoff : int or float, optional
            High cutoff for clutter filtering. Interpretation depends on
            `filter_method`. If not provided, uses method-specific defaults.
        butterworth_order : int, default: 4
            Order of Butterworth filter. Effective order is doubled due to
            forward-backward filtering.
        doppler_window_width : int, optional
            Width of the sliding temporal window for power Doppler integration, in
            volumes. If not provided, equals `clutter_window_width`.
        doppler_window_stride : int, optional
            Stride of the sliding temporal window for power Doppler integration, in
            volumes. If not provided, equals `doppler_window_width`.

        Returns
        -------
        (clutter_windows * doppler_windows, z, y, x) xarray.DataArray
            Power Doppler volumes with updated time coordinates, where
            `clutter_windows` is the number of clutter filter sliding windows and
            `doppler_windows` is the number of power Doppler sliding windows per
            clutter window.

        Examples
        --------
        >>> import xarray as xr
        >>> ds = xr.open_zarr("iq_data.zarr")
        >>> iq = ds["iq"]
        >>> pwd = iq.fusi.iq.process_to_power_doppler(
        ...     clutter_window_width=50,
        ...     doppler_window_width=25,
        ...     low_cutoff=40,
        ... )
        """
        return process_iq_to_power_doppler(
            self._obj,
            clutter_window_width=clutter_window_width,
            clutter_window_stride=clutter_window_stride,
            filter_method=filter_method,
            clutter_mask=clutter_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            butterworth_order=butterworth_order,
            doppler_window_width=doppler_window_width,
            doppler_window_stride=doppler_window_stride,
        )

    def process_to_axial_velocity(
        self,
        clutter_window_width: int | None = None,
        clutter_window_stride: int | None = None,
        filter_method: Literal[
            "svd_indices", "svd_energy", "svd_cumulative_energy", "butterworth"
        ] = "svd_indices",
        clutter_mask: xr.DataArray | None = None,
        low_cutoff: int | float | None = None,
        high_cutoff: int | float | None = None,
        butterworth_order: int = 4,
        velocity_window_width: int | None = None,
        velocity_window_stride: int | None = None,
        lag: int = 1,
        absolute_velocity: bool = False,
        spatial_kernel: int | tuple[int, int, int] | list[int] = 3,
    ) -> xr.DataArray:
        """Process beamformed IQ into axial velocity volumes.

        This method computes axial velocity volumes from beamformed IQ data using
        nested sliding windows. A first sliding window is used for clutter filtering.
        Inside each clutter-filtered window, axial velocity volumes are computed using
        a second sliding window.

        Parameters
        ----------
        clutter_window_width : int, optional
            Width of the sliding temporal window for clutter filtering, in volumes.
            If not provided, uses the chunk size of the IQ data along the temporal
            dimension.
        clutter_window_stride : int, optional
            Stride of the sliding temporal window for clutter filtering, in volumes.
            If not provided, equals `clutter_window_width`.
        filter_method : {"svd_indices", "svd_energy", "svd_cumulative_energy", \
                "butterworth"}, default: "svd_indices"
            Clutter filtering method to apply before velocity computation.

            - `"svd_indices"`: Static SVD filter using singular vector indices.
            - `"svd_energy"`: Adaptive SVD filter using singular vector energies.
            - `"svd_cumulative_energy"`: Adaptive SVD filter using cumulative energies.
            - `"butterworth"`: Butterworth frequency-domain filter.
        clutter_mask : (z, y, x) xarray.DataArray, optional
            Boolean mask to define clutter regions. Only used by SVD-based clutter
            filters to compute clutter vectors from masked voxels. If not provided,
            all voxels are used. The mask spatial coordinates (z, y, x) must match
            the IQ data coordinates.
        low_cutoff : int or float, optional
            Low cutoff for clutter filtering. Interpretation depends on `filter_method`.
            If not provided, uses method-specific defaults.
        high_cutoff : int or float, optional
            High cutoff for clutter filtering. Interpretation depends on
            `filter_method`. If not provided, uses method-specific defaults.
        butterworth_order : int, default: 4
            Order of Butterworth filter. Effective order is doubled due to
            forward-backward filtering.
        velocity_window_width : int, optional
            Width of the sliding temporal window for velocity estimation, in volumes.
            If not provided, equals `clutter_window_width`.
        velocity_window_stride : int, optional
            Stride of the sliding temporal window for velocity estimation, in volumes.
            If not provided, equals `velocity_window_width`.
        lag : int, default: 1
            Temporal lag in volumes for autocorrelation computation. Must be positive.
        absolute_velocity : bool, default: False
            If `True`, compute absolute velocity values. If `False`, preserve sign
            information.
        spatial_kernel : int or tuple[int, int, int] or list[int], default: 3
            Size of the median filter kernel applied spatially to denoise. A scalar
            uses the same kernel size on all spatial axes; a length-3 sequence
            specifies `(z, y, x)` sizes directly. Values must be positive. Any even
            sizes are rounded up to the next odd size. If all sizes are `1`, no
            spatial filtering is applied.

        Returns
        -------
        (clutter_windows * velocity_windows, z, y, x) xarray.DataArray
            Axial velocity volumes with updated time coordinates, where
            `clutter_windows` is the number of clutter filter sliding windows and
            `velocity_windows` is the number of velocity sliding windows per clutter
            window. Velocity values are in meters per second.

        Examples
        --------
        >>> import xarray as xr
        >>> ds = xr.open_zarr("iq_data.zarr")
        >>> iq = ds["iq"]
        >>> velocity = iq.fusi.iq.process_to_axial_velocity(
        ...     clutter_window_width=50,
        ...     velocity_window_width=25,
        ... )
        """
        return process_iq_to_axial_velocity(
            self._obj,
            clutter_window_width=clutter_window_width,
            clutter_window_stride=clutter_window_stride,
            filter_method=filter_method,
            clutter_mask=clutter_mask,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            butterworth_order=butterworth_order,
            velocity_window_width=velocity_window_width,
            velocity_window_stride=velocity_window_stride,
            lag=lag,
            absolute_velocity=absolute_velocity,
            spatial_kernel=spatial_kernel,
        )

    def process_to_bmode(
        self,
        bmode_window_width: int | None = None,
        bmode_window_stride: int | None = None,
    ) -> xr.DataArray:
        """Process beamformed IQ into B-mode volumes.

        This method computes B-mode volumes from beamformed IQ data using a single
        sliding temporal window. Unlike power Doppler, no clutter filtering is applied;
        the mean magnitude (not squared magnitude) of the IQ data within each window
        is computed.

        Parameters
        ----------
        bmode_window_width : int, optional
            Width of the sliding temporal window for B-mode integration, in volumes.
            If not provided, uses the chunk size of the IQ data along the temporal
            dimension.
        bmode_window_stride : int, optional
            Stride of the sliding temporal window, in volumes. If not provided, equals
            `bmode_window_width`.

        Returns
        -------
        (windows, z, y, x) xarray.DataArray
            B-mode volumes with updated time coordinates, where `windows` is the number
            of sliding windows.

        Examples
        --------
        >>> import xarray as xr
        >>> ds = xr.open_zarr("iq_data.zarr")
        >>> iq = ds["iq"]
        >>> bmode = iq.fusi.iq.process_to_bmode(bmode_window_width=50)
        """
        return process_iq_to_bmode(
            self._obj,
            bmode_window_width=bmode_window_width,
            bmode_window_stride=bmode_window_stride,
        )
