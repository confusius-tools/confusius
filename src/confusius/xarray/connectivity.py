"""Xarray accessor for connectivity analysis."""

from typing import Literal

import xarray as xr


class FUSIConnectivityAccessor:
    """Xarray accessor for seed-based functional connectivity analysis.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import confusius  # noqa: F401
    >>>
    >>> data = xr.open_zarr("recording.zarr")["power_doppler"]
    >>> seed_masks = xr.open_zarr("seed_masks.zarr")["masks"]
    >>> mapper = data.fusi.connectivity.seed_map(seed_masks=seed_masks)
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def seed_map(
        self,
        *,
        seed_masks: xr.DataArray | None = None,
        seed_signals: xr.DataArray | None = None,
        labels_reduction: Literal[
            "mean", "sum", "median", "min", "max", "var", "std"
        ] = "mean",
        clean_kwargs: dict | None = None,
    ) -> "confusius.connectivity.SeedBasedMaps":  # type: ignore[name-defined]  # noqa: F821
        """Fit a seed-based correlation map.

        Convenience wrapper around
        [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps] that constructs the
        estimator, calls [`SeedBasedMaps.fit`][confusius.connectivity.SeedBasedMaps.fit]
        on the wrapped DataArray, and returns the fitted estimator.

        Parameters
        ----------
        seed_masks : xarray.DataArray, optional
            Integer label map defining the seed region(s). See
            [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps] for accepted
            formats. Mutually exclusive with `seed_signals`.
        seed_signals : xarray.DataArray, optional
            Pre-computed `(time, ...)` seed signals used directly for correlation.
            When provided, seed extraction from the data is skipped.
            Mutually exclusive with `seed_masks`.
        labels_reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, \
                default: "mean"
            Aggregation function applied across voxels within each seed region.
            Ignored when `seed_signals` is provided.
        clean_kwargs : dict or None, default: None
            Keyword arguments forwarded to [`clean`][confusius.signal.clean]. If not
            provided, no cleaning is applied.

        Returns
        -------
        confusius.connectivity.SeedBasedMaps
            Fitted estimator.  Access `maps_` and `seed_signals_` on the
            returned object.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import confusius  # noqa: F401
        >>>
        >>> data = xr.open_zarr("recording.zarr")["power_doppler"]
        >>> seed_masks = xr.open_zarr("seed_masks.zarr")["masks"]
        >>> mapper = data.fusi.connectivity.seed_map(seed_masks=seed_masks)
        """
        from confusius.connectivity import SeedBasedMaps

        return SeedBasedMaps(
            seed_masks=seed_masks,
            seed_signals=seed_signals,
            labels_reduction=labels_reduction,
            clean_kwargs=clean_kwargs,
        ).fit(self._obj)
