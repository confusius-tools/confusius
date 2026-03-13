"""Xarray accessor for signal extraction."""

from typing import Literal

import xarray as xr


class FUSIExtractAccessor:
    """Xarray accessor for signal extraction operations.

    Provides convenient methods for extracting signals from N-D fUSI data by flattening
    spatial dimensions, and reconstructing N-D volumes from processed signals.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>>
    >>> # 3D+t data: (time, z, y, x)
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>> mask = xr.DataArray(
    ...     np.random.rand(10, 20, 30) > 0.5,
    ...     dims=["z", "y", "x"],
    ... )
    >>>
    >>> # Extract signals
    >>> signals = data.fusi.extract.with_mask(mask)
    >>> signals.dims
    ("time", "space")
    >>>
    >>> # Reconstruct full spatial volume from signals
    >>> reconstructed = signals.fusi.extract.unmask(mask)
    >>> reconstructed.dims
    ("time", "z", "y", "x")
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def with_labels(
        self,
        labels: xr.DataArray,
        reduction: Literal[
            "mean", "sum", "median", "min", "max", "var", "std"
        ] = "mean",
    ) -> xr.DataArray:
        """Extract region-aggregated signals using an integer label map.

        Parameters
        ----------
        labels : xarray.DataArray
            Integer label map in one of two formats:

            - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
              labeled `0`; each unique non-zero integer identifies a distinct,
              non-overlapping region. The `regions` coordinate of the output holds the
              integer label values.
            - **Stacked mask format**: Has a leading `mask` dimension followed by
              spatial dims, e.g. `(mask, z, y, x)`. Each layer has values in `{0,
              region_id}` and regions may overlap. The `region` coordinate of the
              output holds the `mask` coordinate values (e.g., region label).

        reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, default: "mean"
            Aggregation function applied across voxels in each region:

            - `"mean"`: arithmetic mean.
            - `"sum"`: sum of values.
            - `"median"`: median value.
            - `"min"`: minimum value.
            - `"max"`: maximum value.
            - `"var"`: variance.
            - `"std"`: standard deviation.

        Returns
        -------
        xarray.DataArray
            Array with spatial dimensions replaced by a `region` dimension. The
            `region` dimension has integer coordinates corresponding to each unique
            non-zero label in `labels`. All non-spatial dimensions are preserved.

            For example:

            - `(time, z, y, x)` → `(time, region)`
            - `(time, pose, z, y, x)` → `(time, pose, region)`
            - `(z, y, x)` → `(region,)`

        Raises
        ------
        ValueError
            If `labels` dimensions don't match `data`'s spatial dimensions, if
            coordinates don't match, or if `reduction` is not a valid option.
        TypeError
            If `labels` is not integer dtype.

        Examples
        --------
        >>> signals = data.fusi.extract.with_labels(labels)
        >>> signals.dims
        ("time", "region")
        >>> signals.coords["region"].values
        array([1, 2, 3])
        >>>
        >>> # Sum per region instead of mean.
        >>> sums = data.fusi.extract.with_labels(labels, reduction="sum")
        """
        from confusius.extract.labels import extract_with_labels

        return extract_with_labels(self._obj, labels, reduction=reduction)

    def with_mask(self, mask: xr.DataArray) -> xr.DataArray:
        """Extract signals using a boolean or single-label integer mask.

        Parameters
        ----------
        mask : xarray.DataArray
            Mask defining which voxels to extract. Its dimensions define the spatial
            dimensions that will be flattened. Must have boolean dtype, or integer dtype
            with exactly one non-zero value (0 = background, one region id =
            foreground). The latter format is produced by
            [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks]. Coordinates must match
            data.

        Returns
        -------
        xarray.DataArray
            Array with spatial dimensions flattened into a `space` dimension.
            All non-spatial dimensions are preserved. The `space` dimension has a
            MultiIndex storing spatial coordinates.

            For simple round-trip reconstruction, use `.unstack("space")` which
            re-creates the original DataArray using the smallest bounding box. For full
            mask shape reconstruction, use `.fusi.extract.unmask()`.

        Raises
        ------
        ValueError
            If `mask` dimensions don't match `data`'s spatial dimensions.
        TypeError
            If `mask` is not boolean dtype.

        Examples
        --------
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> signals.dims
        ("time", "space")
        >>>
        >>> # Quick bounding box reconstruction
        >>> bbox = signals.unstack("space")
        >>>
        >>> # Full mask shape reconstruction
        >>> full = signals.fusi.extract.unmask(mask)
        """
        from confusius.extract.mask import extract_with_mask

        return extract_with_mask(self._obj, mask)

    def unmask(
        self,
        mask: xr.DataArray,
        fill_value: float = 0.0,
    ) -> xr.DataArray:
        """Reconstruct N-D volume from masked signals.

        Reconstructs the full spatial volume from a DataArray of signals, which must
        have a `voxels` dimension. This is a convenience wrapper around
        `confusius.extract.unmask()`.

        Parameters
        ----------
        mask : xarray.DataArray
            Boolean mask used for the original extraction. Provides spatial dimensions
            and coordinates for reconstruction.
        fill_value : float, default: 0.0
            Value to fill in non-masked voxels.

        Returns
        -------
        xarray.DataArray
            Reconstructed DataArray with shape `(..., z, y, x)` where spatial
            coordinates come from the mask.

        Examples
        --------
        >>> signals = data.fusi.extract.with_mask(mask)
        >>> signals.dims
        ("time", "space")
        >>> reconstructed = signals.fusi.extract.unmask(mask)
        >>> reconstructed.dims
        ("time", "z", "y", "x")
        """
        from confusius.extract.reconstruction import unmask

        return unmask(self._obj, mask=mask, fill_value=fill_value)
