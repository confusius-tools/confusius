"""Extraction of region-aggregated signals using integer label maps."""

from typing import Literal

import numpy as np
import xarray as xr

from confusius.validation import validate_labels

_VALID_REDUCTIONS = {
    "mean": np.mean,
    "sum": np.sum,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "var": np.var,
    "std": np.std,
}
"""Valid reduction functions accepted by `extract_with_labels`."""


def _reduce_by_label(data_nd, *, labels_1d, unique_labels, np_func):
    """Apply np_func to each region's voxels along the last axis.

    Parameters
    ----------
    data_nd : numpy.ndarray
        Array with shape `(..., n_space)` where the last axis contains flattened
        spatial voxels.
    labels_1d : numpy.ndarray
        1-D integer label array of length `n_space`. Zero is background.
    unique_labels : numpy.ndarray
        Sorted array of non-zero unique label values.
    np_func : callable
        NumPy reduction function accepting an `axis` keyword (e.g. `np.mean`).

    Returns
    -------
    numpy.ndarray
        Array with shape `(..., n_regions)`.
    """
    return np.stack(
        [np_func(data_nd[..., labels_1d == label], axis=-1) for label in unique_labels],
        axis=-1,
    )


def _extract_with_flat_labels(
    data: xr.DataArray,
    labels: xr.DataArray,
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"],
    region_coords: "np.ndarray | None" = None,
) -> xr.DataArray:
    """Core extraction logic for a flat (spatial-only) integer label map.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    labels : xarray.DataArray
        Flat integer label map, spatial dimensions only.
    reduction : str
        Aggregation function name.
    region_coords : numpy.ndarray or None
        Coordinate values for the output `regions` dimension. When `None`,
        the unique non-zero labels are used as integer coordinates.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions replaced by a `regions` dimension.
    """
    spatial_dims = list(labels.dims)
    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]

    if non_spatial_dims:
        sel_dict = {d: 0 for d in non_spatial_dims}
        template = data.isel(sel_dict)
    else:
        template = data

    labels_aligned = labels.reindex_like(template)

    unique_labels = np.unique(labels_aligned.values)
    unique_labels = unique_labels[unique_labels != 0]

    np_func = _VALID_REDUCTIONS[reduction]
    labels_np = labels_aligned.values.flatten()

    # Stack spatial dims into a single "_space" axis, then use apply_ufunc to
    # apply the numpy reduction per-region. apply_ufunc with dask="parallelized"
    # keeps Dask arrays lazy; dask auto-rechunks the "_space" core dim to a
    # single chunk (required so each block sees all voxels for a given region).
    data_stacked = data.stack(_space=spatial_dims)

    result = xr.apply_ufunc(
        _reduce_by_label,
        data_stacked,
        kwargs={
            "labels_1d": labels_np,
            "unique_labels": unique_labels,
            "np_func": np_func,
        },
        input_core_dims=[["_space"]],
        output_core_dims=[["regions"]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"regions": len(unique_labels)}},
        keep_attrs=True,
    )

    coords = region_coords if region_coords is not None else unique_labels
    return result.assign_coords(regions=coords)


def extract_with_labels(
    data: xr.DataArray,
    labels: xr.DataArray,
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"] = "mean",
) -> xr.DataArray:
    """Extract region-aggregated signals from fUSI data using an integer label map.

    For each unique non-zero label in `labels`, applies `reduction` across all voxels
    belonging to that region. The spatial dimensions are collapsed into a single
    `regions` dimension.

    Parameters
    ----------
    data : xarray.DataArray
        Input array with spatial dimensions matching `labels`. Can have any number of
        non-spatial dimensions (e.g., `time`, `pose`). The spatial dimensions must match
        those in `labels`.
    labels : xarray.DataArray
        Integer label map in one of two formats:

        - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `masks` dimension followed by spatial
          dims, e.g. `(masks, z, y, x)`. Each layer has values in `{0, region_id}`
          and regions may overlap. The `regions` coordinate of the output holds the
          `masks` coordinate values (e.g., region label).

    reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, default: "mean"
        Aggregation function applied across voxels in each region.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions replaced by a `regions` dimension. All
        non-spatial dimensions are preserved.

        For example (flat label map):

        - `(time, z, y, x)` → `(time, regions)`
        - `(time, pose, z, y, x)` → `(time, pose, regions)`
        - `(z, y, x)` → `(regions,)`

    Raises
    ------
    ValueError
        If `labels` dimensions don't match `data`'s spatial dimensions, if
        coordinates don't match, or if `reduction` is not a valid option.
    TypeError
        If `labels` is not integer dtype.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> from confusius.extract import extract_with_labels
    >>>
    >>> # 3D+t data: (time, z, y, x)
    >>> data = xr.DataArray(
    ...     np.random.randn(100, 10, 20, 30),
    ...     dims=["time", "z", "y", "x"],
    ... )
    >>> labels = xr.DataArray(
    ...     np.zeros((10, 20, 30), dtype=int),
    ...     dims=["z", "y", "x"],
    ... )
    >>> labels[0, :, :] = 1  # Region 1: first z-slice.
    >>> labels[1, :, :] = 2  # Region 2: second z-slice.
    >>> signals = extract_with_labels(data, labels)
    >>> signals.dims
    ('time', 'regions')
    >>> signals.coords["regions"].values
    array([1, 2])
    >>>
    >>> # Stacked mask format from Atlas.get_masks.
    >>> masks = atlas_fusi.get_masks(["VISp", "AUDp"])
    >>> signals = extract_with_labels(data, masks)
    >>> signals.coords["regions"].values
    array(['VISp', 'AUDp'], dtype=object)
    """
    validate_labels(labels, data, "labels")

    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Must be one of: {list(_VALID_REDUCTIONS)}."
        )

    if "masks" in labels.dims:
        region_results = []
        for i in range(labels.sizes["masks"]):
            layer = labels.isel(masks=i)
            region_result = _extract_with_flat_labels(data, layer, reduction)
            region_results.append(region_result.isel(regions=0))

        result = xr.concat(region_results, dim="regions")
        return result.assign_coords(regions=labels.coords["masks"].values)
    else:
        return _extract_with_flat_labels(data, labels, reduction)
