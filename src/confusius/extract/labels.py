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


def extract_with_labels(
    data: xr.DataArray,
    labels: xr.DataArray,
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"] = "mean",
) -> xr.DataArray:
    """Extract region-aggregated signals from fUSI data using an integer label map.

    For each unique non-zero label in `labels`, applies `reduction` across all voxels
    belonging to that region. The spatial dimensions are collapsed into a single
    `regions` dimension whose coordinates are the label integers.

    Parameters
    ----------
    data : xarray.DataArray
        Input array with spatial dimensions matching `labels`. Can have any number of
        non-spatial dimensions (e.g., `time`, `pose`). The spatial dimensions must match
        those in `labels`.
    labels : xarray.DataArray
        Integer label map defining the regions. Background voxels must be labeled `0`.
        Each unique non-zero integer identifies a distinct region. Its dimensions define
        the spatial dimensions that will be reduced over. Must have identical
        coordinates to the data's spatial dimensions.
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
        Array with spatial dimensions replaced by a `regions` dimension. The `regions`
        dimension has integer coordinates corresponding to each unique non-zero label
        in `labels`. All non-spatial dimensions are preserved.

        For example:

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
    >>> # Sum instead of mean.
    >>> sums = extract_with_labels(data, labels, reduction="sum")
    """
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Must be one of: {list(_VALID_REDUCTIONS)}."
        )

    validate_labels(labels, data, "labels")

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

    return result.assign_coords(regions=unique_labels)
