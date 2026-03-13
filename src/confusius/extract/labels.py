"""Extraction of region-aggregated signals using integer label maps."""

from typing import Literal

import numpy as np
import xarray as xr

from confusius.validation import validate_labels

_VALID_REDUCTIONS = frozenset({"mean", "sum", "median", "min", "max", "var", "std"})
"""Valid reduction names accepted by `extract_with_labels`."""


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
        Coordinate values for the output `region` dimension. When `None`,
        the unique non-zero labels are used as integer coordinates.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions replaced by a `region` dimension.
    """
    spatial_dims = list(labels.dims)
    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]

    if non_spatial_dims:
        template = data.isel({d: 0 for d in non_spatial_dims})
    else:
        template = data

    labels_aligned = labels.reindex_like(template)

    unique_labels = np.unique(labels_aligned.values)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        raise ValueError(
            "labels contains no non-zero values: no regions to extract. "
            "If you are using draw_napari_labels, make sure you have painted "
            "at least one label in the napari viewer before calling "
            "extract_with_labels."
        )

    # We mask background (0) to NaN so groupby skips it. Naming the labels array
    # "region" makes the groupby output dimension adopt that name directly.
    data_stacked = data.stack(space=spatial_dims)
    labels_stacked = labels_aligned.stack(space=spatial_dims)
    labels_no_bg = labels_stacked.where(labels_stacked != 0).rename("region")
    result = getattr(data_stacked.groupby(labels_no_bg), reduction)()

    coords = region_coords if region_coords is not None else unique_labels
    return result.assign_coords(region=coords)


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
          non-overlapping region. The `region` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `mask` dimension followed by spatial
          dims, e.g. `(mask, z, y, x)`. Each layer has values in `{0, region_id}`
          and regions may overlap. The `region` coordinate of the output holds the
          `mask` coordinate values (e.g., region label).

    reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, default: "mean"
        Aggregation function applied across voxels in each region.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions replaced by a `region` dimension. All
        non-spatial dimensions are preserved.

        For example (flat label map):

        - `(time, z, y, x)` → `(time, region)`
        - `(time, pose, z, y, x)` → `(time, pose, region)`
        - `(z, y, x)` → `(region,)`

    Raises
    ------
    ValueError
        If `labels` dimensions don't match `data`'s spatial dimensions, if
        coordinates don't match, if `reduction` is not a valid option, or if
        `labels` contains no non-zero values.
    TypeError
        If `labels` is not integer dtype.

    Notes
    -----
    Uses [`xarray.DataArray.groupby`][xarray.DataArray.groupby] with
    [flox](https://flox.readthedocs.io/en/latest/) for efficient, lazy groupby
    reductions on Dask-backed arrays. Data can be chunked along any dimension without
    restriction.

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
    ('time', 'region')
    >>> signals.coords["region"].values
    array([1, 2])
    >>>
    >>> # Stacked mask format from Atlas.get_masks.
    >>> mask = atlas_fusi.get_masks(["VISp", "AUDp"])
    >>> signals = extract_with_labels(data, mask)
    >>> signals.coords["region"].values
    array(['VISp', 'AUDp'], dtype=object)
    """
    validate_labels(labels, data, "labels")

    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Must be one of: {sorted(_VALID_REDUCTIONS)}."
        )

    if "mask" in labels.dims:
        region_results = []
        for i in range(labels.sizes["mask"]):
            # We drop the "mask" dimension here otherwise xr.concat would complain
            # about non-identical coordinates across regions.
            layer = labels.isel(mask=i, drop=True)
            region_result = _extract_with_flat_labels(data, layer, reduction)
            region_results.append(region_result.isel(region=0))

        result = xr.concat(region_results, dim="region", coords="minimal")
        return result.assign_coords(region=labels.coords["mask"].values)
    else:
        return _extract_with_flat_labels(data, labels, reduction)
