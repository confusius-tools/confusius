"""Region aggregation with ConfUSIus acquisition validation."""

from typing import Literal

import xarray as xr
from voxeldata.extract import extract_with_labels as extract_generic_labels

from confusius.validation import ensure_voxeldata


def extract_with_labels(
    data: xr.DataArray,
    labels: xr.DataArray,
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"] = "mean",
) -> xr.DataArray:
    """Aggregate integer regions or overlapping mask layers through VoxelData.

    ConfUSIus retains acquisition defaults/validation. VoxelData owns grouped
    reductions and alignment, using Xarray with optional flox acceleration.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array with arbitrary nonspatial dimensions.
    labels : xarray.DataArray
        Integer VoxelData labels with spatial dimensions or `(mask, k, j, i)`.
        Zero is background. Flat maps identify regions by integer values; stacked
        layers identify them by `mask` coordinates and may overlap or reuse ids.
        Each layer must contain exactly one nonzero id. Physical grids must match.
    reduction : {"mean", "sum", "median", "min", "max", "var", "std"}, default: "mean"
        Reduction across each region's samples.

    Returns
    -------
    xarray.DataArray
        Nonspatial signal dimensions followed by `region`. Dask signals remain
        lazy. With flox, median may require explicit spatial rechunking.

    Raises
    ------
    TypeError
        If labels are not an integer DataArray.
    ValueError
        If geometry, label format, reduction, chunking, or acquisition metadata
        are invalid, or a shared map spans physically varying geometry.
    """
    data = ensure_voxeldata(data, allow_extra_dims=True)
    labels = ensure_voxeldata(labels, allow_extra_dims=True)
    return extract_generic_labels(data, labels, reduction)
