"""Extraction of region-aggregated signals using integer label maps."""

from typing import Hashable, Literal

import flox.xarray
import numpy as np
import xarray as xr

from confusius.validation import validate_labels

_VALID_REDUCTIONS = frozenset({"mean", "sum", "median", "min", "max", "var", "std"})
"""Valid reduction names accepted by `extract_with_labels`."""


def _get_stacked_mask_ids(labels: xr.DataArray) -> list[int]:
    """Validate and return the unique non-zero ID from each stacked mask layer.

    Parameters
    ----------
    labels : xarray.DataArray
        Stacked mask array with a leading `mask` dimension.

    Returns
    -------
    list[int]
        Non-zero region ID for each mask layer, in mask-dimension order.

    Raises
    ------
    ValueError
        If any layer has more or fewer than one non-zero value, or if IDs
        are duplicated across layers.
    """
    per_mask_id: list[int] = []
    for i in range(labels.sizes["mask"]):
        layer_vals = np.unique(labels.isel(mask=i).values)
        layer_vals = layer_vals[layer_vals != 0]
        if len(layer_vals) != 1:
            raise ValueError(
                f"Stacked mask layer {i} must have exactly one unique non-zero "
                f"value, got {len(layer_vals)}: {layer_vals.tolist()}."
            )
        per_mask_id.append(int(layer_vals[0]))

    duplicates = {v for v in per_mask_id if per_mask_id.count(v) > 1}
    if duplicates:
        raise ValueError(
            f"Stacked mask layers have duplicate non-zero IDs: {duplicates}. "
            "Each layer must carry a unique region ID."
        )

    return per_mask_id


def _flox_reduce(
    data: xr.DataArray,
    labels: xr.DataArray,
    spatial_dims: list[Hashable],
    reduction: Literal["mean", "sum", "median", "min", "max", "var", "std"],
) -> xr.DataArray:
    """Reduce `data` over `spatial_dims` grouped by `labels` using flox.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    labels : xarray.DataArray
        Integer label map. May have a leading `mask` dimension for the
        overlapping groups case.
    spatial_dims : list[str]
        Spatial dimensions to reduce over.
    reduction : str
        Aggregation function name.

    Returns
    -------
    xarray.DataArray
        Array with spatial dimensions replaced by a `region` dimension,
        with integer region coordinates. Background (0) is excluded.
    """
    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]
    template = data.isel({d: 0 for d in non_spatial_dims}) if non_spatial_dims else data
    labels_aligned = labels.reindex_like(template)

    if not (labels_aligned != 0).any():
        raise ValueError(
            "labels contains no non-zero values: no regions to extract. "
            "If you are using draw_napari_labels, make sure you have painted "
            "at least one label in the napari viewer before calling "
            "extract_with_labels."
        )

    # flox names the output groupby dimension after the variable name of the `by` array.
    data_stacked = data.stack(space=spatial_dims)
    # Compute labels eagerly: flox cannot determine unique group values from a
    # Dask-backed array without expected_groups. Labels are spatial-only and
    # always small enough to materialise.
    labels_stacked = labels_aligned.stack(space=spatial_dims).rename("region").compute()
    result = flox.xarray.xarray_reduce(data_stacked, labels_stacked, func=reduction)
    return result.isel(region=result.region.values != 0)  # Drop background.


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
    Uses [flox](https://flox.readthedocs.io/en/latest/) for efficient, lazy groupby
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
        spatial_dims = [d for d in labels.dims if d != "mask"]
        per_mask_id = _get_stacked_mask_ids(labels)
        id_to_name = dict(zip(per_mask_id, labels.coords["mask"].values))

        has_overlap = bool(((labels > 0).sum(dim="mask") > 1).any().values)

        if not has_overlap:
            # Sum across mask dim: no-overlap guarantees each voxel keeps its original
            # ID.
            result = _flox_reduce(data, labels.sum(dim="mask"), spatial_dims, reduction)
        else:
            # Use flox's overlapping groups support.
            # See: https://flox.readthedocs.io/en/latest/user-stories/overlaps.html
            result = _flox_reduce(data, labels, spatial_dims, reduction)

        region_names = [id_to_name[int(r)] for r in result.coords["region"].values]
        return result.assign_coords(region=region_names)
    else:
        return _flox_reduce(data, labels, list(labels.dims), reduction)
