"""Mask validation utilities."""

import numpy as np
import xarray as xr

from confusius.validation.coordinates import validate_matching_coordinates


def _validate_spatial_coords(
    spatial_da: xr.DataArray,
    data: xr.DataArray,
    name: str,
    rtol: float,
    atol: float,
) -> None:
    """Validate spatial dimensions and coordinates of `spatial_da` against `data`.

    Asserts that `spatial_da.dims` is a subset of `data.dims`, then delegates
    coordinate comparison to
    [`validate_matching_coordinates`][confusius.validation.coordinates.validate_matching_coordinates].

    Parameters
    ----------
    spatial_da : xarray.DataArray
        Spatial DataArray to validate (mask or labels).
    data : xarray.DataArray
        Reference data array.
    name : str
        Label used for `spatial_da` in error messages.
    rtol : float
        Relative tolerance for coordinate comparison.
    atol : float
        Absolute tolerance for coordinate comparison.

    Raises
    ------
    ValueError
        If dimensions don't match or coordinates differ.
    """
    spatial_dims = list(spatial_da.dims)

    if not set(spatial_dims).issubset(set(data.dims)):
        missing_dims = set(spatial_dims) - set(data.dims)
        raise ValueError(
            f"Data is missing spatial dimensions from {name}: {missing_dims}. "
            f"Data dims: {data.dims}, {name} dims: {spatial_dims}."
        )

    non_spatial_dims = [d for d in data.dims if d not in spatial_dims]
    template = data.isel({d: 0 for d in non_spatial_dims}) if non_spatial_dims else data

    # Skip dims that have no coord on either side (bare-dim arrays). When exactly
    # one side has a coord, validate_matching_coordinates raises informatively.
    checkable = [
        d for d in spatial_dims if d in spatial_da.coords or d in template.coords
    ]
    if not checkable:
        return

    validate_matching_coordinates(
        spatial_da,
        template,
        checkable,
        left_name=name,
        right_name="data",
        rtol=rtol,
        atol=atol,
    )


def validate_mask(
    mask: xr.DataArray,
    data: xr.DataArray,
    mask_name: str = "mask",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Validate that a mask matches data spatial dimensions and coordinates.

    Parameters
    ----------
    mask : xarray.DataArray
        Mask to validate. Must have boolean dtype, or integer dtype with exactly one
        non-zero value (0 = background, one region id = foreground). The latter format
        is produced by [`Atlas.get_masks`][confusius.atlas.Atlas.get_masks]. Coordinates
        must match data.
    data : xarray.DataArray
        Data array to validate mask against.
    mask_name : str, default: "mask"
        Name of the mask parameter (used in error messages).
    rtol : float, default: 1e-5
        Relative tolerance for coordinate comparison.
    atol : float, default: 1e-8
        Absolute tolerance for coordinate comparison.

    Raises
    ------
    TypeError
        If `mask` is not a boolean or single-label integer DataArray.
    ValueError
        If `mask` dimensions don't match `data` or if coordinates don't match.
    """
    if not isinstance(mask, xr.DataArray):
        raise TypeError(
            f"{mask_name} must be an xarray.DataArray, got {type(mask).__name__}."
        )

    if mask.dtype == bool:
        pass
    elif np.issubdtype(mask.dtype, np.integer):
        non_zero = np.unique(mask.values[mask.values != 0])
        if len(non_zero) > 1:
            raise TypeError(
                f"{mask_name} has integer dtype with {len(non_zero)} distinct non-zero "
                f"values. A mask must be boolean or have exactly one non-zero label "
                f"(0 = background, one region id = foreground). "
                f"For multi-region extraction use extract_with_labels instead."
            )
    else:
        raise TypeError(
            f"{mask_name} must be boolean dtype or a single-label integer dtype, "
            f"got {mask.dtype}."
        )

    _validate_spatial_coords(mask, data, mask_name, rtol, atol)


def validate_labels(
    labels: xr.DataArray,
    data: xr.DataArray,
    labels_name: str = "labels",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Validate that a label map matches data spatial dimensions and coordinates.

    Parameters
    ----------
    labels : xarray.DataArray
        Label map to validate. Must have integer dtype and coordinates must match data.
        Accepts two formats:

        - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `mask` dimension followed by
          spatial dims, e.g. `(mask, z, y, x)`. Each layer has values in `{0,
          region_id}` and regions may overlap. The `region` coordinate of the
          output holds the `mask` coordinate values (e.g., region label).

    data : xarray.DataArray
        Data array to validate labels against.
    labels_name : str, default: "labels"
        Name of the labels parameter (used in error messages).
    rtol : float, default: 1e-5
        Relative tolerance for coordinate comparison.
    atol : float, default: 1e-8
        Absolute tolerance for coordinate comparison.

    Raises
    ------
    TypeError
        If `labels` is not an integer dtype DataArray.
    ValueError
        If `labels` dimensions don't match `data` or if coordinates don't match.
    """
    if not isinstance(labels, xr.DataArray):
        raise TypeError(
            f"{labels_name} must be an xarray.DataArray, got {type(labels).__name__}."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{labels_name} must be integer dtype, got {labels.dtype}.")

    # For stacked format, validate spatial dims only (drop the mask axis).
    spatial_labels = labels.isel(mask=0, drop=True) if "mask" in labels.dims else labels
    _validate_spatial_coords(spatial_labels, data, labels_name, rtol, atol)
