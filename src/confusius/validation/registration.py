"""Validation helpers for registration transform DataArrays."""

import xarray as xr

from confusius._utils.geometry import get_voxel_to_world_spatial_dims
from confusius.validation.fusi import validate_fusi


def validate_bspline(da: xr.DataArray) -> None:
    """Raise ValueError if `da` is not a valid B-spline transform DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If `da.attrs["transform_type"] != "bspline_transform"`, required attrs are
        missing, or `da` is not a canonical ConfUSIus voxel-grid DataArray.
    """
    transform_type = da.attrs.get("transform_type", da.attrs.get("type"))
    if transform_type != "bspline_transform":
        raise ValueError(
            "Expected a DataArray with attrs['transform_type'] == "
            "'bspline_transform'; "
            f"got {transform_type!r}."
        )
    if "order" not in da.attrs:
        raise ValueError(
            "B-spline transform DataArray is missing required attribute 'order'."
        )
    if da.dims[0] != "component":
        raise ValueError(
            f"B-spline transform DataArray must have 'component' as its first "
            f"dimension; got {da.dims[0]!r}."
        )
    validate_fusi(
        da,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=True,
    )
    spatial_dims = get_voxel_to_world_spatial_dims(da)
    if da.sizes["component"] != len(spatial_dims):
        raise ValueError(
            "B-spline transform DataArray component count must match its spatial "
            f"dimensionality; got {da.sizes['component']} components for "
            f"{len(spatial_dims)} spatial dimensions."
        )


def validate_displacement_field(da: xr.DataArray) -> None:
    """Raise ValueError if `da` is not a valid displacement field DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to validate.

    Raises
    ------
    ValueError
        If `da.attrs["type"] != "displacement_field_transform"`, `da` does not have
        `"component"` as its first dimension, or `da` is not a canonical ConfUSIus
        voxel-grid DataArray.
    """
    if da.attrs.get("type") != "displacement_field_transform":
        raise ValueError(
            "Expected a DataArray with attrs['type'] == 'displacement_field_transform'; "
            f"got {da.attrs.get('type')!r}."
        )
    if da.dims[0] != "component":
        raise ValueError(
            f"Displacement field DataArray must have 'component' as its first "
            f"dimension; got {da.dims[0]!r}."
        )
    validate_fusi(
        da,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=True,
    )
    spatial_dims = get_voxel_to_world_spatial_dims(da)
    if da.sizes["component"] != len(spatial_dims):
        raise ValueError(
            "Displacement field DataArray component count must match its spatial "
            f"dimensionality; got {da.sizes['component']} components for "
            f"{len(spatial_dims)} spatial dimensions."
        )
