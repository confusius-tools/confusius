"""Xarray extensions for fUSI data analysis."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "FUSIAccessor",
    "FUSIConnectivityAccessor",
    "FUSIIQAccessor",
    "FUSIPlotAccessor",
    "FUSIRegistrationAccessor",
    "FUSIScaleAccessor",
    "TransformCenter",
    "apply_affine",
    "apply_flip",
    "apply_rotation",
    "apply_translation",
    "apply_zoom",
    "create_voxeldata",
    "db_scale",
    "get_bounding_box",
    "get_geometry_center",
    "get_relative_affine",
    "log_scale",
    "power_scale",
    "reindex_voxels",
    "reindex_voxels_like",
]

from confusius.xarray.accessors import FUSIAccessor

_ATTR_TO_MODULE = {
    "create_voxeldata": "confusius.xarray.create",
    "FUSIConnectivityAccessor": "confusius.xarray.connectivity",
    "FUSIIQAccessor": "confusius.xarray.iq",
    "FUSIPlotAccessor": "confusius.xarray.plotting",
    "FUSIRegistrationAccessor": "confusius.xarray.registration",
    "FUSIScaleAccessor": "confusius.xarray.scale",
    "db_scale": "confusius.xarray.scale",
    "log_scale": "confusius.xarray.scale",
    "power_scale": "confusius.xarray.scale",
    "TransformCenter": "confusius.xarray.affine",
    "apply_affine": "confusius.xarray.affine",
    "apply_flip": "confusius.xarray.affine",
    "apply_rotation": "confusius.xarray.affine",
    "apply_translation": "confusius.xarray.affine",
    "apply_zoom": "confusius.xarray.affine",
    "get_bounding_box": "confusius.xarray.affine",
    "get_geometry_center": "confusius.xarray.affine",
    "get_relative_affine": "confusius.xarray.affine",
    "reindex_voxels": "confusius.xarray.affine",
    "reindex_voxels_like": "confusius.xarray.affine",
}


# SPEC-0001 recommends PEP 562-based lazy loading for public namespaces.
def __getattr__(name: str) -> Any:
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'confusius.xarray' has no attribute {name!r}")

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from confusius.xarray.affine import (
        TransformCenter,
        apply_affine,
        apply_flip,
        apply_rotation,
        apply_translation,
        apply_zoom,
        get_bounding_box,
        get_geometry_center,
        get_relative_affine,
        reindex_voxels,
        reindex_voxels_like,
    )
    from confusius.xarray.connectivity import FUSIConnectivityAccessor
    from confusius.xarray.create import create_voxeldata
    from confusius.xarray.iq import FUSIIQAccessor
    from confusius.xarray.plotting import FUSIPlotAccessor
    from confusius.xarray.registration import FUSIRegistrationAccessor
    from confusius.xarray.scale import (
        FUSIScaleAccessor,
        db_scale,
        log_scale,
        power_scale,
    )

# Accessor is registered automatically via @xr.register_dataarray_accessor decorator.
