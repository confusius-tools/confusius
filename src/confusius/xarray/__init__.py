"""Xarray extensions for fUSI data analysis."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "FUSIAccessor",
    "FUSIAffineAccessor",
    "FUSIConnectivityAccessor",
    "FUSIIQAccessor",
    "FUSIPlotAccessor",
    "FUSIRegistrationAccessor",
    "FUSIScaleAccessor",
    "FUSIExtractAccessor",
    "db_scale",
    "log_scale",
    "power_scale",
]

from confusius.xarray.accessors import FUSIAccessor

_ATTR_TO_MODULE = {
    "FUSIAffineAccessor": "confusius.xarray.affine",
    "FUSIConnectivityAccessor": "confusius.xarray.connectivity",
    "FUSIExtractAccessor": "confusius.xarray.extract",
    "FUSIIQAccessor": "confusius.xarray.iq",
    "FUSIPlotAccessor": "confusius.xarray.plotting",
    "FUSIRegistrationAccessor": "confusius.xarray.registration",
    "FUSIScaleAccessor": "confusius.xarray.scale",
    "db_scale": "confusius.xarray.scale",
    "log_scale": "confusius.xarray.scale",
    "power_scale": "confusius.xarray.scale",
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
    from confusius.xarray.affine import FUSIAffineAccessor
    from confusius.xarray.connectivity import FUSIConnectivityAccessor
    from confusius.xarray.extract import FUSIExtractAccessor
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
