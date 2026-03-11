"""Xarray extensions for fUSI data analysis."""

__all__ = [
    "FUSIAccessor",
    "FUSIAffineAccessor",
    "FUSIConnectivityAccessor",
    "FUSIIOAccessor",
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
from confusius.xarray.affine import FUSIAffineAccessor
from confusius.xarray.connectivity import FUSIConnectivityAccessor
from confusius.xarray.extract import FUSIExtractAccessor
from confusius.xarray.io import FUSIIOAccessor
from confusius.xarray.iq import FUSIIQAccessor
from confusius.xarray.plotting import FUSIPlotAccessor
from confusius.xarray.registration import FUSIRegistrationAccessor
from confusius.xarray.scale import FUSIScaleAccessor, db_scale, log_scale, power_scale

# Accessor is registered automatically via @xr.register_dataarray_accessor decorator.
