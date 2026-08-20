"""Data validation utilities for confusius."""

from confusius.validation.atlas import validate_atlas
from confusius.validation.coordinates import validate_matching_coordinates
from confusius.validation.fusi import (
    canonicalize_voxeldata,
    ensure_voxeldata,
    validate_voxeldata,
)
from confusius.validation.mask import (
    ensure_labels,
    ensure_mask,
    validate_labels,
    validate_mask,
)
from confusius.validation.registration import (
    validate_bspline,
    validate_displacement_field,
)
from confusius.validation.time_series import validate_time_series
from confusius.validation.units import validate_matching_spatial_units

__all__ = [
    "canonicalize_voxeldata",
    "ensure_labels",
    "ensure_mask",
    "ensure_voxeldata",
    "validate_atlas",
    "validate_bspline",
    "validate_displacement_field",
    "validate_labels",
    "validate_mask",
    "validate_matching_coordinates",
    "validate_matching_spatial_units",
    "validate_time_series",
    "validate_voxeldata",
]
