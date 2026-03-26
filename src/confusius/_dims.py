"""ConfUSIus dimension name conventions.

All xarray DataArrays produced by ConfUSIus use a fixed set of dimension names. This
module defines them as constants so that IO, processing, validation, and UI code can
reference them without duplicating string literals.
"""

SPATIAL_DIMS: tuple[str, ...] = ("z", "y", "x")
"""Spatial dimension names in ConfUSIus order (elevation, axial, lateral)."""

POSE_DIM: str = "pose"
"""Dimension name for discrete probe positions in multi-pose acquisitions."""

SPATIAL_DIMS_WITH_POSE: tuple[str, ...] = (POSE_DIM, *SPATIAL_DIMS)
"""All dimensions that represent physical space, including pose."""

TIME_DIM: str = "time"
"""Default signal (x-axis) dimension name."""
