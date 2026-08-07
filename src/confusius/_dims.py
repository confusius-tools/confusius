"""ConfUSIus dimension name conventions.

All xarray DataArrays produced by ConfUSIus use a fixed set of dimension names. This
module defines them as constants so that IO, processing, validation, and UI code can
reference them without duplicating string literals.
"""

SPATIAL_DIMS: tuple[str, ...] = ("z", "y", "x")
"""Physical spatial dimension names in ConfUSIus order."""

VOXEL_DIMS: tuple[str, ...] = ("k", "j", "i")
"""Native voxel-space dimension names in ConfUSIus order."""

POSE_DIM: str = "pose"
"""Dimension name for discrete probe positions in multi-pose acquisitions."""

TIME_DIM: str = "time"
"""Default signal (x-axis) dimension name."""
