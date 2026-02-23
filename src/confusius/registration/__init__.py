"""Registration module for fUSI data."""

from confusius.registration._progress import RegistrationProgressPlotter
from confusius.registration.affines import (
    compose_affine,
    decompose_affine,
)
from confusius.registration.motion import (
    compute_framewise_displacement,
    create_motion_dataframe,
    extract_motion_parameters,
)
from confusius.registration.resampling import (
    resample_like,
    resample_volume,
)
from confusius.registration.volume import register_volume
from confusius.registration.volumewise import register_volumewise

__all__ = [
    "RegistrationProgressPlotter",
    "compose_affine",
    "decompose_affine",
    "register_volume",
    "resample_volume",
    "resample_like",
    "register_volumewise",
    "extract_motion_parameters",
    "compute_framewise_displacement",
    "create_motion_dataframe",
]
