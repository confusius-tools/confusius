"""Registration module for fUSI data."""

from confusius.registration.affines import compose_affine, decompose_affine
from confusius.registration.bspline import (
    invert_displacement_field,
    sample_displacement_field,
    sample_displacement_field_like,
)
from confusius.registration.diagnostics import RegistrationDiagnostics
from confusius.registration.exceptions import RegistrationAbortedError
from confusius.registration.motion import (
    compute_framewise_displacement,
    create_motion_dataframe,
    extract_motion_parameters,
)
from confusius.registration.progress import (
    MatplotlibRegistrationProgressPlotter,
    RegistrationProgress,
)
from confusius.registration.resampling import (
    resample_like,
    resample_volume,
)
from confusius.registration.volume import register_volume
from confusius.registration.volumewise import register_volumewise
from confusius.registration.volumewise_progress import VolumewiseProgressReporter

__all__ = [
    "MatplotlibRegistrationProgressPlotter",
    "RegistrationAbortedError",
    "RegistrationDiagnostics",
    "RegistrationProgress",
    "VolumewiseProgressReporter",
    "compose_affine",
    "compute_framewise_displacement",
    "create_motion_dataframe",
    "decompose_affine",
    "extract_motion_parameters",
    "invert_displacement_field",
    "register_volume",
    "register_volumewise",
    "resample_like",
    "resample_volume",
    "sample_displacement_field",
    "sample_displacement_field_like",
]
