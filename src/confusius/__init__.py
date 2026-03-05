"""Python package for analysis and visualization of functional ultrasound imaging data."""

__all__ = [
    "atlas",
    "extract",
    "io",
    "iq",
    "qc",
    "multipose",
    "plotting",
    "registration",
    "signal",
    "validation",
    "xarray",
    "__version__",
]

from importlib import metadata

__version__ = metadata.version("confusius")

from confusius import (
    atlas,
    extract,
    io,
    iq,
    multipose,
    plotting,
    qc,
    registration,
    signal,
    validation,
    xarray,
)
