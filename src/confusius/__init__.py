"""Python package for analysis and visualization of functional ultrasound imaging data."""

__all__ = [
    "atlas",
    "connectivity",
    "extract",
    "io",
    "iq",
    "load",
    "save",
    "qc",
    "multipose",
    "plotting",
    "registration",
    "signal",
    "spatial",
    "validation",
    "xarray",
    "__version__",
]

from importlib import metadata

__version__ = metadata.version("confusius")

from confusius import (
    atlas,
    connectivity,
    extract,
    io,
    iq,
    multipose,
    plotting,
    qc,
    registration,
    signal,
    spatial,
    validation,
    xarray,
)
from confusius.io.loadsave import load, save
