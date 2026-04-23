"""Python package for analysis and visualization of functional ultrasound imaging data."""

__all__ = [
    "atlas",
    "connectivity",
    "decomposition",
    "datasets",
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
    "timing",
    "validation",
    "xarray",
    "__version__",
]

from importlib import metadata

__version__ = metadata.version("confusius")

from confusius import (
    atlas,
    connectivity,
    decomposition,
    datasets,
    extract,
    io,
    iq,
    multipose,
    plotting,
    qc,
    registration,
    signal,
    spatial,
    timing,
    validation,
    xarray,
)
from confusius.io.loadsave import load, save
