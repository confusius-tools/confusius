"""Python package for analysis and visualization of functional ultrasound imaging data."""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

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

__version__ = metadata.version("confusius")

_SUBMODULES = {
    "atlas",
    "connectivity",
    "decomposition",
    "datasets",
    "extract",
    "io",
    "iq",
    "multipose",
    "plotting",
    "qc",
    "registration",
    "signal",
    "spatial",
    "timing",
    "validation",
    "xarray",
}

_ATTR_TO_MODULE = {
    "load": "confusius.io.loadsave",
    "save": "confusius.io.loadsave",
}


# SPEC-0001 recommends PEP 562-based lazy loading for top-level namespaces.
def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"confusius.{name}")
        globals()[name] = module
        return module

    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is not None:
        module = import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'confusius' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
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
