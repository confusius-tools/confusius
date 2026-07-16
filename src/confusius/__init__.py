"""Python package for analysis and visualization of functional ultrasound imaging data."""

from importlib import import_module, metadata
from typing import TYPE_CHECKING, Any

__all__ = [
    "atlas",
    "connectivity",
    "decomposition",
    "datasets",
    "extract",
    "glm",
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
    "stats",
    "timing",
    "validation",
    "xarray",
    "__version__",
]

__version__ = metadata.version("confusius")

# Import lightweight accessor-registration modules eagerly so `import confusius`
# continues to register the `.fusi` and `.atlas` accessors on xarray objects.
from confusius import atlas, xarray

_SUBMODULES = {
    "atlas",
    "connectivity",
    "decomposition",
    "datasets",
    "extract",
    "glm",
    "io",
    "iq",
    "multipose",
    "plotting",
    "qc",
    "registration",
    "signal",
    "spatial",
    "stats",
    "timing",
    "validation",
}

_ATTR_TO_MODULE = {
    "load": "confusius.io.loadsave",
    "save": "confusius.io.loadsave",
}

# Purge submodules cached by Python's import machinery so reload() resets lazy state.
for _name in _SUBMODULES:
    globals().pop(_name, None)
del _name


# SPEC-0001 recommends PEP 562-based lazy loading for top-level namespaces.
def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return import_module(f"confusius.{name}")

    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is not None:
        return getattr(import_module(module_name), name)

    raise AttributeError(f"module 'confusius' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from confusius import (
        atlas,
        connectivity,
        datasets,
        decomposition,
        extract,
        glm,
        io,
        iq,
        multipose,
        plotting,
        qc,
        registration,
        signal,
        spatial,
        stats,
        timing,
        validation,
        xarray,
    )
    from confusius.io.loadsave import load, save
