"""Shared helpers for ConfUSIus I/O modules."""

import json
import warnings
from typing import Any

import numpy as np

from confusius._utils.stack import find_stack_level

ZARR_V3_CONSOLIDATED_METADATA_WARNING = (
    "Consolidated metadata is currently not part in the Zarr format 3 specification."
)
"""Zarr v3 warning text emitted when consolidated metadata is written."""


def convert_to_json_serializable(value: Any) -> Any:
    """Recursively convert numpy containers and scalars to native Python objects.

    Parameters
    ----------
    value : Any
        Attribute value, possibly a numpy array or scalar or a `dict`/`list`/`tuple`
        nesting them.

    Returns
    -------
    Any
        Equivalent value with every `numpy.ndarray` replaced by a nested list and every
        numpy scalar replaced by its Python counterpart. Non-numpy values are returned
        unchanged.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: convert_to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [convert_to_json_serializable(item) for item in value]
    return value


def make_attrs_zarr_safe(attrs: dict[str, Any]) -> dict[str, Any]:
    """Make attributes safe to store as Zarr attributes.

    Zarr stores attributes as JSON. Numpy arrays and scalars, including those nested
    inside dicts or lists such as `attrs["affines"]`, are converted to native Python
    objects. Any remaining value that cannot be JSON-encoded (e.g. matplotlib colormap
    and normalization objects on atlas-derived data) is dropped with a warning.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attributes to sanitize.

    Returns
    -------
    dict[str, Any]
        Copy of `attrs` with numpy values converted to native Python and
        non-JSON-serializable entries removed.
    """
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in attrs.items():
        converted = convert_to_json_serializable(value)
        try:
            json.dumps(converted)
        except TypeError:
            dropped.append(key)
        else:
            safe[key] = converted

    if dropped:
        warnings.warn(
            f"Dropping non-JSON-serializable attrs from Zarr store: {dropped}.",
            stacklevel=find_stack_level(),
        )

    return safe


def restore_affines_in_attrs(attrs: dict[str, Any]) -> None:
    """Restore `attrs["affines"]` dict values to numpy arrays in place.

    Zarr stores the affines as nested lists (see
    [`make_attrs_zarr_safe`][confusius.io._utils.make_attrs_zarr_safe]); this converts
    them back to numpy arrays so a Zarr round-trip matches the NIfTI and SCAN loaders. A
    no-op when `affines` is absent or not a dict.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attributes to update in place.

    Returns
    -------
    None
        This function mutates `attrs` and returns nothing.
    """
    affines = attrs.get("affines")
    if not isinstance(affines, dict):
        return
    attrs["affines"] = {key: np.asarray(value) for key, value in affines.items()}
