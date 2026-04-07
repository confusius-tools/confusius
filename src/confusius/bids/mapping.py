"""FUSI-BIDS field mapping between ConfUSIus and BIDS naming conventions.

This module provides bidirectional conversion between ConfUSIus internal (snake_case)
attribute names and fUSI-BIDS standard (PascalCase) metadata field names.

The Pydantic model in [`confusius.bids.validation`][confusius.bids.validation] is the
single source of truth for field definitions.
"""

import re
from collections.abc import Mapping
from typing import Final

from confusius.bids.validation import FUSI_BIDS_FIELDS


def _snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase.

    Parameters
    ----------
    name : str
        snake_case name.

    Returns
    -------
    str
        PascalCase name.

    Examples
    --------
    >>> _snake_to_pascal("repetition_time")
    'RepetitionTime'
    >>> _snake_to_pascal("probe_serial_number")
    'ProbeSerialNumber'
    """
    return "".join(part.capitalize() for part in name.split("_"))


def _pascal_to_snake(name: str) -> str:
    """Convert PascalCase to snake_case.

    Handles acronyms correctly by inserting underscores before word boundaries.

    Parameters
    ----------
    name : str
        PascalCase name.

    Returns
    -------
    str
        snake_case name.

    Examples
    --------
    >>> _pascal_to_snake("RepetitionTime")
    'repetition_time'
    >>> _pascal_to_snake("CogAtlasID")
    'cog_atlas_id'
    """
    # Add an underscore before each uppercase letter that is followed by a lowercase
    # letter.
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    # Add an underscore before each lowercase letter that is preceded by an uppercase
    # letter.
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


EXPLICIT_BIDS_FIELD_MAPPINGS: Final[dict[str, str]] = {
    "transmit_frequency": "UltrasoundTransmitFrequency",
    "pulse_repetition_frequency": "UltrasoundPulseRepetitionFrequency",
    "volume_acquisition_duration": "FrameAcquisitionDuration",
}
"""Explicit mappings for standard BIDS fields with non-automatic names.

Maps ConfUSIus internal names to fUSI-BIDS field names.
"""

_REVERSE_EXPLICIT_BIDS_FIELD_MAPPINGS: Final[dict[str, str]] = {
    v: k for k, v in EXPLICIT_BIDS_FIELD_MAPPINGS.items()
}
"""Reverse explicit mappings for loading standard BIDS fields."""

CONFUSIUS_INTERNAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "sform_code",
        "qform_code",
        "affines",
        "units",
        "voxdim",
        "long_name",
        "cmap",
        "bmode_integration_duration",
        "bmode_integration_stride",
        "axial_velocity_integration_duration",
        "axial_velocity_integration_stride",
        "axial_velocity_lag",
        "axial_velocity_absolute",
        "axial_velocity_spatial_kernel",
        "axial_velocity_estimation_method",
    }
)
"""ConfUSIus-only fields that should be prefixed with `ConfUSIus` in BIDS.

These are stored with ConfUSIus prefix in PascalCase in the sidecar.
"""

_CONFUSIUS_INTERNAL_TO_BIDS: Final[dict[str, str]] = {
    key: f"ConfUSIus{_snake_to_pascal(key)}" for key in CONFUSIUS_INTERNAL_FIELDS
}
"""Mappings from ConfUSIus-only internal fields to prefixed BIDS keys."""

_BIDS_TO_CONFUSIUS_INTERNAL: Final[dict[str, str]] = {
    value: key for key, value in _CONFUSIUS_INTERNAL_TO_BIDS.items()
}
"""Reverse mappings from prefixed BIDS keys to ConfUSIus-only internal fields."""


def to_bids(attrs: Mapping[str, object]) -> dict[str, object]:
    """Convert ConfUSIus attributes to fUSI-BIDS format.

    Only converts known fUSI-BIDS fields. Internal ConfUSIus attributes are prefixed
    with "ConfUSIus". Unknown fields are preserved as-is.

    Parameters
    ----------
    attrs : Mapping[str, object]
        Dictionary with ConfUSIus (snake_case) attribute names.

    Returns
    -------
    dict[str, object]
        Dictionary with fUSI-BIDS (PascalCase) field names.

    Examples
    --------
    >>> attrs = {"repetition_time": 1.5, "task_name": "rest", "custom_field": 123}
    >>> to_bids(attrs)
    {'RepetitionTime': 1.5, 'TaskName': 'rest', 'custom_field': 123}
    """
    bids_attrs: dict[str, object] = {}

    for key, value in attrs.items():
        if key in CONFUSIUS_INTERNAL_FIELDS:
            bids_attrs[_CONFUSIUS_INTERNAL_TO_BIDS[key]] = value
        elif key in EXPLICIT_BIDS_FIELD_MAPPINGS:
            # Use explicit mapping for fields that don't follow automatic conversion.
            bids_attrs[EXPLICIT_BIDS_FIELD_MAPPINGS[key]] = value
        else:
            pascal_key = _snake_to_pascal(key)

            if pascal_key in FUSI_BIDS_FIELDS:
                bids_attrs[pascal_key] = value
            else:
                bids_attrs[key] = value

    return bids_attrs


def from_bids(bids_attrs: Mapping[str, object]) -> dict[str, object]:
    """Convert fUSI-BIDS metadata to ConfUSIus format.

    Known fUSI-BIDS fields are converted to snake_case. ConfUSIus-prefixed attributes
    are converted back to internal names. Unknown fields are preserved as-is to ensure
    round-trip safety.

    Parameters
    ----------
    bids_attrs : Mapping[str, object]
        Dictionary with fUSI-BIDS (PascalCase) field names.

    Returns
    -------
    dict[str, object]
        Dictionary with ConfUSIus (snake_case) attribute names.

    Examples
    --------
    >>> bids_attrs = {"RepetitionTime": 1.5, "TaskName": "rest", "CustomField": 123}
    >>> from_bids(bids_attrs)
    {'repetition_time': 1.5, 'task_name': 'rest', 'CustomField': 123}
    """
    attrs: dict[str, object] = {}

    for key, value in bids_attrs.items():
        if key in _BIDS_TO_CONFUSIUS_INTERNAL:
            attrs[_BIDS_TO_CONFUSIUS_INTERNAL[key]] = value
        elif key in _REVERSE_EXPLICIT_BIDS_FIELD_MAPPINGS:
            # Use reverse explicit mapping for fields that don't follow automatic
            # conversion.
            attrs[_REVERSE_EXPLICIT_BIDS_FIELD_MAPPINGS[key]] = value
        elif key in FUSI_BIDS_FIELDS:
            attrs[_pascal_to_snake(key)] = value
        else:
            attrs[key] = value

    return attrs
