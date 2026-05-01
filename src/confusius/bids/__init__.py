"""FUSI-BIDS utilities for ConfUSIus.

This module provides utilities for working with fUSI-BIDS metadata, including conversion
between ConfUSIus and fUSI-BIDS naming conventions, validation of metadata, and
coordinate handling.

Examples
--------
>>> from confusius import bids
>>>
>>> # Convert ConfUSIus attributes to BIDS format
>>> attrs = {"repetition_time": 1.5, "task_name": "rest", "transmit_frequency": 15e6}
>>> bids_attrs = bids.to_bids(attrs)
>>> bids_attrs
{"RepetitionTime": 1.5, "TaskName": "rest", "UltrasoundTransmitFrequency": 15000000.0}
>>>
>>> # Convert BIDS metadata back to ConfUSIus format
>>> attrs_back = bids.from_bids(bids_attrs)
>>> attrs_back
{"repetition_time": 1.5, "task_name": "rest", "transmit_frequency": 15000000.0}
>>>
>>> # Validate fUSI-BIDS metadata
>>> metadata = bids.validate_metadata({
...     "TaskName": "rest",
...     "RepetitionTime": 1.5,
... })
"""

from confusius.bids.coordinates import (
    DIM_TO_SLICE_ENCODING_DIRECTION,
    SLICE_ENCODING_DIRECTION_TO_DIM,
    create_bids_slice_timing_from_coordinate,
    create_slice_time_coordinate_from_bids,
)
from confusius.bids.mapping import (
    EXPLICIT_BIDS_FIELD_MAPPINGS,
    from_bids,
    to_bids,
)
from confusius.bids.validation import (
    FUSI_BIDS_FIELDS,
    FUSIBIDSMetadata,
    validate_metadata,
)

__all__ = [
    # Mapping functions
    "to_bids",
    "from_bids",
    "FUSI_BIDS_FIELDS",
    "EXPLICIT_BIDS_FIELD_MAPPINGS",
    # Validation
    "FUSIBIDSMetadata",
    "validate_metadata",
    # Coordinates
    "SLICE_ENCODING_DIRECTION_TO_DIM",
    "DIM_TO_SLICE_ENCODING_DIRECTION",
    "create_bids_slice_timing_from_coordinate",
    "create_slice_time_coordinate_from_bids",
]
