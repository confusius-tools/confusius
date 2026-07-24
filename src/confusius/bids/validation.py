"""Pydantic models for fUSI-BIDS metadata validation.

This module provides Pydantic models for validating fUSI-BIDS sidecar metadata, ensuring
compliance with the fUSI-BIDS specification.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from confusius._utils.stack import find_stack_level


class FUSIBIDSMetadata(BaseModel):
    """FUSI-BIDS metadata model for sidecar JSON validation.

    This model represents the fUSI-BIDS sidecar metadata as defined in the fUSI-BIDS
    extension proposal (BEP) v0.0.12.

    Notes
    -----
    At least one of `RepetitionTime` or `VolumeTiming` must be provided for valid
    fUSI-BIDS timing information.

    Examples
    --------
    >>> metadata = FUSIBIDSMetadata(
    ...     TaskName="rest",
    ...     RepetitionTime=1.5,
    ...     Manufacturer="Verasonics",
    ... )
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
        arbitrary_types_allowed=True,
    )

    # Scanner and probe hardware
    Manufacturer: str | None = Field(
        default=None,
        description="Manufacturer of the ultrasound system.",
    )
    ManufacturersModelName: str | None = Field(
        default=None,
        description="Manufacturer's model name of the ultrasound system.",
    )
    DeviceSerialNumber: list[str] | str | None = Field(
        default=None,
        description="Serial number of the acquisition device.",
    )
    StationName: str | None = Field(
        default=None,
        description="User defined name of the ultrasound system.",
    )
    SoftwareVersion: str | None = Field(
        default=None,
        description="Version of the ultrasound system software.",
    )
    ProbeManufacturer: str | None = Field(
        default=None,
        description="Manufacturer of the ultrasound probe.",
    )
    ProbeType: str | None = Field(
        default=None,
        description="Type of the ultrasound probe (e.g., 'linear', 'RCA').",
    )
    ProbeModel: str | None = Field(
        default=None,
        description="Model name of the ultrasound probe.",
    )
    ProbeSerialNumber: list[str] | str | None = Field(
        default=None,
        description="Serial number of the ultrasound probe.",
    )
    ProbeCentralFrequency: float | None = Field(
        default=None,
        gt=0,
        description="Central frequency of the probe in hertz.",
    )
    ProbeNumberOfElements: int | tuple[int, int] | None = Field(
        default=None,
        description="Number of elements in the probe array. For 1D probes, an integer. For 2D probes, a tuple (azimuth, elevation).",
    )
    ProbePitch: float | tuple[float, float] | None = Field(
        default=None,
        description="Distance between probe elements in millimeters. For 1D probes, a float. For 2D probes, a tuple (azimuth, elevation).",
    )
    ProbeRadiusOfCurvature: float | None = Field(
        default=None,
        description="Radius of curvature for curved probes in millimeters.",
    )
    ProbeFocalWidth: float | None = Field(
        default=None,
        gt=0,
        description="Focal width of the probe in millimeters.",
    )
    ProbeFocalDepth: float | None = Field(
        default=None,
        gt=0,
        description="Focal depth of the probe in millimeters.",
    )
    ProbeAperture: float | None = Field(
        default=None,
        gt=0,
        description="Aperture size of the probe in millimeters.",
    )

    # Sequence specifics
    Depth: tuple[float, float] | None = Field(
        default=None,
        description="Minimal and maximal imaging depth of the field of view in millimeters, given as (minimal_depth, maximal_depth).",
    )
    UltrasoundTransmitFrequency: float | None = Field(
        default=None,
        gt=0,
        description="Transmit frequency in hertz.",
    )
    UltrasoundPulseRepetitionFrequency: float | None = Field(
        default=None,
        gt=0,
        description="Pulse repetition frequency in hertz.",
    )
    PlaneWaveAngles: list[float] | list[list[float]] | np.ndarray | None = Field(
        default=None,
        description="Plane wave transmission angles in degrees. For 1D probes, a list of angles. For 2D matrix probes, a list of [azimuth, elevation] pairs.",
    )
    VirtualSources: int | None = Field(
        default=None,
        gt=0,
        description="Number of virtual sources for synthetic aperture.",
    )
    CompoundSamplingFrequency: float | None = Field(
        default=None,
        gt=0,
        description="Compound sampling frequency in hertz.",
    )
    ProbeVoltage: float | None = Field(
        default=None,
        ge=0,
        description="Transmit voltage of the probe in volts.",
    )
    SequenceName: str | None = Field(
        default=None,
        description="Name of the acquisition sequence.",
    )
    TransmitAperture: list[int] | list[list[int]] | None = Field(
        default=None,
        description="Start and end index in each dimension for transmitting elements. For instance, [[10, 90], [10, 40]] would indicate masking the first and last ten elements of a 100 x 50 matrix probe.",
    )
    ReceiveAperture: list[int] | list[list[int]] | None = Field(
        default=None,
        description="Start and end index in each dimension for receiving elements. For instance, [[10, 90], [10, 40]] would indicate masking the first and last ten elements of a 100 x 50 matrix probe.",
    )

    # Beamforming
    BeamformingMethod: str | None = Field(
        default=None,
        description="Beamforming method used.",
    )
    BeamformingSoundVelocity: float | None = Field(
        default=None,
        gt=0,
        description="Speed of sound assumed during beamforming in meters per second.",
    )

    # Clutter filtering
    ClutterFilterWindowDuration: float | None = Field(
        default=None,
        gt=0,
        description="Duration of the clutter filter window in seconds.",
    )
    ClutterFilterWindowStride: float | None = Field(
        default=None,
        gt=0,
        description="Stride of the clutter filter window in seconds.",
    )
    ClutterFilters: str | list[str] | None = Field(
        default=None,
        description="List of clutter filter specifications.",
    )

    # Power Doppler integration.
    PowerDopplerIntegrationDuration: float | None = Field(
        default=None,
        gt=0,
        description="Duration of Power Doppler integration in seconds.",
    )
    PowerDopplerIntegrationStride: float | None = Field(
        default=None,
        gt=0,
        description="Stride of Power Doppler integration in seconds.",
    )

    # Timing parameters.
    VolumeTiming: list[float] | np.ndarray | None = Field(
        default=None,
        description="Onset time of each volume in seconds.",
    )
    RepetitionTime: float | None = Field(
        default=None,
        gt=0,
        description="Time in seconds between the start of consecutive volume acquisitions.",
    )
    SliceTiming: list[float] | np.ndarray | None = Field(
        default=None,
        description="Time at which each slice was acquired within each volume.",
    )
    SliceEncodingDirection: Literal["i", "j", "k", "i-", "j-", "k-"] | None = Field(
        default=None,
        description="Direction of slice acquisition.",
    )
    DelayTime: float | None = Field(
        default=None,
        ge=0,
        description="User specified time in seconds to delay the acquisition.",
    )
    FrameAcquisitionDuration: float | None = Field(
        default=None,
        gt=0,
        description="Duration of the acquisition in seconds.",
    )
    DelayAfterTrigger: float | None = Field(
        default=None,
        ge=0,
        description="Delay after trigger in seconds (required for pose entity).",
    )

    # Task information.
    TaskName: str | None = Field(
        default=None,
        description="Name of the task.",
    )
    TaskDescription: str | None = Field(
        default=None,
        description="Description of the task.",
    )
    CogAtlasID: str | None = Field(
        default=None,
        description="URI of the corresponding Cognitive Atlas Task term.",
    )
    CogPOID: str | None = Field(
        default=None,
        description="URI of the corresponding CogPO term.",
    )

    # Institution information.
    InstitutionName: str | None = Field(
        default=None,
        description="Name of the institution.",
    )
    InstitutionAddress: str | None = Field(
        default=None,
        description="Address of the institution.",
    )
    InstitutionalDepartmentName: str | None = Field(
        default=None,
        description="Name of the institutional department.",
    )

    @field_validator(
        "VolumeTiming",
        "SliceTiming",
        "PlaneWaveAngles",
        mode="before",
    )
    @classmethod
    def convert_numpy_array(cls, v: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization.

        Parameters
        ----------
        v : Any
            Value to convert. If a numpy array, converted to list.

        Returns
        -------
        Any
            List if input was numpy array, otherwise input unchanged.
        """
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @model_validator(mode="after")
    def validate_timing(self) -> FUSIBIDSMetadata:
        """Validate timing field constraints per fUSI-BIDS spec.

        Enforces:
        - `RepetitionTime` and `VolumeTiming` are mutually exclusive.
        - `FrameAcquisitionDuration` is REQUIRED when `VolumeTiming` is used AND
          `SliceTiming` is not set.
        - `FrameAcquisitionDuration` is mutually exclusive with `RepetitionTime`.

        Returns
        -------
        FUSIBIDSMetadata
            Self, if validation passes.
        """
        # Only warn about missing timing if any time-related fields are present. For
        # pure 3D data without time, no timing fields are expected.
        has_any_timing = any(
            [
                self.RepetitionTime is not None,
                self.VolumeTiming is not None,
                self.SliceTiming is not None,
                self.SliceEncodingDirection is not None,
                self.DelayTime is not None,
                self.FrameAcquisitionDuration is not None,
                self.DelayAfterTrigger is not None,
            ]
        )
        if has_any_timing and self.RepetitionTime is None and self.VolumeTiming is None:
            warnings.warn(
                "fUSI-BIDS recommends providing either RepetitionTime or VolumeTiming "
                "for timing information.",
                stacklevel=find_stack_level(),
            )

        if self.RepetitionTime is not None and self.VolumeTiming is not None:
            raise ValueError(
                "RepetitionTime and VolumeTiming are mutually exclusive. "
                "Provide one or the other, not both."
            )

        if (
            self.VolumeTiming is not None
            and self.SliceTiming is None
            and self.FrameAcquisitionDuration is None
        ):
            raise ValueError(
                "FrameAcquisitionDuration is REQUIRED when VolumeTiming is used and SliceTiming is not set."
            )

        if (
            self.RepetitionTime is not None
            and self.FrameAcquisitionDuration is not None
        ):
            raise ValueError(
                "FrameAcquisitionDuration must not be provided when RepetitionTime is specified."
            )

        return self

    @model_validator(mode="after")
    def validate_slice_timing_consistency(self) -> FUSIBIDSMetadata:
        """Validate that `SliceTiming` and `SliceEncodingDirection` are consistent.

        Returns
        -------
        FUSIBIDSMetadata
            Self, if validation passes.
        """
        if self.SliceTiming is not None and self.SliceEncodingDirection is None:
            warnings.warn(
                "SliceTiming is provided but SliceEncodingDirection is missing. "
                "SliceEncodingDirection is required to interpret SliceTiming.",
                stacklevel=find_stack_level(),
            )
        return self


def validate_metadata(metadata: Mapping[str, Any]) -> FUSIBIDSMetadata:
    """Validate fUSI-BIDS metadata dictionary.

    Parameters
    ----------
    metadata : Mapping[str, Any]
        Dictionary containing fUSI-BIDS metadata fields.

    Returns
    -------
    FUSIBIDSMetadata
        Validated metadata model.

    Raises
    ------
    ValidationError
        If the metadata fails validation.

    Examples
    --------
    >>> metadata = validate_metadata({
    ...     "TaskName": "rest",
    ...     "RepetitionTime": 1.5,
    ... })
    >>> metadata.TaskName
    'rest'
    """
    return FUSIBIDSMetadata.model_validate(metadata)


def format_validation_error(error: ValidationError) -> str:
    """Format a Pydantic ValidationError into a concise, readable message.

    Extracts just the essential error information without the verbose details
    like input values, types, or URLs.

    Parameters
    ----------
    error : ValidationError
        The Pydantic validation error to format.

    Returns
    -------
    str
        A formatted error message with just the error locations and messages.

    Examples
    --------
    >>> try:
    ...     validate_metadata({"Manufacturer": "Test"})
    ... except ValidationError as e:
    ...     msg = format_validation_error(e)
    """
    error_messages = []
    for err in error.errors():
        location = ".".join(str(part) for part in err.get("loc", ()))
        if location:
            error_messages.append(f"  - {location}: {err['msg']}")
        else:
            error_messages.append(f"  - {err['msg']}")
    return "\n".join(error_messages)


FUSI_BIDS_FIELDS: frozenset[str] = frozenset(FUSIBIDSMetadata.model_fields.keys())
"""Set of all fUSI-BIDS field names derived from the Pydantic model.

This is the single source of truth for field names.
"""
