"""Utilities for loading and saving NIfTI files.

This module provides functions to load NIfTI neuroimaging files as lazy
VoxelData arrays using nibabel's proxy arrays and Dask for out-of-core
processing. Following the VoxelData model, data is stored with native voxel dimensions
`(..., time, k, j, i)` and VoxelToWorldIndex-derived world coordinates `z`, `y`, `x`.
"""

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr
from pydantic import ValidationError

from confusius._dims import SPATIAL_DIMS, VOXEL_DIMS
from confusius._utils.coordinates import (
    get_coordinate_spacing_info,
    get_representative_step,
)
from confusius._utils.geometry import (
    get_voxel_to_world_affine,
    get_voxel_to_world_index_spacing,
)
from confusius._utils.stack import find_stack_level
from confusius.bids import (
    create_bids_slice_timing_from_coordinate,
    create_slice_time_coordinate_from_bids,
    from_bids,
    to_bids,
)
from confusius.bids.mapping import CONFUSIUS_INTERNAL_FIELDS
from confusius.bids.validation import format_validation_error, validate_metadata
from confusius.io.utils import check_path
from confusius.registration.affines import decompose_affine
from confusius.timing import (
    TIMING_REFERENCE_FACTORS,
    VolumeAcquisitionReference,
    convert_time_reference,
    convert_time_units,
)
from confusius.validation import ensure_voxeldata
from confusius.xarray.create import create_voxeldata

if TYPE_CHECKING:
    import nibabel as nib

NiftiVersion: TypeAlias = Literal[1, 2]
"""Type alias for NIfTI file format version."""

_NIFTI_DIM_ORDER = ("i", "j", "k", "time", "dim4", "dim5", "dim6")
"""ConfUSIus and NIfTI dimension mapping.

Maps ConfUSIus dimension names to their NIfTI axis order. NIfTI axes 0/1/2 (the
standard spatial axis slots) are named with the native voxel dims `i`/`j`/`k`
directly -- `load_nifti` never names a raw axis `x`/`y`/`z`; naming world
coordinates is `create_voxeldata`'s job, not the NIfTI loader's.
"""

_NIFTI_TO_CONFUSIUS_SPACE_UNITS: dict[str, str] = {
    "meter": "m",
    "mm": "mm",
    "micron": "um",
}
"""Mapping from NIfTI spatial unit strings to ConfUSIus conventions."""

_NIFTI_TO_CONFUSIUS_TIME_UNITS: dict[str, str] = {
    "sec": "s",
    "msec": "ms",
    "usec": "us",
}
"""Mapping from NIfTI time unit strings to ConfUSIus conventions."""

_CONFUSIUS_TO_NIFTI_SPACE_UNITS: dict[str, str] = {
    v: k for k, v in _NIFTI_TO_CONFUSIUS_SPACE_UNITS.items()
}
"""Mapping from ConfUSIus spatial unit strings to NIfTI conventions."""

_CONFUSIUS_TO_NIFTI_TIME_UNITS: dict[str, str] = {
    v: k for k, v in _NIFTI_TO_CONFUSIUS_TIME_UNITS.items()
}
"""Mapping from ConfUSIus time unit strings to NIfTI conventions."""

_TIME_ATTRS_TO_SECONDS: frozenset[str] = frozenset(
    {
        "clutter_filter_window_duration",
        "clutter_filter_window_stride",
        "power_doppler_integration_duration",
        "power_doppler_integration_stride",
        "axial_velocity_integration_duration",
        "axial_velocity_integration_stride",
        "bmode_integration_duration",
        "bmode_integration_stride",
    }
)
"""Time-valued processing attrs that are expressed in time-coordinate units."""

_RESOLVABLE_NIFTI_AXES: frozenset[int] = frozenset({4, 5, 6})
"""NIfTI axis indices (0-based) whose dim name can be overridden by the sidecar.

NIfTI axes 0-3 are reserved for `(i, j, k, time)`, so the load side only consults the
sidecar for the 5th, 6th, and 7th NIfTI axes.
"""

_RESERVED_DIM_NAMES: frozenset[str] = frozenset({*VOXEL_DIMS, *SPATIAL_DIMS})
"""Dim names a sidecar `ConfUSIusDim{N}Name` override may never claim.

`k`/`j`/`i` are the native voxel dims and `z`/`y`/`x` are the derived world
coordinate names; both are reserved for the canonical spatial axes so an extra
(non-spatial) axis can never collide with them.
"""

_VOXEL_TO_WORLD_NAME: dict[str, str] = dict(zip(VOXEL_DIMS, SPATIAL_DIMS, strict=True))
"""Maps each native voxel dim name (`k`/`j`/`i`) to its world coordinate name."""

_NIFTI_EXTRA_DIM_NAMES: tuple[str, ...] = ("dim4", "dim5", "dim6")
"""Generic dim names for non-time, non-spatial extra axes in NIfTI order.

A NIfTI file can hold up to 7 axes: `(x, y, z, time, dim4, dim5, dim6)`. The last
three are the fallback names used when a sidecar does not declare a more
specific dim name (e.g. `"component"` for a B-spline control grid).
"""

_MAX_NIFTI_EXTRA_DIMS: int = len(_NIFTI_EXTRA_DIM_NAMES)
"""Maximum number of non-time, non-spatial extra dimensions NIfTI can store."""

_EXTRA_DIM_ATTR_KEYS: frozenset[str] = frozenset(
    key
    for key in CONFUSIUS_INTERNAL_FIELDS
    if key.startswith("dim") and key.endswith(("_name", "_coordinates", "_attrs"))
)
"""Sidecar attribute keys consumed when reconstructing extra-dim metadata."""

_TEMPORAL_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "volume_timing",
        "repetition_time",
        "delay_after_trigger",
        "delay_time",
        "volume_acquisition_duration",
        "slice_timing",
        "slice_encoding_direction",
    }
)
"""Metadata keys whose presence enables temporal coordinate reconstruction."""


class _NiftiHeaderExtractor:
    """Extract relevant metadata from NIfTI header."""

    def __init__(
        self,
        header: "nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header",
    ) -> None:
        self.header = header

    def get_voxel_dimensions(self) -> dict[str, float]:
        """Get voxel dimensions (pixdim) in their native header units.

        Returns
        -------
        dict[str, float]
            Voxel dimensions keyed by ConfUSIus dimension name (`"x"`, `"y"`,
            `"z"`), in the units declared by the NIfTI header (see
            `get_unit_strings`). For headers without a valid affine, the raw
            signed `pixdim` values are used.
        """
        pixdim = np.asarray(self.header["pixdim"], dtype=float)
        nifti_spatial = [("x", 1), ("y", 2), ("z", 3)]
        return {name: float(pixdim[i]) for name, i in nifti_spatial if i < len(pixdim)}

    def get_repetition_time(self) -> float | None:
        """Get repetition time (TR) in its native header units.

        Returns
        -------
        float or None
            Repetition time in the units declared by the NIfTI header (see
            `get_unit_strings`), or `None` if not available. A non-positive
            `pixdim[4]` is treated as "TR not set" and yields `None` so the
            caller can fall back to a default sampling grid instead of an
            all-zeros time coordinate.
        """
        zooms = self.header.get_zooms()
        assert len(zooms) >= 4
        tr = float(zooms[3])
        return tr if tr > 0 else None

    def get_unit_strings(self) -> tuple[str | None, str | None]:
        """Get spatial and temporal unit strings in ConfUSIus conventions.

        Returns
        -------
        space_unit : str or None
            Spatial unit string (`"m"`, `"mm"`, or `"um"`), or `None`
            if the header declares unknown units.
        time_unit : str or None
            Temporal unit string (`"s"`, `"ms"`, or `"us"`), or `None`
            if the header declares unknown units.
        """
        space_nib, time_nib = self.header.get_xyzt_units()
        return (
            _NIFTI_TO_CONFUSIUS_SPACE_UNITS.get(space_nib),
            _NIFTI_TO_CONFUSIUS_TIME_UNITS.get(time_nib),
        )

    def to_attrs(self) -> dict[str, Any]:
        """Convert header information to attributes dictionary."""
        _, sform_code = self.header.get_sform(coded=True)
        _, qform_code = self.header.get_qform(coded=True)

        attrs: dict[str, Any] = {}

        # Code > 0 indicates a valid affine; code 0 means unknown.
        if sform_code > 0:
            attrs["sform_code"] = int(sform_code)
        if qform_code > 0:
            attrs["qform_code"] = int(qform_code)

        return attrs


def _select_affines(
    header: "nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header",
    *,
    coordinate_affine: Literal["auto", "sform", "qform"] = "auto",
) -> tuple[npt.NDArray[np.floating] | None, npt.NDArray[np.floating] | None]:
    """Select primary and secondary affine matrices from a NiBabel header.

    Sform is preferred over qform when both codes are positive. When both are
    valid, the qform is returned as the secondary affine so that scanner-space
    coordinates can be stored alongside the template-space primary coordinates.
    When both codes are zero, `(None, None)` is returned.

    Parameters
    ----------
    header : nibabel.nifti1.Nifti1Header or nibabel.nifti2.Nifti2Header
        NiBabel NIfTI header.

    Returns
    -------
    primary_affine : (4, 4) numpy.ndarray or None
        Primary affine for computing spatial coordinates, or `None` when both
        sform and qform codes are zero.
    secondary_affine : (4, 4) numpy.ndarray or None
        Qform affine when both sform and qform codes are positive; `None` otherwise.
    """
    sform, sform_code = header.get_sform(coded=True)
    qform, qform_code = header.get_qform(coded=True)
    sform_valid = sform_code > 0 and sform is not None
    qform_valid = qform_code > 0 and qform is not None

    if coordinate_affine == "sform":
        if not sform_valid:
            return None, None
        return sform, (qform if qform_valid else None)
    if coordinate_affine == "qform":
        if not qform_valid:
            return None, None
        return qform, (sform if sform_valid else None)

    if sform_valid:
        return sform, (qform if qform_valid else None)
    elif qform_valid:
        return qform, None
    else:
        return None, None


def _load_nifti_sidecar(path: Path) -> dict[str, Any]:
    """Load and validate a BIDS JSON sidecar for a NIfTI file.

    Looks for a `.json` file with the same stem as `path`. For `.nii.gz` files, the
    sidecar is `stem.json` (e.g. `sub-01_bold.json` for `sub-01_bold.nii.gz`).

    Parameters
    ----------
    path : pathlib.Path
        Path to the NIfTI file (`.nii` or `.nii.gz`).

    Returns
    -------
    dict[str, Any]
        Sidecar attributes converted to ConfUSIus (snake_case) naming via `from_bids`.
        Empty dict when no sidecar is found.
    """
    sidecar_path = path.with_suffix("").with_suffix(".json")
    if not sidecar_path.exists():
        return {}

    with open(sidecar_path) as f:
        sidecar_data = json.load(f)

    if sidecar_data:
        try:
            validate_metadata(sidecar_data)
        except ValidationError as e:
            warnings.warn(
                f"fUSI-BIDS validation warning:\n{format_validation_error(e)}",
                stacklevel=find_stack_level(),
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"fUSI-BIDS validation warning: {e}", stacklevel=find_stack_level()
            )

    return from_bids(sidecar_data)


def _load_nifti_with_nibabel(
    path: Path,
) -> tuple[
    "nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image",
    "_NiftiHeaderExtractor",
]:
    """Load a NIfTI file and return the image object with its header extractor.

    Parameters
    ----------
    path : pathlib.Path
        Path to the NIfTI file (`.nii` or `.nii.gz`).

    Returns
    -------
    img : nib.nifti1.Nifti1Image or nib.nifti2.Nifti2Image
        NiBabel NIfTI image object with proxy data array.
    extractor : _NiftiHeaderExtractor
        Header extractor for the loaded image.
    """
    import nibabel as nib

    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image):
        raise ValueError(  # noqa: TRY004
            "Only NIfTI-1 and NIfTI-2 formats are supported when loading files with"
            " .nii or .nii.gz suffixes."
        )

    return img, _NiftiHeaderExtractor(img.header)


def _create_spatial_coords_from_nifti(
    img: "nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image",
    extractor: "_NiftiHeaderExtractor",
    dims: tuple[str, ...],
    *,
    coordinate_affine: Literal["auto", "sform", "qform"] = "auto",
) -> tuple[
    npt.NDArray[np.floating],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Resolve spatial geometry from a NIfTI image's header affine.

    Selects the primary affine (sform preferred over qform when both are valid)
    and derives the full voxel-to-world affine plus per-axis metadata, ready to
    pass straight to `create_voxeldata` as `voxel_to_world=`/
    `world_coord_attrs=`. When a valid affine is available, `voxel_to_world`
    carries it exactly (any rotation/shear included) -- no axis-aligned-only
    decomposition. The world coordinate frame is defined by the primary
    affine's translation and zoom; every affine (primary and secondary) is
    re-expressed against that shared frame, so a secondary qform keeps its
    relationship to the world coordinates.

    When both sform and qform codes are zero, `voxel_to_world` is built from
    pixdim only (origin 0, step = voxel size). A warning is emitted and no
    `"affines"` entry is added to `extra_attrs`.

    Parameters
    ----------
    img : nib.nifti1.Nifti1Image or nib.nifti2.Nifti2Image
        Loaded NiBabel NIfTI image.
    extractor : _NiftiHeaderExtractor
        Header extractor for `img`.
    dims : tuple[str, ...]
        Dimension names in NIfTI order `(i, j, k[, time, ...])`.
    coordinate_affine : {"auto", "sform", "qform"}, default: "auto"
        Header affine to use as the primary coordinate-defining geometry.

    Returns
    -------
    voxel_to_world : (4, 4) numpy.ndarray
        Primary voxel-to-world affine in ConfUSIus world `z`/`y`/`x` row and
        native voxel `k`/`j`/`i` column order.
    world_coord_attrs : dict[str, dict[str, Any]]
        `units` attributes keyed by world coordinate name (`"z"`, `"y"`, `"x"`), for
        each spatial dim present in `dims`.
    extra_attrs : dict[str, Any]
        Affine-derived DataArray attributes. Contains `"affines"` when a valid
        secondary affine is present; empty otherwise.
    """
    voxel_sizes = extractor.get_voxel_dimensions()
    space_unit, _ = extractor.get_unit_strings()
    primary_affine, secondary_affine = _select_affines(
        img.header,
        coordinate_affine=coordinate_affine,
    )

    # (native voxel dim name, world coordinate name), in NIfTI axis order
    # (0, 1, 2). `voxel_sizes`/`img.shape` stay keyed/indexed by NIfTI's own
    # (x, y, z) axis order since that reflects the header layout, not a
    # DataArray dim name.
    axes = (("i", "x"), ("j", "y"), ("k", "z"))

    world_coord_attrs: dict[str, dict[str, Any]] = {}
    extra_attrs: dict[str, Any] = {}

    if primary_affine is None:
        # Both sform_code and qform_code are 0: no spatial orientation encoded.
        # Coordinates are built from pixdim only (origin 0, step = voxel size).
        warnings.warn(
            "Both sform_code and qform_code are 0 in the NIfTI header. Coordinates "
            "will be computed from the voxel dimensions only, which may not reflect "
            "the true spatial orientation of the data.",
            stacklevel=find_stack_level(),
        )
        nifti_affine = np.eye(4, dtype=np.float64)
        for col, (voxel_dim, header_dim) in enumerate(axes):
            if voxel_dim not in dims:
                continue
            coord_attrs: dict[str, Any] = {}
            if space_unit is not None:
                coord_attrs["units"] = space_unit
            step = voxel_sizes.get(header_dim, 1.0)
            world_coord_attrs[_VOXEL_TO_WORLD_NAME[voxel_dim]] = coord_attrs
            nifti_affine[col, col] = step
        voxel_to_world = nifti_affine[[2, 1, 0, 3]][:, [2, 1, 0, 3]]
        return voxel_to_world, world_coord_attrs, extra_attrs

    # Determine which form is primary the same way `_select_affines` did, so the
    # secondary form (when present) gets the correct name -- an explicit
    # coordinate_affine="qform" makes sform the *secondary* form, not qform.
    _, sform_code = img.header.get_sform(coded=True)
    sform_valid = sform_code > 0
    if coordinate_affine == "sform":
        primary_prefix = "sform"
    elif coordinate_affine == "qform":
        primary_prefix = "qform"
    else:
        primary_prefix = "sform" if sform_valid else "qform"
    secondary_prefix = "qform" if primary_prefix == "sform" else "sform"

    # The voxel-to-world index represents the full primary affine (any
    # rotation/shear included) exactly, so it becomes `voxel_to_world`
    # directly -- no axis-aligned-only decomposition, and no separate rotation
    # residual to store for the primary form (unlike an independent-1D-coordinate
    # model, which could only absorb an axis-aligned translation and zoom). A
    # missing spatial dim (e.g. a genuinely 2D scan) needs no special handling
    # either: `create_voxeldata` drops that affine column itself, which is
    # exactly equivalent to evaluating the affine at voxel-index 0 along that
    # axis (index 0 zeroes out that column's contribution).
    #
    # The primary form has no named `attrs["affines"]` entry: it is not a
    # separately tracked relationship, it *is* the array's own world frame,
    # so on save it always mirrors the current `voxel_to_world` (see
    # `_build_selected_nifti_affine`'s `affine_key=None` fallback). Only the
    # secondary form is a genuinely distinct, named relationship, so only it
    # gets one: secondary = world_to_<secondary> @ primary, so
    # world_to_<secondary> = secondary @ inv(primary).
    voxel_to_world = primary_affine[[2, 1, 0, 3]][:, [2, 1, 0, 3]]
    if secondary_affine is not None:
        world_to_secondary = secondary_affine @ np.linalg.inv(primary_affine)
        extra_attrs["affines"] = {
            f"world_to_{secondary_prefix}": world_to_secondary[[2, 1, 0, 3]][
                :, [2, 1, 0, 3]
            ]
        }

    for voxel_dim, header_dim in axes:
        if voxel_dim not in dims:
            continue
        coord_attrs = {}
        if space_unit is not None:
            coord_attrs["units"] = space_unit
        world_coord_attrs[_VOXEL_TO_WORLD_NAME[voxel_dim]] = coord_attrs

    return voxel_to_world, world_coord_attrs, extra_attrs


def _validate_volume_timing_length(
    volume_timing: npt.NDArray[np.floating], *, n_time: int
) -> tuple[npt.NDArray[np.floating] | None, bool]:
    """Validate sidecar `VolumeTiming` against the NIfTI time-axis length.

    Parameters
    ----------
    volume_timing : numpy.ndarray
        1D `VolumeTiming` values from the sidecar.
    n_time : int
        Length of the NIfTI time dimension.

    Returns
    -------
    volume_timing : numpy.ndarray or None
        `volume_timing` unchanged when its shape and length are valid, or `None` when
        the sidecar values cannot be used safely.
    length_mismatch : bool
        Whether the sidecar had a valid 1D shape but the wrong length.
    """
    if volume_timing.ndim != 1 or volume_timing.size == 0:
        warnings.warn(
            "`VolumeTiming` metadata is not a non-empty 1D array. Ignoring it.",
            stacklevel=find_stack_level(),
        )
        return None, False

    if volume_timing.size == n_time:
        return volume_timing, False

    warnings.warn(
        f"`VolumeTiming` length ({volume_timing.size}) does not match the data time "
        f"dimension ({n_time}). Ignoring it.",
        stacklevel=find_stack_level(),
    )
    return None, True


def _create_temporal_coords_from_nifti(
    img: "nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image",
    extractor: "_NiftiHeaderExtractor",
    attrs: dict[str, Any],
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Create temporal coordinate arrays from a NIfTI image and BIDS sidecar fields.

    Builds the `time` coordinate and, when `SliceTiming` is available, the `slice_time`
    coordinate. Timing fields (`VolumeTiming`, `RepetitionTime`, `DelayAfterTrigger`,
    `DelayTime`, `FrameAcquisitionDuration`, `SliceTiming`, `SliceEncodingDirection`)
    are removed from the returned attributes dict.

    The priority for the time coordinate values is:

    1. `VolumeTiming` from the sidecar (irregular timestamps).
    2. `RepetitionTime` (+ optional `DelayAfterTrigger`) from the sidecar.
    3. `pixdim[4]` from the NIfTI header.
    4. Integer indices when no timing information is available.

    Parameters
    ----------
    img : nib.nifti1.Nifti1Image or nib.nifti2.Nifti2Image
        Loaded NiBabel NIfTI image.
    extractor : _NiftiHeaderExtractor
        Header extractor for `img`.
    attrs : dict[str, Any]
        DataArray attributes, typically merged from the NIfTI header and the
        BIDS sidecar.

    Returns
    -------
    coords : dict[str, xarray.DataArray]
        Temporal coordinate DataArrays keyed by name. Always contains `"time"`;
        contains `"slice_time"` when `SliceTiming` and `SliceEncodingDirection`
        are both present in `attrs`.
    remaining_attrs : dict[str, Any]
        Copy of `attrs` with all consumed temporal fields removed.
    """

    attrs = dict(attrs)
    n_time = img.shape[3]
    sampling_period_nifti = extractor.get_repetition_time()
    _, time_unit = extractor.get_unit_strings()

    time_attrs: dict[str, Any] = {}
    if time_unit is not None:
        time_attrs["units"] = time_unit

    # BIDS always uses onset timing.
    time_attrs["volume_acquisition_reference"] = "start"
    if "volume_acquisition_duration" in attrs:
        volume_duration = float(attrs.pop("volume_acquisition_duration"))
        if time_unit is not None:
            volume_duration = float(
                convert_time_units(
                    volume_duration,
                    from_unit="s",
                    to_unit=time_unit,
                    raise_on_unknown=True,
                )
            )
        time_attrs["volume_acquisition_duration"] = volume_duration

    time_values: npt.NDArray[np.floating] | None = None
    volume_timing_length_mismatch = False
    if "volume_timing" in attrs:
        raw_volume_timing = np.asarray(attrs.pop("volume_timing"), dtype=np.float64)
        volume_timing, volume_timing_length_mismatch = _validate_volume_timing_length(
            raw_volume_timing,
            n_time=n_time,
        )
        if volume_timing is not None:
            time_values = volume_timing
            if time_unit is not None:
                time_values = convert_time_units(
                    time_values,
                    from_unit="s",
                    to_unit=time_unit,
                    raise_on_unknown=True,
                )

    if time_values is None and "repetition_time" in attrs:
        sampling_period_sidecar = float(attrs.pop("repetition_time"))
        delay = float(attrs.pop("delay_after_trigger", 0.0))
        delay_time = float(attrs.pop("delay_time", 0.0))
        if time_unit is not None:
            sampling_period_sidecar = float(
                convert_time_units(
                    sampling_period_sidecar,
                    from_unit="s",
                    to_unit=time_unit,
                    raise_on_unknown=True,
                )
            )
            delay = float(
                convert_time_units(
                    delay,
                    from_unit="s",
                    to_unit=time_unit,
                    raise_on_unknown=True,
                )
            )
            delay_time = float(
                convert_time_units(
                    delay_time,
                    from_unit="s",
                    to_unit=time_unit,
                    raise_on_unknown=True,
                )
            )
        if (
            sampling_period_nifti is not None
            and sampling_period_nifti > 0
            and not np.isclose(
                sampling_period_sidecar, sampling_period_nifti, rtol=1e-3
            )
        ):
            warnings.warn(
                f"Sidecar RepetitionTime ({sampling_period_sidecar}) does not match "
                f"pixdim[4] ({sampling_period_nifti}) in the NIfTI header. Using "
                "sidecar value.",
                stacklevel=find_stack_level(),
            )
        if "volume_acquisition_duration" not in time_attrs:
            volume_duration = sampling_period_sidecar - delay_time
            if volume_duration > 0:
                time_attrs["volume_acquisition_duration"] = volume_duration
            elif delay_time > 0:
                warnings.warn(
                    "DelayTime is greater than or equal to RepetitionTime, so "
                    "`time.attrs['volume_acquisition_duration']` cannot be inferred.",
                    stacklevel=find_stack_level(),
                )
                # `create_voxeldata` backfills a missing duration from the time
                # coordinate's own regular spacing (the repetition time) -- exactly
                # the guess just declined above as unsafe. This private marker tells
                # `load_nifti` to strip that backfilled value again after
                # construction, rather than silently keeping a value known to be
                # wrong.
                time_attrs["_duration_uninferable"] = True
        time_values = delay + sampling_period_sidecar * np.arange(n_time)
    elif time_values is None and sampling_period_nifti is not None:
        if volume_timing_length_mismatch:
            warnings.warn(
                f"Ignoring mismatched `VolumeTiming`; using NIfTI header `pixdim[4]` "
                f"({sampling_period_nifti}) for timing.",
                stacklevel=find_stack_level(),
            )
        time_values = sampling_period_nifti * np.arange(n_time)
    elif time_values is None:
        if volume_timing_length_mismatch:
            warnings.warn(
                "Ignoring mismatched `VolumeTiming`; NIfTI header `pixdim[4]` is not "
                "positive. Falling back to frame indices.",
                stacklevel=find_stack_level(),
            )
        time_values = np.arange(n_time, dtype=np.float64)

    coords: dict[str, xr.DataArray] = {
        "time": xr.DataArray(time_values, dims=["time"], attrs=time_attrs)
    }

    if "slice_timing" in attrs and "slice_encoding_direction" in attrs:
        coords["slice_time"] = create_slice_time_coordinate_from_bids(
            volume_times=coords["time"].values,
            slice_timing=convert_time_units(
                attrs.pop("slice_timing"),
                from_unit="s",
                to_unit=coords["time"].attrs.get("units", "s"),
                raise_on_unknown=True,
            ),
            slice_encoding_direction=attrs.pop("slice_encoding_direction"),
            units=coords["time"].attrs.get("units", "s"),
        )

    return coords, attrs


def _create_scalar_temporal_coords_from_nifti(
    extractor: "_NiftiHeaderExtractor",
    attrs: dict[str, Any],
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Create scalar temporal coordinates for non-temporal NIfTI payloads.

    When a NIfTI payload has no `time` dimension but the sidecar still carries timing
    metadata (for example after saving a 3D snapshot with a scalar `time`
    coordinate), this helper reconstructs scalar temporal coordinates and removes the
    consumed timing fields from attrs.

    Parameters
    ----------
    extractor : _NiftiHeaderExtractor
        Header extractor for the loaded image.
    attrs : dict[str, Any]
        DataArray attributes, typically merged from the NIfTI header and sidecar.

    Returns
    -------
    coords : dict[str, xarray.DataArray]
        Scalar temporal coordinate DataArrays keyed by name. Empty dict when no scalar
        timing can be reconstructed.
    remaining_attrs : dict[str, Any]
        Copy of `attrs` with consumed temporal fields removed.
    """
    attrs = dict(attrs)
    _, time_unit = extractor.get_unit_strings()

    time_attrs: dict[str, Any] = {"volume_acquisition_reference": "start"}
    if time_unit is not None:
        time_attrs["units"] = time_unit
    if "volume_acquisition_duration" in attrs:
        frame_duration = float(attrs.pop("volume_acquisition_duration"))
        if time_unit is not None:
            frame_duration = float(
                convert_time_units(
                    frame_duration,
                    from_unit="s",
                    to_unit=time_unit,
                    raise_on_unknown=True,
                )
            )
        time_attrs["volume_acquisition_duration"] = frame_duration

    time_value: float | None = None
    if "volume_timing" in attrs:
        volume_timing = np.asarray(attrs.pop("volume_timing"))
        if volume_timing.ndim != 1 or volume_timing.size == 0:
            warnings.warn(
                "`volume_timing` metadata is not a non-empty 1D array. Omitting scalar "
                "`time` coordinate reconstruction.",
                stacklevel=find_stack_level(),
            )
            return {}, attrs
        if volume_timing.size > 1:
            warnings.warn(
                "`volume_timing` has multiple entries but the image has no `time` "
                "dimension. Using the first timestamp for scalar `time`.",
                stacklevel=find_stack_level(),
            )
        time_value = float(volume_timing[0])
    elif "repetition_time" in attrs:
        attrs.pop("repetition_time")
        time_value = float(attrs.pop("delay_after_trigger", 0.0))
        attrs.pop("delay_time", None)
    elif "delay_after_trigger" in attrs:
        time_value = float(attrs.pop("delay_after_trigger"))
        attrs.pop("delay_time", None)

    if time_value is None:
        return {}, attrs

    if time_unit is not None:
        time_value = float(
            convert_time_units(
                time_value,
                from_unit="s",
                to_unit=time_unit,
                raise_on_unknown=True,
            )
        )

    coords: dict[str, xr.DataArray] = {
        "time": xr.DataArray(np.float64(time_value), attrs=time_attrs)
    }

    if "slice_timing" in attrs and "slice_encoding_direction" in attrs:
        slice_encoding_direction = str(attrs["slice_encoding_direction"])
        if slice_encoding_direction.removesuffix("-") not in VOXEL_DIMS:
            return coords, attrs
        attrs.pop("slice_encoding_direction")

        slice_timing = convert_time_units(
            attrs.pop("slice_timing"),
            from_unit="s",
            to_unit=coords["time"].attrs.get("units", "s"),
            raise_on_unknown=True,
        )
        if slice_timing.ndim != 1:
            return coords, attrs
        if slice_encoding_direction.endswith("-"):
            slice_timing = slice_timing[::-1]

        spatial_dim = slice_encoding_direction.removesuffix("-")
        coords["slice_time"] = xr.DataArray(
            time_value + slice_timing,
            dims=[spatial_dim],
            attrs={"units": coords["time"].attrs.get("units", "s")},
        )

    return coords, attrs


def _get_volume_acquisition_reference(
    attrs: dict[str, Any], *, coord_name: str, warn_on_missing: bool = False
) -> VolumeAcquisitionReference:
    """Return a coordinate timing reference, defaulting to onset timing.

    When the reference is missing, ConfUSIus assumes timestamps correspond to the start
    of each acquisition.

    Parameters
    ----------
    attrs : dict[str, Any]
        Coordinate attribute mapping.
    coord_name : str
        Coordinate name used in warning and error messages.
    warn_on_missing : bool, default: False
        Whether to emit a warning when `volume_acquisition_reference` is absent.

    Returns
    -------
    {"start", "center", "end"}
        The validated timing reference. Returns `"start"` when the attribute is
        missing.

    Raises
    ------
    ValueError
        If `volume_acquisition_reference` is present but not one of `"start"`,
        `"center"`, or `"end"`.

    Warns
    -----
    UserWarning
        If `warn_on_missing=True` and the reference attribute is absent.
    """
    reference = attrs.get("volume_acquisition_reference")

    if reference is None:
        if warn_on_missing:
            warnings.warn(
                f"Coordinate '{coord_name}' has no `volume_acquisition_reference` "
                "attribute. Assuming timings correspond to volume acquisition onset.",
                stacklevel=find_stack_level(),
            )
        return "start"

    if reference not in TIMING_REFERENCE_FACTORS:
        raise ValueError(
            f"Unknown {coord_name} volume_acquisition_reference: {reference!r}. "
            "Must be 'start', 'center', or 'end'."
        )

    return reference


def _resolve_nifti_extra_dims(
    nifti_dims: tuple[str, ...], attrs: dict[str, Any]
) -> tuple[
    tuple[str, ...],
    dict[str, npt.NDArray[np.generic]],
    dict[str, dict[str, Any]],
]:
    """Apply sidecar dim-name overrides to the NIfTI axis order.

    For each NIfTI extra-dim axis (4, 5, 6) with a `dim{N}_name` entry in the sidecar
    attrs (where `N` is the 0-based NIfTI axis), the corresponding placeholder name in
    `nifti_dims` is replaced by the sidecar value -- unless that value is a reserved
    canonical spatial name (`k`/`j`/`i`/`z`/`y`/`x`, see `_RESERVED_DIM_NAMES`), in
    which case the override is rejected, a warning is emitted, and the default
    `dim{N}` placeholder name is kept instead.

    Parameters
    ----------
    nifti_dims : tuple[str, ...]
        NIfTI axis order, e.g. `("i", "j", "k", "time", "dim4", "dim5", "dim6")`.
    attrs : dict[str, Any]
        DataArray attributes merged from the NIfTI header and sidecar.

    Returns
    -------
    resolved_dims : tuple[str, ...]
        NIfTI axis order with sidecar dim-name overrides applied.
    extra_coord_values : dict[str, numpy.ndarray]
        Mapping from resolved extra dim name to its coordinate values, sourced
        from `dim{N}_coordinates` in the sidecar.
    extra_coord_attrs : dict[str, dict[str, Any]]
        Mapping from resolved extra dim name to its coordinate attrs, sourced
        from `dim{N}_attrs` in the sidecar.

    Warns
    -----
    UserWarning
        If a sidecar `dim{N}_name` entry names a reserved canonical spatial dim
        (`k`/`j`/`i`/`z`/`y`/`x`). The override is ignored in that case.
    """
    resolved: list[str] = []
    extra_coord_values: dict[str, npt.NDArray[np.generic]] = {}
    extra_coord_attrs: dict[str, dict[str, Any]] = {}

    for nifti_axis, dim_name in enumerate(nifti_dims):
        if nifti_axis in _RESOLVABLE_NIFTI_AXES:
            sidecar_name = attrs.get(f"dim{nifti_axis}_name")
            if sidecar_name is not None and str(sidecar_name) in _RESERVED_DIM_NAMES:
                warnings.warn(
                    f"Sidecar dim{nifti_axis}_name {sidecar_name!r} is a reserved "
                    "canonical spatial dim name and cannot be used for an extra "
                    f"axis. Falling back to the default {dim_name!r} name.",
                    stacklevel=find_stack_level(),
                )
                sidecar_name = None
            if sidecar_name is not None:
                resolved.append(str(sidecar_name))
                sidecar_coords = attrs.get(f"dim{nifti_axis}_coordinates")
                if sidecar_coords is not None:
                    sidecar_coords_array = np.asarray(sidecar_coords)
                    extra_coord_values[str(sidecar_name)] = (
                        sidecar_coords_array.astype(np.float64)
                        if np.issubdtype(sidecar_coords_array.dtype, np.number)
                        else sidecar_coords_array.astype(np.str_)
                    )
                sidecar_attrs = attrs.get(f"dim{nifti_axis}_attrs")
                if isinstance(sidecar_attrs, dict):
                    extra_coord_attrs[str(sidecar_name)] = dict(sidecar_attrs)
                continue
        resolved.append(dim_name)

    return tuple(resolved), extra_coord_values, extra_coord_attrs


def _build_extra_dim_coords(
    nifti_dims: tuple[str, ...],
    extra_coord_values: dict[str, npt.NDArray[np.generic]],
    extra_coord_attrs: dict[str, dict[str, Any]],
    nifti_pixdim: npt.NDArray[np.floating],
    nifti_axis_sizes: tuple[int, ...],
) -> dict[str, xr.DataArray]:
    """Build extra-dim coordinate DataArrays from sidecar or pixdim values.

    For each extra dim in `nifti_dims`:
    - If a sidecar value is present (from `ConfUSIusDimNCoordinates`), use it.
    - Otherwise, build `step * arange(size)` from `nifti_pixdim[N]` when the
      zoom is non-zero. The sign of the zoom is preserved.
    - Otherwise, fall back to `arange(size)` (unit spacing).

    Parameters
    ----------
    nifti_dims : tuple[str, ...]
        Resolved NIfTI axis order, with sidecar dim names applied.
    extra_coord_values : dict[str, numpy.ndarray]
        Mapping from extra dim name to its sidecar-supplied coord values.
    extra_coord_attrs : dict[str, dict[str, Any]]
        Mapping from extra dim name to its sidecar-supplied coordinate attrs.
    nifti_pixdim : numpy.ndarray
        Raw NIfTI `pixdim` array, where `pixdim[N + 1]` is the zoom for the
        NIfTI axis with 0-based index `N`.
    nifti_axis_sizes : tuple[int, ...]
        Size of the data along each NIfTI axis, in NIfTI order
        `(i, j, k, time, *extras)`. Used to size the fallback coord.

    Returns
    -------
    dict[str, xarray.DataArray]
        Coordinate DataArrays keyed by name, for each extra dim in `nifti_dims`.
    """
    coords: dict[str, xr.DataArray] = {}
    for nifti_axis, dim_name in enumerate(nifti_dims):
        if dim_name in VOXEL_DIMS or dim_name == "time":
            continue
        coord_attrs = dict(extra_coord_attrs.get(dim_name, {}))
        if dim_name in extra_coord_values:
            coords[dim_name] = xr.DataArray(
                extra_coord_values[dim_name], dims=[dim_name], attrs=coord_attrs
            )
            continue
        size = nifti_axis_sizes[nifti_axis]
        pixdim_index = nifti_axis + 1
        step = (
            float(nifti_pixdim[pixdim_index])
            if pixdim_index < len(nifti_pixdim)
            else 0.0
        )
        if step != 0.0:
            coord_values = step * np.arange(size, dtype=np.float64)
        else:
            coord_values = np.arange(size, dtype=np.float64)
        coords[dim_name] = xr.DataArray(
            coord_values,
            dims=[dim_name],
            attrs=coord_attrs,
        )
    return coords


def _pop_extra_dim_attrs(attrs: dict[str, Any]) -> None:
    """Remove consumed extra-dim metadata entries from `attrs` in place.

    The `dim{N}_name` and `dim{N}_coordinates` entries are read from the sidecar
    by [`_resolve_nifti_extra_dims`][confusius.io.nifti._resolve_nifti_extra_dims]
    to rename axes and rebuild extra-dim coordinates. Once applied, they have
    no remaining role on the in-memory DataArray and are removed so `attrs`
    only carries user-meaningful metadata.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attribute dictionary to mutate in place.

    Returns
    -------
    None
        This function mutates `attrs` and returns nothing.
    """
    for key in _EXTRA_DIM_ATTR_KEYS:
        attrs.pop(key, None)


def _reverse_chunk_spec(
    chunks: int | tuple[int, ...] | str | None, ndim: int
) -> int | tuple[int, ...] | str | None:
    """Reverse a per-axis Dask chunk specification to match NIfTI's native axis order.

    `load_nifti`'s `chunks` parameter is documented in ConfUSIus (post-transpose) axis
    order, but the Dask array is now built directly from the NIfTI `ArrayProxy` in its
    native (pre-transpose) axis order and transposed afterward. A tuple `chunks`
    specification must therefore be reversed before being passed to `da.from_array` so
    that, once transposed, it lines up with the axes the caller intended.

    Parameters
    ----------
    chunks : int or tuple[int, ...] or str or None
        Chunk specification as documented for `load_nifti`, in ConfUSIus axis order.
    ndim : int
        Number of dimensions of the array being chunked.

    Returns
    -------
    int or tuple[int, ...] or str or None
        `chunks` unchanged if order-independent (`int`, `str`, or `None`), or with its
        top-level axis order reversed if given as a tuple.
    """
    if not isinstance(chunks, tuple):
        return chunks
    if len(chunks) != ndim:
        return chunks
    return chunks[::-1]


def load_nifti(
    path: str | Path,
    chunks: int | tuple[int, ...] | str | None = "auto",
    *,
    coordinate_affine: Literal["auto", "sform", "qform"] = "auto",
) -> xr.DataArray:
    """Load a NIfTI file as a lazy VoxelData array.

    Loads NIfTI files using nibabel's proxy arrays for memory-efficient access, wrapping
    the data in Dask arrays for chunked, parallel processing. The data is transposed to
    ConfUSIus conventions with voxel-space dimensions `(time, k, j, i)` and derived
    world coordinates `z`, `y`, `x`.

    A BIDS-style JSON sidecar file (same name, `.json` extension) is loaded
    automatically when present.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NIfTI file (`.nii` or `.nii.gz`).
    chunks : int or tuple[int, ...] or str or None, default: "auto"
        How to chunk the array. Must be one of the following forms.

        - A blocksize like `1000`.
        - A blockshape like `(1000, 1000)`.
        - Explicit sizes of all blocks along all dimensions like `((1000, 1000,
          500), (400, 400))`.
        - A size in bytes, like `"100 MiB"` which will choose a uniform block-like
          shape.
        - The word `"auto"` to let Dask choose chunk sizes based on heuristics. See
          `dask.array.normalize_chunks` for more details on how chunk sizes are
          determined.
        - `-1` or `None` as a blocksize indicate the size of the corresponding
          dimension.
    coordinate_affine : {"auto", "sform", "qform"}, default: "auto"
        Header affine to use as the primary coordinate-defining geometry.

        - `"auto"` prefers sform when both sform and qform are valid, and falls
          back to qform when only qform is valid.
        - `"sform"` forces the sform to define the in-memory coordinate geometry.
        - `"qform"` forces the qform to define the in-memory coordinate geometry.

        The non-selected valid header affine is still preserved in
        `data_array.attrs["affines"]` as a world-to-world transform.

    Returns
    -------
    xarray.DataArray
        Lazy VoxelData array with voxel-space dimensions in ConfUSIus
        order (`k`, `j`, `i` plus optional `time`) and world coordinates `z`, `y`,
        `x`. Data is wrapped
        in a Dask array for out-of-core computation.

    Notes
    -----
    In memory, the coordinate-defining geometry uses native voxel dimensions `k`, `j`,
    `i` and derived world coordinates `z`, `y`, `x`. `coordinate_affine` controls which
    NIfTI header affine defines that voxel-to-world mapping.

    World-to-world affines are stored in `da.attrs["affines"]`, a dict keyed by
    affine name. Each value is a 4×4 affine in ConfUSIus `(z, y, x)` convention that
    maps **world coordinates** (as stored in `da.coords`) to world-space
    coordinates. Apply as `da.attrs["affines"]["world_to_sform"] @ np.array([pz, py,
    px, 1.0])` to get `[wz, wy, wx, 1]`, where `pz`, `py`, `px` come from
    `da.coords["z"]`, `da.coords["y"]`, `da.coords["x"]` respectively.

    Unlike the NIfTI affine (which maps voxel *indices* to world space), the
    `world_to_*` affines are invariant to any slicing or downsampling because they
    operate on world positions, not grid indices.

    With `coordinate_affine="auto"`, affine selection follows NIfTI conventions:

    - If `sform_code > 0`: sform is used as the primary affine; a
      `"world_to_sform"` entry is written. When `qform_code > 0` as well, a
      `"world_to_qform"` entry is also stored as secondary.
    - Else, if only `qform_code > 0`: qform is used as the primary affine; only
      `"world_to_qform"` is written.
    - If both codes are zero: a warning is emitted, coordinates are built from
      `pixdim` only (origin 0, step = voxel size), and no `"affines"` entry is
      stored in `da.attrs`.

    Choosing `coordinate_affine="sform"` or `"qform"` overrides the automatic
    preference only when that header affine is valid. If the requested affine is not
    present, loading falls back to the same `pixdim`-only path as a file with no valid
    header affine.

    The raw integer form codes are stored as `da.attrs["qform_code"]` and
    `da.attrs["sform_code"]` (only when > 0) so that a save/load roundtrip can
    reproduce the original NIfTI header codes.

    Examples
    --------
    >>> import confusius as cf
    >>> da = cf.io.load_nifti("brain.nii.gz")
    >>> print(da.dims)
    ("time", "k", "j", "i")
    >>> da = cf.io.load_nifti("brain.nii.gz", coordinate_affine="qform")
    """
    path = check_path(path, type="file")

    img, extractor = _load_nifti_with_nibabel(path)

    attrs = extractor.to_attrs()
    attrs.update(_load_nifti_sidecar(path))

    # img.dataobj is a nibabel ArrayProxy, which already exposes .shape, .dtype, and
    # numpy-style __getitem__ slicing (applying any slope/intercept scaling lazily per
    # chunk). Passing it to da.from_array directly keeps the array lazy; converting it
    # through np.asanyarray first would force it into a concrete numpy.memmap, which
    # da.from_array then copies while building the graph.
    # NIfTI stores data with shape (x, y, z, time) in column-major order. ArrayProxy has
    # no .T, so we build the Dask array in that native order and transpose afterward
    # (a metadata-only Dask op) to reach ConfUSIus order (time, k, j, i).
    from nibabel.arrayproxy import ArrayProxy

    proxy = img.dataobj
    assert isinstance(proxy, ArrayProxy)
    dask_arr = da.from_array(
        proxy,
        chunks=_reverse_chunk_spec(chunks, proxy.ndim),
        asarray=False,
        name=False,
        meta=np.empty((), dtype=proxy.dtype),
    ).T

    # NIfTI dim order is (i, j, k, time, ...); ConfUSIus order is the reverse.
    nifti_dims = _NIFTI_DIM_ORDER[: dask_arr.ndim]

    # Apply sidecar names to the anonymous extra NIfTI axes (`dim4`, `dim5`, `dim6`) so
    # downstream code sees semantic dim names instead of placeholders. Names colliding
    # with a reserved canonical spatial name (k/j/i/z/y/x) are rejected with a warning.
    nifti_dims, extra_coord_values, extra_coord_attrs = _resolve_nifti_extra_dims(
        nifti_dims, attrs
    )
    _pop_extra_dim_attrs(attrs)

    voxel_to_world, world_coord_attrs, affine_attrs = _create_spatial_coords_from_nifti(
        img=img,
        extractor=extractor,
        dims=nifti_dims,
        coordinate_affine=coordinate_affine,
    )
    # Merge NIfTI-header-derived affines with any pre-existing affines (e.g. loaded
    # from the sidecar) so sidecar entries like `bspline_initialization` are not
    # clobbered by the qform/sform keys coming from the file header.
    if "affines" in attrs and "affines" in affine_attrs:
        affine_attrs["affines"] = {**attrs["affines"], **affine_attrs["affines"]}
    attrs.update(affine_attrs)

    extra_coords = _build_extra_dim_coords(
        nifti_dims,
        extra_coord_values,
        extra_coord_attrs,
        np.asarray(img.header.structarr["pixdim"], dtype=float),
        tuple(int(s) for s in img.shape),
    )

    has_extra_dims = any(dim not in VOXEL_DIMS and dim != "time" for dim in nifti_dims)
    has_explicit_time_metadata = any(key in attrs for key in _TEMPORAL_METADATA_KEYS)

    # `time_coord` feeds `create_voxeldata`'s `time=` parameter and is only
    # meaningful when `time` is a real dimension; `scalar_time_coord` and
    # `slice_time_coord` are non-dimension coordinates attached after the fact
    # (a scalar `time` has no dimension of its own, and `slice_time` is not one
    # of `create_voxeldata`'s recognized coordinates).
    time_coord: xr.DataArray | None = None
    scalar_time_coord: xr.DataArray | None = None
    slice_time_coord: xr.DataArray | None = None
    if "time" in nifti_dims:
        temporal_coords, attrs = _create_temporal_coords_from_nifti(
            img=img, extractor=extractor, attrs=attrs
        )
        time_coord = temporal_coords["time"]
        slice_time_coord = temporal_coords.get("slice_time")
    else:
        scalar_temporal_coords, attrs = _create_scalar_temporal_coords_from_nifti(
            extractor=extractor, attrs=attrs
        )
        scalar_time_coord = scalar_temporal_coords.get("time")
        slice_time_coord = scalar_temporal_coords.get("slice_time")

    nifti_name = path.with_suffix("").stem if path.suffix == ".gz" else path.stem

    # Build the canonical array in one `create_voxeldata` call, regardless of
    # how many spatial dims are actually present -- it pads any missing k/j/i to a
    # singleton itself, so every loaded array ends up indexed, never a bare
    # DataArray. The singleton-squeeze must happen first, directly on the Dask array
    # and dims bookkeeping, since `create_voxeldata` validates immediately on
    # construction and cannot accept a synthetic singleton `time` axis.
    create_dims = list(nifti_dims[::-1])
    create_data = dask_arr
    if (
        "time" in create_dims
        and dask_arr.shape[create_dims.index("time")] == 1
        and has_extra_dims
        and not has_explicit_time_metadata
    ):
        time_axis = create_dims.index("time")
        create_data = create_data.squeeze(axis=time_axis)
        create_dims.pop(time_axis)
        time_coord = None

    data_array = create_voxeldata(
        create_data,
        dims=create_dims,
        time=time_coord,
        extra_coords=extra_coords,
        voxel_to_world=voxel_to_world,
        world_coord_attrs=world_coord_attrs,
        attrs=attrs,
        name=nifti_name,
    )
    if "time" in data_array.coords and data_array.coords["time"].attrs.pop(
        "_duration_uninferable", False
    ):
        data_array.coords["time"].attrs.pop("volume_acquisition_duration", None)
    if scalar_time_coord is not None:
        data_array = data_array.assign_coords(time=scalar_time_coord)
    if slice_time_coord is not None:
        data_array = data_array.assign_coords(slice_time=slice_time_coord)
    return data_array


def _infer_repetition_time(
    timings: npt.NDArray[np.floating],
) -> tuple[float | None, float]:
    """Infer the repetition time and onset delay from a time coordinate.

    Parameters
    ----------
    timings : ndarray
        Time values. Scalars are treated as a single-element coordinate.

    Returns
    -------
    tr : float or None
        Uniform repetition time (spacing between volumes) when sampling is regular
        within `rtol=1e-5`, or `None` for a single volume or irregular sampling.
    delay : float
        Onset time of the first volume (`timings[0]`).
    """
    timings = np.atleast_1d(timings)
    delay = float(timings[0])
    if len(timings) < 2:
        return None, delay
    diffs = np.diff(timings)
    if np.allclose(diffs, diffs[0], rtol=1e-5):
        return float(diffs[0]), delay
    return None, delay


def _infer_frame_acquisition_duration(
    time_attrs: dict[str, Any],
    time_values: npt.NDArray[np.floating] | None = None,
) -> float | None:
    """Infer `FrameAcquisitionDuration` from time metadata when possible.

    Parameters
    ----------
    time_attrs : dict[str, Any]
        Attributes attached to `data_array.coords["time"]`.
    time_values : numpy.ndarray, optional
        Time values in seconds. When `volume_acquisition_duration` is absent from the
        time coordinate attrs, the median spacing is used as a best-effort
        approximation.

    Returns
    -------
    float or None
        Frame acquisition duration in seconds, or `None` when it cannot be inferred.
    """
    duration = time_attrs.get("volume_acquisition_duration")
    if isinstance(duration, int | float) and duration > 0:
        return float(duration)

    if time_values is not None and len(time_values) >= 2:
        time_step, non_uniform = get_representative_step(time_values)
        if time_step is not None:
            if non_uniform:
                warnings.warn(
                    ".coords['time'].attrs['volume_acquisition_duration'] is missing. "
                    "Approximating it from the median time spacing; this may differ "
                    "from the true per-volume acquisition duration.",
                    stacklevel=find_stack_level(),
                )
            return time_step

    return None


def _extract_nifti_slice_timing_metadata(data_array: xr.DataArray) -> dict[str, Any]:
    """Extract BIDS slice timing metadata from a `slice_time` coordinate.

    `SliceTiming` is exported from either:

    - a 2D absolute `slice_time` coordinate with dims `(time, spatial_dim)` when
      onset-relative offsets are constant across volumes, or
    - a 1D absolute `slice_time` coordinate with dim `(spatial_dim,)` when a scalar
      `time` coordinate is available.

    BIDS cannot represent per-volume variation in slice offsets.

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray containing a `slice_time` coordinate with absolute timestamps.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `SliceTiming` and `SliceEncodingDirection` for BIDS
        export, or an empty dict when the `slice_time` coordinate is not suitable for
        export.
    """
    if "slice_time" not in data_array.coords:
        return {}

    slice_time_coord = data_array.coords["slice_time"]
    if len(slice_time_coord.dims) == 1:
        spatial_dim = slice_time_coord.dims[0]
        if spatial_dim not in VOXEL_DIMS:
            return {}

        if "time" not in data_array.coords:
            warnings.warn(
                "Cannot infer onset-relative SliceTiming from a 1D `slice_time` "
                "coordinate without a `time` coordinate. Omitting BIDS SliceTiming "
                "export.",
                stacklevel=find_stack_level(),
            )
            return {}

        time_values_seconds = np.atleast_1d(
            convert_time_units(
                data_array.coords["time"].values,
                data_array.coords["time"].attrs.get("units"),
                "s",
                raise_on_unknown=True,
            )
        )
        if time_values_seconds.size != 1:
            warnings.warn(
                "A 1D `slice_time` coordinate can only be exported when `time` is "
                "scalar. Use a 2D `(time, spatial_dim)` coordinate for time series "
                "data. Omitting BIDS SliceTiming export.",
                stacklevel=find_stack_level(),
            )
            return {}

        frame_acquisition_duration = _infer_frame_acquisition_duration(
            data_array.coords["time"].attrs,
            time_values_seconds,
        )
        time_reference = _get_volume_acquisition_reference(
            data_array.coords["time"].attrs,
            coord_name="time",
        )
        if frame_acquisition_duration is None:
            warnings.warn(
                "Cannot infer frame acquisition duration for a 1D `slice_time` "
                "coordinate. Omitting BIDS SliceTiming export.",
                stacklevel=find_stack_level(),
            )
            return {}

        volume_onset_seconds = float(
            convert_time_reference(
                time_values_seconds,
                frame_acquisition_duration,
                from_reference=time_reference,
                to_reference="start",
            )[0]
        )
        slice_time_seconds = convert_time_units(
            slice_time_coord.values,
            slice_time_coord.attrs.get("units"),
            "s",
            raise_on_unknown=True,
        )
        slice_duration = slice_time_coord.attrs.get("volume_acquisition_duration")
        slice_reference = slice_time_coord.attrs.get(
            "volume_acquisition_reference", "start"
        )
        if isinstance(slice_duration, int | float) and slice_duration > 0:
            slice_time_seconds = convert_time_reference(
                slice_time_seconds,
                float(slice_duration),
                from_reference=slice_reference,
                to_reference="start",
            )

        return {
            "SliceTiming": (slice_time_seconds - volume_onset_seconds).tolist(),
            "SliceEncodingDirection": spatial_dim,
        }

    if len(slice_time_coord.dims) != 2 or "time" not in slice_time_coord.dims:
        warnings.warn(
            "`slice_time` must be either a 2D coordinate with dims `(time, "
            "spatial_dim)` or a 1D spatial coordinate paired with scalar `time` to "
            "be exported as BIDS SliceTiming. Omitting SliceTiming export.",
            stacklevel=find_stack_level(),
        )
        return {}

    spatial_dims = [dim for dim in slice_time_coord.dims if dim != "time"]
    if len(spatial_dims) != 1 or spatial_dims[0] not in VOXEL_DIMS:
        return {}

    if "time" not in data_array.coords:
        warnings.warn(
            "Cannot infer onset-relative SliceTiming from a 2D `slice_time` "
            "coordinate without a `time` coordinate. Omitting BIDS SliceTiming export.",
            stacklevel=find_stack_level(),
        )
        return {}

    time_values_seconds = convert_time_units(
        data_array.coords["time"].values,
        data_array.coords["time"].attrs.get("units"),
        "s",
        raise_on_unknown=True,
    )
    frame_acquisition_duration = _infer_frame_acquisition_duration(
        data_array.coords["time"].attrs,
        time_values_seconds,
    )
    time_reference = _get_volume_acquisition_reference(
        data_array.coords["time"].attrs,
        coord_name="time",
    )
    if frame_acquisition_duration is None:
        warnings.warn(
            "Cannot infer frame acquisition duration for a 2D `slice_time` "
            "coordinate. Omitting BIDS SliceTiming export.",
            stacklevel=find_stack_level(),
        )
        return {}

    volume_onsets_seconds = convert_time_reference(
        time_values_seconds,
        frame_acquisition_duration,
        from_reference=time_reference,
        to_reference="start",
    )
    slice_time_seconds = convert_time_units(
        slice_time_coord.transpose("time", spatial_dims[0]).values,
        slice_time_coord.attrs.get("units"),
        "s",
        raise_on_unknown=True,
    )
    slice_duration = slice_time_coord.attrs.get("volume_acquisition_duration")
    slice_reference = slice_time_coord.attrs.get(
        "volume_acquisition_reference", "start"
    )
    if isinstance(slice_duration, int | float) and slice_duration > 0:
        slice_time_seconds = convert_time_reference(
            slice_time_seconds,
            float(slice_duration),
            from_reference=slice_reference,
            to_reference="start",
        )
    slice_time_seconds_coord = xr.DataArray(
        slice_time_seconds,
        dims=("time", spatial_dims[0]),
        attrs={"units": "s"},
    )
    try:
        slice_timing, slice_encoding_direction = (
            create_bids_slice_timing_from_coordinate(
                slice_time_seconds_coord,
                volume_onsets_seconds,
            )
        )
    except ValueError:
        warnings.warn(
            "2D `slice_time` varies across time points after converting to "
            "onset-relative offsets. Omitting BIDS SliceTiming because the format "
            "cannot represent per-volume variation.",
            stacklevel=find_stack_level(),
        )
        return {}

    return {
        "SliceTiming": slice_timing.tolist(),
        "SliceEncodingDirection": slice_encoding_direction,
    }


def _build_extra_dim_sidecar_metadata(data_array: xr.DataArray) -> dict[str, Any]:
    """Build sidecar entries for non-time, non-spatial extra dimensions.

    NIfTI's 5th/6th/7th axes are anonymous. To preserve the original DataArray dim names
    across a save/load roundtrip, each extra dim is written under `dim{N}_name` (where
    `N` is the 0-based NIfTI axis, always 4 + extra_index since NIfTI axis 3 is reserved
    for time). The corresponding coordinate values are written under
    `dim{N}_coordinates` only when they cannot be reconstructed from `pixdim` alone—i.e.
    when the coord does not start at 0 with regular spacing. The keys get the
    `ConfUSIus` prefix in the BIDS sidecar (e.g. `ConfUSIusDim4Name`).

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.

    Returns
    -------
    dict[str, Any]
        Mapping from `dim{N}_name` and (when needed) `dim{N}_coordinates` keys to their
        values. Empty when the DataArray has no extra (non-spatial, non-time)
        dimensions.
    """
    current_dims = tuple(str(dim) for dim in data_array.dims)
    extras = [d for d in current_dims if d not in (*VOXEL_DIMS, "time")]

    extra_metadata: dict[str, Any] = {}
    for extra_index, dim_name in enumerate(extras):
        nifti_axis = 4 + extra_index
        extra_metadata[f"dim{nifti_axis}_name"] = dim_name
        if dim_name in data_array.coords:
            coord = data_array.coords[dim_name]
            coord_values = np.asarray(coord.values)
            if not _coord_starts_at_zero_with_regular_spacing(coord_values):
                extra_metadata[f"dim{nifti_axis}_coordinates"] = coord_values.tolist()
            if coord.attrs:
                extra_metadata[f"dim{nifti_axis}_attrs"] = dict(coord.attrs)

    return extra_metadata


def _coord_starts_at_zero_with_regular_spacing(
    coord_values: npt.NDArray[np.generic],
) -> bool:
    """Whether a coord is `[0, step, 2*step, ...]` (recoverable from `pixdim`).

    A coord matching this pattern can be reconstructed on load from the NIfTI
    `pixdim[N]` entry, so storing it in the sidecar would be redundant.

    Parameters
    ----------
    coord_values : numpy.ndarray
        Coordinate values for one dimension.

    Returns
    -------
    bool
        Whether `coord_values` can be reconstructed from a signed spacing and
        an implicit zero origin.
    """
    if len(coord_values) == 0:
        return True
    if not np.issubdtype(coord_values.dtype, np.number):
        return False
    if not np.isclose(coord_values[0], 0.0):
        return False
    if len(coord_values) < 2:
        return True
    numeric_coord_values = coord_values.astype(np.float64)
    step, _ = get_representative_step(numeric_coord_values)
    if step is None or np.isclose(step, 0.0):
        return False
    expected = step * np.arange(len(coord_values))
    return bool(np.allclose(numeric_coord_values, expected, rtol=1e-5, atol=0.0))


def _prepare_data_for_nifti(
    data_array: xr.DataArray,
) -> tuple[np.ndarray, bool, list[str]]:
    """Return array data reordered to NIfTI axis order.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array to serialize.

    Returns
    -------
    data : numpy.ndarray
        Array reordered to NIfTI axis order. The 4th NIfTI axis is present when
        `time` is a real dimension or when a synthetic singleton time axis is needed so non-time
        extra axes land at NIfTI axes 4, 5, 6. Boolean arrays are cast to
        `uint8` because NIfTI does not support `bool` payload dtypes.
    has_time_axis : bool
        Whether the serialized NIfTI payload includes the 4th axis. This is true when
        `time` is a real dimension or when a synthetic singleton time axis is needed to
        keep extra dimensions out of NIfTI's time slot.
    extra_dims : list[str]
        Non-canonical dims from `data_array`, preserving their input order.

    Raises
    ------
    ValueError
        If `data_array` has more than `_MAX_NIFTI_EXTRA_DIMS` non-spatial,
        non-time dimensions. NIfTI supports at most 7 axes total
        (3 spatial + time + 3 extras), so the limit is 3 extras regardless of
        whether `time` is present.
    """
    data = np.asarray(data_array)
    current_dims = tuple(str(dim) for dim in data_array.dims)

    canonical_order = [*reversed(VOXEL_DIMS), "time"]
    extras = [d for d in current_dims if d not in canonical_order]
    if len(extras) > _MAX_NIFTI_EXTRA_DIMS:
        raise ValueError(
            f"Cannot save DataArray with {len(extras)} extra (non-spatial, "
            f"non-time) dimensions to NIfTI: NIfTI supports at most "
            f"{_MAX_NIFTI_EXTRA_DIMS} extra dimensions. Extra dims found: "
            f"{extras!r}."
        )

    target_order = []
    for dim in canonical_order:
        if dim in current_dims:
            target_order.append(current_dims.index(dim))

    for i, dim in enumerate(current_dims):
        if dim not in canonical_order:
            target_order.append(i)

    data = np.transpose(data, target_order)

    nifti_spatial_dims = tuple(reversed(VOXEL_DIMS))
    for insert_pos, dim in enumerate(nifti_spatial_dims):
        if dim not in current_dims:
            data = np.expand_dims(data, axis=insert_pos)

    has_time_axis = "time" in current_dims or bool(extras)
    if has_time_axis and "time" not in current_dims:
        data = np.expand_dims(data, axis=3)

    if np.issubdtype(data.dtype, np.bool_):
        data = data.astype(np.uint8, copy=False)

    return data, has_time_axis, extras


def _get_spatial_spacings(data_array: xr.DataArray) -> list[float]:
    """Return signed spatial spacings for NIfTI header serialization.

    `data_array` is always a VoxelData array here: `save_nifti`
    (the only caller)
    calls `ensure_voxeldata` before reaching this point, so spacing always comes from the
    `VoxelToWorldIndex` -- there is no plain-coordinate fallback to consider.

    Parameters
    ----------
    data_array : xarray.DataArray
        VoxelData array being serialized.

    Returns
    -------
    list[float]
        Signed spatial spacings for the NIfTI `x`, `y`, and `z` axes.
    """
    voxel_spacings = get_voxel_to_world_index_spacing(data_array)
    spacings = [voxel_spacings.get(dim) for dim in reversed(VOXEL_DIMS)]
    return [1.0 if spacing is None else float(spacing) for spacing in spacings]


def _build_nifti_timing_metadata(
    data_array: xr.DataArray,
) -> tuple[dict[str, Any], float | None]:
    """Return timing-related BIDS metadata and the NIfTI temporal zoom.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.

    Returns
    -------
    timing_metadata : dict[str, Any]
        Timing fields to place in the BIDS sidecar, including regular or irregular
        volume timing information and slice timing when available.
    tr_pixdim : float or None
        Temporal zoom to write to the NIfTI header. This is the repetition time for
        regular sampling, `0.0` for irregular sampling, or `None` when there is no time
        coordinate.
    """
    timing_metadata: dict[str, Any] = {}
    tr_pixdim: float | None = None

    if "time" in data_array.coords:
        time_values_raw = data_array.coords["time"].values
        time_attrs = data_array.coords["time"].attrs
        time_unit = time_attrs.get("units")
        warn_on_missing_reference = (
            "volume_acquisition_duration" in time_attrs
            or "slice_time" in data_array.coords
        )
        time_reference = _get_volume_acquisition_reference(
            time_attrs,
            coord_name="time",
            warn_on_missing=warn_on_missing_reference,
        )

        time_values_seconds = convert_time_units(
            time_values_raw,
            time_unit,
            "s",
            raise_on_unknown=True,
        )
        time_values_seconds = np.atleast_1d(time_values_seconds)
        frame_acquisition_duration = _infer_frame_acquisition_duration(
            time_attrs, time_values_seconds
        )
        if frame_acquisition_duration is not None and time_reference != "start":
            time_values_seconds = convert_time_reference(
                time_values_seconds,
                frame_acquisition_duration,
                from_reference=time_reference,
                to_reference="start",
            )

        tr_spacing, delay = _infer_repetition_time(time_values_seconds)
        if tr_spacing is not None:
            tr_pixdim = tr_spacing
            timing_metadata["RepetitionTime"] = tr_spacing
            if not np.isclose(delay, 0.0):
                timing_metadata["DelayAfterTrigger"] = delay
            if frame_acquisition_duration is not None:
                delay_time = tr_spacing - frame_acquisition_duration
                if delay_time > 0 and not np.isclose(delay_time, 0.0):
                    timing_metadata["DelayTime"] = delay_time
        else:
            tr_pixdim = 0.0
            if len(time_values_seconds) >= 2:
                warnings.warn(
                    "Coordinate 'time' has non-uniform sampling. Exact timings are "
                    "saved in the JSON sidecar as VolumeTiming, but the NIfTI "
                    "header's pixdim[4] will be set as 0.0 as it cannot represent "
                    "irregular acquisition times.",
                    stacklevel=find_stack_level(),
                )
            timing_metadata["VolumeTiming"] = time_values_seconds.tolist()
            if frame_acquisition_duration is not None:
                timing_metadata["FrameAcquisitionDuration"] = frame_acquisition_duration

    timing_metadata.update(_extract_nifti_slice_timing_metadata(data_array))

    return timing_metadata, tr_pixdim


def _resolve_nifti_affine_key(
    stored_affines: dict[str, Any],
    *,
    form_name: Literal["qform", "sform"],
    selected_key: str | None,
    default_key: str,
) -> str | None:
    """Resolve the affine key to serialize for a NIfTI xform field.

    Parameters
    ----------
    stored_affines : dict[str, Any]
        Affines stored in `data_array.attrs["affines"]`.
    form_name : {"qform", "sform"}
        Name of the NIfTI xform field being resolved.
    selected_key : str, optional
        Explicit affine key requested by the caller.
    default_key : str
        Fallback affine key to use when `selected_key` is not provided.

    Returns
    -------
    str or None
        Selected affine key, or `None` when neither the explicit key nor the fallback
        key is available.

    Raises
    ------
    ValueError
        If an explicit `selected_key` is requested but is not present in
        `stored_affines`.
    """
    affine_key = selected_key if selected_key is not None else default_key
    if affine_key in stored_affines:
        return affine_key

    if selected_key is not None:
        raise ValueError(
            f"{form_name}={selected_key!r} not found in data_array.attrs['affines']."
        )

    return None


def _resolve_nifti_xform_code(
    data_array: xr.DataArray,
    *,
    form_name: Literal["qform", "sform"],
    code: int | None,
) -> int:
    """Resolve the NIfTI qform/sform code to write.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.
    form_name : {"qform", "sform"}
        Name of the NIfTI xform field being resolved.
    code : int, optional
        Explicit qform/sform code override.

    Returns
    -------
    int
        Resolved xform code. Explicit `code` takes precedence, then the corresponding
        `data_array.attrs["<form_name>_code"]` value when present. Otherwise defaults
        to `2` (`NIFTI_XFORM_ALIGNED_ANAT`), since `voxel_to_world` no longer
        necessarily represents any one specific reference frame (e.g. scanner space)
        once transforms such as registration have been applied.
    """
    if code is not None:
        return code

    attr_name = f"{form_name}_code"
    if attr_name in data_array.attrs:
        return int(data_array.attrs[attr_name])

    return 2


def _build_nifti_voxel_to_world_affine(
    data_array: xr.DataArray,
) -> npt.NDArray[np.floating]:
    """Build the NIfTI voxel-to-world affine for serialized grid geometry."""
    voxel_dims = tuple(dim for dim in VOXEL_DIMS if dim in data_array.dims)
    voxel_indices = [VOXEL_DIMS.index(dim) for dim in voxel_dims]

    index_to_voxel = np.eye(4, dtype=np.float64)
    for dim in voxel_dims:
        axis = VOXEL_DIMS.index(dim)
        coord = np.asarray(data_array.coords[dim].values, dtype=np.float64)
        if coord.size == 0:
            raise ValueError(f"Cannot save empty voxel coordinate {dim!r} to NIfTI.")
        if coord.size == 1:
            start = float(coord[0])
            step = 1.0
        else:
            step, approximate = get_representative_step(coord)
            if approximate or step is None:
                raise ValueError(
                    "Saving voxel-to-world data to NIfTI requires regularly sampled "
                    f"voxel coordinates, but {dim!r} is irregular."
                )
            start = float(coord[0])
        index_to_voxel[axis, axis] = float(step)
        index_to_voxel[axis, 3] = start

    voxel_to_world = get_voxel_to_world_affine(data_array)
    full_voxel_to_world = np.eye(4, dtype=np.float64)
    full_voxel_to_world[np.ix_(voxel_indices, voxel_indices)] = voxel_to_world[:-1, :-1]
    full_voxel_to_world[voxel_indices, 3] = voxel_to_world[:-1, -1]
    confusius_affine = full_voxel_to_world @ index_to_voxel
    return confusius_affine[[2, 1, 0, 3]][:, [2, 1, 0, 3]]


def _build_selected_nifti_affine(
    data_array: xr.DataArray,
    *,
    stored_affines: dict[str, Any],
    affine_key: str | None,
) -> npt.NDArray[np.floating]:
    """Build a NIfTI header affine from voxel geometry plus an optional transform."""
    voxel_to_world = _build_nifti_voxel_to_world_affine(data_array)
    if affine_key is None:
        return voxel_to_world

    transform = _validate_affine_matrix(stored_affines[affine_key], name=affine_key)
    return np.asarray(transform)[[2, 1, 0, 3]][:, [2, 1, 0, 3]] @ voxel_to_world


def _nifti_affine_has_shear(
    affine: npt.NDArray[np.floating], *, atol: float = 1e-8
) -> bool:
    """Whether a NIfTI-order affine contains shear and cannot be stored in qform."""
    _, _, _, shear = decompose_affine(np.asarray(affine, dtype=np.float64))
    return not np.allclose(shear, 0.0, atol=atol)


def _prepare_nifti_xforms(
    data_array: xr.DataArray,
    *,
    qform: str | None,
    sform: str | None,
    qform_code: int | None,
    sform_code: int | None,
) -> tuple[
    dict[str, Any],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
    int,
    int,
    set[str],
]:
    """Prepare qform/sform affines, codes, and sidecar omission keys.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.
    qform : str, optional
        Explicit affine key to use for qform serialization.
    sform : str, optional
        Explicit affine key to use for sform serialization.
    qform_code : int, optional
        Explicit qform code override.
    sform_code : int, optional
        Explicit sform code override.

    Returns
    -------
    stored_affines : dict[str, Any]
        Affines stored in `data_array.attrs["affines"]`.
    qform_affine : (4, 4) numpy.ndarray
        Final qform affine to write to the NIfTI header.
    sform_affine : (4, 4) numpy.ndarray or None
        Final sform affine to write to the NIfTI header, or `None` when no sform is
        written.
    resolved_qform_code : int
        Final qform code.
    resolved_sform_code : int
        Final sform code.
    written_header_affine_keys : set[str]
        Keys from `stored_affines` that were actually written into the NIfTI header and
        should therefore be omitted from `ConfUSIusAffines` in the sidecar.
    """
    stored_affines: dict[str, Any] = data_array.attrs.get("affines", {})
    selected_keys = {"qform": qform, "sform": sform}
    explicit_codes = {"qform": qform_code, "sform": sform_code}
    default_affine_keys = {
        "qform": "world_to_qform",
        "sform": "world_to_sform",
    }
    resolved_codes: dict[str, int] = {}
    header_affines: dict[str, npt.NDArray[np.floating] | None] = {}
    resolved_keys: dict[str, str | None] = {}
    written_header_affine_keys: set[str] = set()

    for form_name in ("qform", "sform"):
        resolved_key = _resolve_nifti_affine_key(
            stored_affines,
            form_name=form_name,
            selected_key=selected_keys[form_name],
            default_key=default_affine_keys[form_name],
        )
        resolved_keys[form_name] = resolved_key
        resolved_code = _resolve_nifti_xform_code(
            data_array,
            form_name=form_name,
            code=explicit_codes[form_name],
        )

        header_affines[form_name] = (
            _build_selected_nifti_affine(
                data_array,
                stored_affines=stored_affines,
                affine_key=resolved_key,
            )
            if form_name == "qform" or resolved_code > 0
            else None
        )
        resolved_codes[form_name] = resolved_code

        if resolved_code > 0 and resolved_key is not None:
            written_header_affine_keys.add(resolved_key)

    assert header_affines["qform"] is not None
    if resolved_codes["qform"] > 0 and _nifti_affine_has_shear(header_affines["qform"]):
        resolved_codes["qform"] = 0
        qform_key = resolved_keys["qform"]
        if qform_key is not None:
            written_header_affine_keys.discard(qform_key)

        if resolved_keys["sform"] is None:
            # sform has no explicit/stored source of its own (it was only going to
            # fall back to raw voxel_to_world), so it's safe to rescue the sheared
            # geometry into it instead of overwriting a deliberately requested sform.
            warnings.warn(
                "The coordinate-defining affine contains shear, which NIfTI qform "
                "cannot represent. Writing this geometry to sform instead and "
                "disabling qform.",
                stacklevel=find_stack_level(),
            )
            header_affines["sform"] = header_affines["qform"].copy()
            resolved_codes["sform"] = _resolve_nifti_xform_code(
                data_array,
                form_name="sform",
                code=explicit_codes["sform"],
            )
            if qform_key is not None:
                written_header_affine_keys.add(qform_key)
        else:
            # sform is already spoken for by an explicit/stored affine of its own, so
            # the sheared qform geometry can't be rescued there without overwriting
            # it. It is simply disabled and dropped from the NIfTI header; it
            # remains available via the ConfUSIusAffines sidecar under its own key.
            warnings.warn(
                "The coordinate-defining affine contains shear, which NIfTI qform "
                "cannot represent, and sform is already used for a different "
                "affine. Disabling qform; the sheared geometry is not written to "
                "the NIfTI header (it remains recorded in the ConfUSIusAffines "
                "sidecar).",
                stacklevel=find_stack_level(),
            )
    return (
        stored_affines,
        header_affines["qform"],
        header_affines["sform"],
        resolved_codes["qform"],
        resolved_codes["sform"],
        written_header_affine_keys,
    )


def _validate_affine_matrix(
    affine: npt.ArrayLike, *, name: str
) -> npt.NDArray[np.floating]:
    """Return a validated 4x4 affine matrix.

    Parameters
    ----------
    affine : array-like
        Candidate affine matrix.
    name : str
        Affine key used to report validation errors.

    Returns
    -------
    (4, 4) numpy.ndarray
        Validated affine matrix as a float array.

    Raises
    ------
    ValueError
        If `affine` does not have shape `(4, 4)`.
    """
    affine_array = np.asarray(affine, dtype=float)
    if affine_array.shape != (4, 4):
        raise ValueError(
            f"data_array.attrs['affines'][{name!r}] must have shape (4, 4), got "
            f"{affine_array.shape}."
        )

    return affine_array


def _build_nifti_sidecar_metadata(
    data_array: xr.DataArray,
    timing_metadata: dict[str, Any],
    stored_affines: dict[str, Any],
    written_header_affine_keys: set[str],
) -> dict[str, Any]:
    """Build the JSON sidecar payload before BIDS field conversion.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.
    timing_metadata : dict[str, Any]
        Timing-related metadata returned by `_build_nifti_timing_metadata`.
    stored_affines : dict[str, Any]
        Affines stored in `data_array.attrs["affines"]`.
    written_header_affine_keys : set of str
        Keys from `data_array.attrs["affines"]` that were actually written into the
        NIfTI qform and/or sform header fields.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary combining serializable DataArray attrs, non-header affines,
        and timing metadata.
    """

    sidecar_attrs = {
        k: v
        for k, v in data_array.attrs.items()
        if k not in ("sform_code", "qform_code", "affines", "voxel_to_world")
    }
    if "time" in data_array.coords:
        from_unit = data_array.coords["time"].attrs.get("units")
        for key in _TIME_ATTRS_TO_SECONDS:
            value = sidecar_attrs.get(key)
            if isinstance(value, int | float | np.integer | np.floating):
                sidecar_attrs[key] = float(convert_time_units(value, from_unit, "s"))

    extra_affines = {
        k: np.asarray(v).tolist()
        for k, v in stored_affines.items()
        if k not in written_header_affine_keys
    }
    if extra_affines:
        sidecar_attrs["affines"] = extra_affines

    sidecar_attrs.update(_build_extra_dim_sidecar_metadata(data_array))
    sidecar_attrs.update(timing_metadata)
    return _drop_non_json_serializable_attrs(sidecar_attrs)


def _drop_non_json_serializable_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop attrs whose values cannot be serialized to JSON as-is.

    Parameters
    ----------
    attrs : dict[str, Any]
        Attributes to filter.

    Returns
    -------
    dict[str, Any]
        Copy of `attrs` containing only entries whose value round-trips through
        `json.dumps` without coercion.
    """
    serializable_attrs = {}
    dropped_keys = []
    for key, value in attrs.items():
        try:
            json.dumps(value)
        except TypeError:
            dropped_keys.append(key)
        else:
            serializable_attrs[key] = value

    if dropped_keys:
        warnings.warn(
            f"Dropping non-JSON-serializable attrs from NIfTI sidecar: {dropped_keys}.",
            stacklevel=find_stack_level(),
        )

    return serializable_attrs


def _create_nifti_image(
    data_array: xr.DataArray,
    data: np.ndarray,
    *,
    nifti_version: NiftiVersion,
    spacings: list[float],
    qform_affine: npt.NDArray[np.floating],
    sform_affine: npt.NDArray[np.floating] | None,
    resolved_qform_code: int,
    resolved_sform_code: int,
) -> "nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image":
    """Create a configured nibabel NIfTI image ready to write.

    Parameters
    ----------
    data_array : xarray.DataArray
        Source array being serialized.
    data : numpy.ndarray
        Data reordered to NIfTI axis order.
    nifti_version : {1, 2}
        NIfTI image version to instantiate.
    spacings : list[float]
        Full NIfTI signed spacing vector, including temporal and extra-dimension entries when
        present.
    qform_affine : (4, 4) numpy.ndarray
        Resolved qform affine in NIfTI axis order.
    sform_affine : (4, 4) numpy.ndarray or None
        Resolved sform affine in NIfTI axis order, or `None` when no sform is written.
    resolved_qform_code : int
        Final qform code.
    resolved_sform_code : int
        Final sform code.

    Returns
    -------
    nibabel.nifti1.Nifti1Image or nibabel.nifti2.Nifti2Image
        Configured image with spacings, qform/sform, and units written to the header.
    """
    import nibabel as nib

    img_class = nib.Nifti1Image if nifti_version == 1 else nib.Nifti2Image
    constructor_affine = sform_affine if sform_affine is not None else qform_affine
    nifti_img = img_class(data, constructor_affine)

    # pixdim[4:] for temporal and extra dimensions are set independently of the qform.
    for axis_index, spacing in enumerate(spacings[3:], start=4):
        nifti_img.header.structarr["pixdim"][axis_index] = np.float32(spacing)

    # Setting the qform also sets several parameters in the NIfTI header:
    # the pixdim[1:4] (from zooms), qoffset_xyz (from translation), qfac (from
    # determinant of the rotation block) and quatern_bcd (from the quaternion
    # representation). When qform_code is 0, leave qform unset rather than letting
    # nibabel silently strip an unrepresentable shear.
    if resolved_qform_code > 0:
        nifti_img.header.set_qform(qform_affine, code=resolved_qform_code)
    else:
        nifti_img.header.set_qform(None, code=0)

    if sform_affine is not None:
        nifti_img.header.set_sform(sform_affine, code=resolved_sform_code)
    else:
        nifti_img.header.set_sform(None, code=0)

    # When qform is disabled (for example because the coordinate-defining affine
    # contains shear and was promoted to sform), keep the header voxel sizes tied to
    # the coordinate spacing we chose for serialization. Do not do this when qform is
    # active: qform owns pixdim[1:4], including signed-axis handling via qfac.
    if resolved_qform_code == 0:
        nifti_img.header.structarr["pixdim"][1:4] = np.asarray(
            np.abs(spacings[:3]), dtype=np.float32
        )

    spatial_units = set()
    for dim in ("x", "y", "z"):
        if dim in data_array.coords and "units" in data_array.coords[dim].attrs:
            spatial_units.add(data_array.coords[dim].attrs["units"])
    if len(spatial_units) > 1:
        warnings.warn(
            f"Spatial dimensions have different units: {spatial_units}. "
            f"NIfTI only supports a single spatial unit; using the first one found.",
            stacklevel=find_stack_level(),
        )

    space_unit_nib = None
    for dim in ("x", "y", "z"):
        if dim in data_array.coords and "units" in data_array.coords[dim].attrs:
            confusius_unit = data_array.coords[dim].attrs["units"]
            space_unit_nib = _CONFUSIUS_TO_NIFTI_SPACE_UNITS.get(confusius_unit)
            break

    if space_unit_nib is not None:
        nifti_img.header.set_xyzt_units(xyz=space_unit_nib, t="sec")

    return nifti_img


def save_nifti(
    data_array: xr.DataArray,
    path: str | Path,
    nifti_version: NiftiVersion = 1,
    *,
    qform: str | None = None,
    sform: str | None = None,
    qform_code: int | None = None,
    sform_code: int | None = None,
) -> None:
    """Save a VoxelData array to NIfTI format.

    Saves the DataArray to a NIfTI file and always writes a BIDS-style JSON sidecar
    alongside it. The data is transposed to NIfTI convention `(x, y, z, time)` before
    saving.

    Parameters
    ----------
    data_array : xarray.DataArray
        VoxelData array to save.
    path : str or pathlib.Path
        Output path for the NIfTI file, with `.nii` or `.nii.gz` extension. If
        `.nii.gz` is used, the file will be saved in compressed format.
    nifti_version : {1, 2}, default: 1
        NIfTI format version to use. Version 2 is a simple extension to support
        larger files and arrays with dimension sizes greater than 32,767.
    qform : str, optional
        Key in `data_array.attrs["affines"]` to write into the NIfTI qform. When not
        provided, `"world_to_qform"` is used if present; otherwise qform falls back
        to the coordinate-defining voxel geometry (`voxel_to_world`) directly.

        If the coordinate-defining affine contains shear, qform writing is disabled
        because the NIfTI qform cannot represent shear. In that case, the geometry is
        written to sform instead.
    sform : str, optional
        Key in `data_array.attrs["affines"]` to write into the NIfTI sform. When not
        provided, `"world_to_sform"` is used if present; otherwise sform also falls
        back to `voxel_to_world` directly (like qform), unless disabled via
        `sform_code=0` or `attrs["sform_code"] = 0`.
    qform_code : int, optional
        NIfTI qform code to write. When provided, takes precedence over
        `data_array.attrs["qform_code"]`. When not provided, the value from
        `attrs["qform_code"]` is used if present; otherwise defaults to `2`
        (`NIFTI_XFORM_ALIGNED_ANAT`).
    sform_code : int, optional
        NIfTI sform code to write. When provided, takes precedence over
        `data_array.attrs["sform_code"]`. When not provided, the value from
        `attrs["sform_code"]` is used if present; otherwise defaults to `2`
        (`NIFTI_XFORM_ALIGNED_ANAT`).

    Notes
    -----
    Time coordinates are automatically converted to seconds for BIDS compliance. If the
    time coordinate has a "units" attribute, values are converted from "ms" or "us" to
    "s". If no units are specified, seconds are assumed. Known time-valued processing
    metadata stored in `data_array.attrs` is converted to seconds using the same unit
    convention.

    A warning is issued if spatial dimensions `(x, y, z)` have inconsistent units, as
    NIfTI only supports a single spatial unit in the `xyzt_units` header field.

    Examples
    --------
    >>> import confusius as cf
    >>> import numpy as np
    >>> da = cf.xarray.create_voxeldata(
    ...     np.random.rand(10, 1, 32, 64),
    ...     dims=("time", "k", "j", "i"),
    ...     dt=0.5,
    ...     spacing=(0.4, 0.1, 0.1),
    ... )
    >>> cf.io.save_nifti(da, "output.nii.gz")
    >>> da.attrs["affines"] = {"world_to_template": np.eye(4)}
    >>> cf.io.save_nifti(da, "output.nii.gz", sform="world_to_template")
    """
    # NIfTI exposes only one spatial qform/sform, so it cannot represent per-pose
    # geometry (or even a shared affine ambiguously labeled per pose). Callers must
    # select one pose first, e.g. `save_nifti(data.isel(pose=0), path)`.
    data_array = ensure_voxeldata(data_array, allow_pose=False)
    path = Path(path)
    if not path.name.endswith(".nii") and not path.name.endswith(".nii.gz"):
        raise ValueError("Output file must have .nii or .nii.gz extension.")

    data, has_time_axis, extra_dims = _prepare_data_for_nifti(data_array)
    spatial_spacings = _get_spatial_spacings(data_array)
    timing_metadata, tr_pixdim = _build_nifti_timing_metadata(data_array)

    spacings = spatial_spacings.copy()
    if has_time_axis:
        temporal_spacing = tr_pixdim if tr_pixdim is not None else 0.0
        spacings.append(temporal_spacing)
    for dim_name in extra_dims:
        spacing = get_coordinate_spacing_info(
            dim_name, data_array, uniformity_tolerance=1e-2
        )
        if spacing.value is not None:
            spacings.append(float(spacing.value))
        else:
            spacings.append(1.0)

    (
        stored_affines,
        qform_affine,
        sform_affine,
        resolved_qform_code,
        resolved_sform_code,
        written_header_affine_keys,
    ) = _prepare_nifti_xforms(
        data_array,
        qform=qform,
        sform=sform,
        qform_code=qform_code,
        sform_code=sform_code,
    )

    nifti_img = _create_nifti_image(
        data_array,
        data,
        nifti_version=nifti_version,
        spacings=spacings,
        qform_affine=qform_affine,
        sform_affine=sform_affine,
        resolved_qform_code=resolved_qform_code,
        resolved_sform_code=resolved_sform_code,
    )

    nifti_img.to_filename(path)

    sidecar_attrs = _build_nifti_sidecar_metadata(
        data_array, timing_metadata, stored_affines, written_header_affine_keys
    )

    if path.suffix == ".gz":
        sidecar_path = path.with_suffix("").with_suffix(".json")
    else:
        sidecar_path = path.with_suffix(".json")

    bids_attrs = to_bids(sidecar_attrs)

    if bids_attrs:
        try:
            validate_metadata(bids_attrs)
        except ValidationError as e:
            warnings.warn(
                f"fUSI-BIDS validation warning when saving:\n{format_validation_error(e)}",
                stacklevel=find_stack_level(),
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"fUSI-BIDS validation warning when saving: {e}",
                stacklevel=find_stack_level(),
            )

    with open(sidecar_path, "w") as f:
        json.dump(bids_attrs, f, indent=2)
