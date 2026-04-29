"""Utilities for loading and saving NIfTI files.

This module provides functions to load NIfTI neuroimaging files as lazy Xarray
DataArrays using nibabel's proxy arrays and Dask for out-of-core processing. Following
ConfUSIus conventions, data is stored with dimensions `(time, z, y, x)`.
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

from confusius._utils import (
    find_stack_level,
    get_coordinate_spacing_info,
    get_representative_step,
)
from confusius.bids import (
    DIM_TO_SLICE_ENCODING_DIRECTION,
    SLICE_ENCODING_DIRECTION_TO_DIM,
    create_bids_slice_timing_from_coordinate,
    create_slice_time_coordinate_from_bids,
    from_bids,
    to_bids,
)
from confusius.bids.validation import format_validation_error, validate_metadata
from confusius.io.utils import check_path
from confusius.registration.affines import decompose_affine
from confusius.timing import convert_time_reference, convert_time_values

if TYPE_CHECKING:
    import nibabel as nib

NiftiVersion: TypeAlias = Literal[1, 2]
"""Type alias for NIfTI file format version."""

_NIFTI_DIM_ORDER = ("x", "y", "z", "time", "dim4", "dim5", "dim6")
"""ConfUSIus and NIfTI dimension mapping.

Maps ConfUSIus dimension names to their NIfTI axis order.
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
            `get_unit_strings`).
        """
        zooms = self.header.get_zooms()
        nifti_spatial = [("x", 0), ("y", 1), ("z", 2)]
        return {name: float(zooms[i]) for name, i in nifti_spatial if i < len(zooms)}

    def get_repetition_time(self) -> float | None:
        """Get repetition time (TR) in its native header units.

        Returns
        -------
        float or None
            Repetition time in the units declared by the NIfTI header (see
            `get_unit_strings`), or `None` if not available.
        """
        zooms = self.header.get_zooms()
        if len(zooms) < 4:
            return None
        return float(zooms[3])

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
        except Exception as e:
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
        raise ValueError(
            "Only NIfTI-1 and NIfTI-2 formats are supported when loading files with"
            " .nii or .nii.gz suffixes."
        )

    return img, _NiftiHeaderExtractor(img.header)


def _create_spatial_coords_from_nifti(
    img: "nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image",
    extractor: "_NiftiHeaderExtractor",
    dims: tuple[str, ...],
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Create spatial coordinate arrays and affine attributes from a NIfTI image.

    Selects the primary affine (sform preferred over qform when both are valid)
    and builds one coordinate per spatial dimension. When a valid affine is
    available, each axis starts at `affine[col, 3]` (origin) and is stepped by
    the true voxel size (from `decompose_affine`). The rotation and shear are
    captured in a 4×4 `A_physical` affine stored in the returned attributes.

    When both sform and qform codes are zero, coordinates are built from
    pixdim only (origin 0, step = voxel size). A warning is emitted and no
    `"affines"` entry is added to the returned attributes.

    Parameters
    ----------
    img : nib.nifti1.Nifti1Image or nib.nifti2.Nifti2Image
        Loaded NiBabel NIfTI image.
    extractor : _NiftiHeaderExtractor
        Header extractor for `img`.
    dims : tuple[str, ...]
        Dimension names in NIfTI order `(x, y, z[, time])`.

    Returns
    -------
    coords : dict[str, xarray.DataArray]
        Spatial coordinate DataArrays keyed by name (`"x"`, `"y"`, `"z"`).
    extra_attrs : dict[str, Any]
        Affine-derived DataArray attributes. Contains `"affines"` when a valid
        affine is present; empty otherwise.
    """
    voxel_sizes = extractor.get_voxel_dimensions()
    space_unit, _ = extractor.get_unit_strings()
    primary_affine, secondary_affine = _select_affines(img.header)

    coords: dict[str, xr.DataArray] = {}
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
        for col, dim in enumerate(("x", "y", "z")):
            if dim not in dims:
                continue
            coord_attrs: dict[str, Any] = {}
            if space_unit is not None:
                coord_attrs["units"] = space_unit
            step = voxel_sizes[dim] if dim in voxel_sizes else 1.0
            if dim in voxel_sizes:
                coord_attrs["voxdim"] = voxel_sizes[dim]
            coords[dim] = xr.DataArray(
                step * np.arange(img.shape[col]),
                dims=[dim],
                attrs=coord_attrs,
            )
    else:
        # Sform is preferred; when both codes are positive, qform is secondary.
        _, sform_code = img.header.get_sform(coded=True)
        primary_prefix = "sform" if sform_code > 0 else "qform"

        affines_dict: dict[str, npt.NDArray[np.floating]] = {}
        nifti_affines: list[tuple[npt.NDArray[np.floating], str]] = [
            (primary_affine, primary_prefix),
        ]
        if secondary_affine is not None:
            nifti_affines.append((secondary_affine, "qform"))

        for affine, prefix in nifti_affines:
            T, _, Z, _ = decompose_affine(affine)

            # Orientation matrix D satisfies D @ diag(Z) = RZS and captures rotation
            # and shear.
            D = affine[:3, :3] / Z

            # Build A_physical: the 4×4 affine mapping physical coordinates (in the
            # spatial units declared by the NIfTI header) to sform/qform world-space
            # coordinates (same units). In NIfTI (x, y, z) convention:
            #   A_nifti[:3, :3] = D,  A_nifti[:3, 3] = T − D @ T
            # so that A_nifti @ [px, py, pz, 1] → [wx, wy, wz, 1].
            # A_physical is invariant to all slicing and downsampling because it maps
            # physical positions, not voxel indices.
            A_nifti = np.eye(4)
            A_nifti[:3, :3] = D
            A_nifti[:3, 3] = T - D @ T

            # Permute to ConfUSIus (z, y, x) convention by swapping rows and columns
            # 0 ↔ 2 (equivalent to P @ A_nifti @ P where P is the flip permutation).
            # Result: A_physical @ [pz, py, px, 1] → [wz, wy, wx, 1].
            A_physical = A_nifti[[2, 1, 0, 3]][:, [2, 1, 0, 3]]
            affines_dict[f"physical_to_{prefix}"] = A_physical

            # Only the primary affine contributes dimension coordinates.
            if prefix != primary_prefix:
                continue

            for col, dim in enumerate(("x", "y", "z")):
                if dim not in dims:
                    continue
                coord_attrs_a: dict[str, Any] = {}
                if space_unit is not None:
                    coord_attrs_a["units"] = space_unit
                if dim in voxel_sizes:
                    coord_attrs_a["voxdim"] = voxel_sizes[dim]

                # Coordinates encode translation (T[col]) plus zoom (Z[col] from
                # decompose_affine, the true voxel spacing along this axis).
                coords[dim] = xr.DataArray(
                    T[col] + Z[col] * np.arange(img.shape[col]),
                    dims=[dim],
                    attrs=coord_attrs_a,
                )

        if affines_dict:
            extra_attrs["affines"] = affines_dict

    return coords, extra_attrs


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
        time_attrs["volume_acquisition_duration"] = attrs.pop(
            "volume_acquisition_duration"
        )

    if "volume_timing" in attrs:
        time_values = np.asarray(attrs.pop("volume_timing"))
    elif "repetition_time" in attrs:
        sampling_period_sidecar = float(attrs.pop("repetition_time"))
        delay = float(attrs.pop("delay_after_trigger", 0.0))
        delay_time = float(attrs.pop("delay_time", 0.0))
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
        time_values = delay + sampling_period_sidecar * np.arange(n_time)
    elif sampling_period_nifti is not None:
        time_values = sampling_period_nifti * np.arange(n_time)
    else:
        time_values = np.arange(n_time, dtype=np.float64)

    coords: dict[str, xr.DataArray] = {
        "time": xr.DataArray(time_values, dims=["time"], attrs=time_attrs)
    }

    if "slice_timing" in attrs and "slice_encoding_direction" in attrs:
        coords["slice_time"] = create_slice_time_coordinate_from_bids(
            volume_times=coords["time"].values,
            slice_timing=convert_time_values(
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
        time_attrs["volume_acquisition_duration"] = attrs.pop(
            "volume_acquisition_duration"
        )

    time_value: float | None = None
    if "volume_timing" in attrs:
        volume_timing = np.asarray(attrs.pop("volume_timing"), dtype=np.float64)
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

    coords: dict[str, xr.DataArray] = {
        "time": xr.DataArray(np.float64(time_value), attrs=time_attrs)
    }

    if "slice_timing" in attrs and "slice_encoding_direction" in attrs:
        slice_encoding_direction = str(attrs.pop("slice_encoding_direction"))
        if slice_encoding_direction not in SLICE_ENCODING_DIRECTION_TO_DIM:
            return coords, attrs

        slice_timing = convert_time_values(
            attrs.pop("slice_timing"),
            from_unit="s",
            to_unit=coords["time"].attrs.get("units", "s"),
            raise_on_unknown=True,
        )
        slice_timing = np.asarray(slice_timing, dtype=np.float64)
        if slice_timing.ndim != 1:
            return coords, attrs
        if slice_encoding_direction.endswith("-"):
            slice_timing = slice_timing[::-1]

        spatial_dim = SLICE_ENCODING_DIRECTION_TO_DIM[slice_encoding_direction]
        coords["slice_time"] = xr.DataArray(
            time_value + slice_timing,
            dims=[spatial_dim],
            attrs={"units": coords["time"].attrs.get("units", "s")},
        )

    return coords, attrs


def _get_volume_acquisition_reference(
    attrs: dict[str, Any], *, coord_name: str, warn_on_missing: bool = False
) -> Literal["start", "center", "end"]:
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

    if reference not in {"start", "center", "end"}:
        raise ValueError(
            f"Unknown {coord_name} volume_acquisition_reference: {reference!r}. "
            "Must be 'start', 'center', or 'end'."
        )

    return reference


def load_nifti(
    path: str | Path, chunks: int | tuple[int, ...] | str | None = "auto"
) -> xr.DataArray:
    """Load a NIfTI file as a lazy Xarray DataArray.

    Loads NIfTI files using nibabel's proxy arrays for memory-efficient access, wrapping
    the data in Dask arrays for chunked, parallel processing. The data is transposed to
    ConfUSIus conventions with dimensions `(time, z, y, x)`.

    A BIDS-style JSON sidecar file (same name, `.json` extension) is loaded
    automatically when present.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NIfTI file (`.nii` or `.nii.gz`).
    chunks : int or tuple[int, ...] or str or None, default: "auto"
        How to chunk the array. Must be one of the following forms:

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

    Returns
    -------
    xarray.DataArray
        Lazy DataArray with dimensions in ConfUSIus order. Data is wrapped in a Dask
        array for out-of-core computation.

    Examples
    --------
    >>> import confusius as cf
    >>> da = cf.io.load_nifti("brain.nii.gz")
    >>> print(da.dims)
    ("time", "z", "y", "x")

    Notes
    -----
    Physical-to-world affines are stored in `da.attrs["affines"]`, a dict keyed by
    affine name. Each value is a 4×4 affine in ConfUSIus `(z, y, x)` convention that
    maps **physical coordinates** (as stored in `da.coords`) to world-space
    coordinates. Apply as `da.attrs["affines"]["physical_to_sform"] @ np.array([pz, py,
    px, 1.0])` to get `[wz, wy, wx, 1]`, where `pz`, `py`, `px` come from
    `da.coords["z"]`, `da.coords["y"]`, `da.coords["x"]` respectively.

    Unlike the NIfTI affine (which maps voxel *indices* to world space), the
    `physical_to_*` affines are invariant to any slicing or downsampling because they
    operate on physical positions, not grid indices.

    Affine selection follows NIfTI conventions:

    - If `sform_code > 0`: sform is used as the primary affine; a
      `"physical_to_sform"` entry is written. When `qform_code > 0` as well, a
      `"physical_to_qform"` entry is also stored as secondary.
    - Else, if only `qform_code > 0`: qform is used as the primary affine; only
      `"physical_to_qform"` is written.
    - If both codes are zero: a warning is emitted, coordinates are built from
      `pixdim` only (origin 0, step = voxel size), and no `"affines"` entry is
      stored in `da.attrs`.

    The raw integer form codes are stored as `da.attrs["qform_code"]` and
    `da.attrs["sform_code"]` (only when > 0) so that a save/load roundtrip can
    reproduce the original NIfTI header codes.

    Voxel dimensions are stored in their native header units as a `voxdim`
    attribute on each spatial coordinate array, consistent with the `units`
    attribute of that coordinate.
    """
    path = check_path(path, type="file")

    img, extractor = _load_nifti_with_nibabel(path)

    attrs = extractor.to_attrs()
    attrs.update(_load_nifti_sidecar(path))

    # We use np.asanyarray to get the memory-mapped array behind Nibabel's proxy. This
    # will allow Dask to create a lazy array without loading the entire dataset into
    # memory.
    # NIfTI stores data with shape (x, y, z, time) in column-major order; transposing
    # to row-major gives ConfUSIus order (time, z, y, x) without copying data.
    dask_arr = da.from_array(np.asanyarray(img.dataobj).T, chunks=chunks, asarray=False)

    # NIfTI dim order is (x, y, z, time, ...); ConfUSIus order is the reverse.
    nifti_dims = _NIFTI_DIM_ORDER[: dask_arr.ndim]

    spatial_coords, affine_attrs = _create_spatial_coords_from_nifti(
        img=img, extractor=extractor, dims=nifti_dims
    )
    attrs.update(affine_attrs)

    if "time" in nifti_dims:
        temporal_coords, attrs = _create_temporal_coords_from_nifti(
            img=img, extractor=extractor, attrs=attrs
        )
        coords = {**spatial_coords, **temporal_coords}
    else:
        scalar_temporal_coords, attrs = _create_scalar_temporal_coords_from_nifti(
            extractor=extractor, attrs=attrs
        )
        coords = {**spatial_coords, **scalar_temporal_coords}

    nifti_name = path.with_suffix("").stem if path.suffix == ".gz" else path.stem
    data_array = xr.DataArray(
        dask_arr, dims=nifti_dims[::-1], coords=coords, attrs=attrs, name=nifti_name
    )

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
    timings = np.atleast_1d(np.asarray(timings, dtype=np.float64))
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
        if spatial_dim not in DIM_TO_SLICE_ENCODING_DIRECTION:
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
            np.asarray(
                convert_time_values(
                    data_array.coords["time"].values,
                    data_array.coords["time"].attrs.get("units"),
                    "s",
                    raise_on_unknown=True,
                ),
                dtype=np.float64,
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
        slice_time_seconds = np.asarray(
            convert_time_values(
                slice_time_coord.values,
                slice_time_coord.attrs.get("units"),
                "s",
                raise_on_unknown=True,
            ),
            dtype=np.float64,
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
            "SliceEncodingDirection": DIM_TO_SLICE_ENCODING_DIRECTION[spatial_dim],
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
    if len(spatial_dims) != 1 or spatial_dims[0] not in DIM_TO_SLICE_ENCODING_DIRECTION:
        return {}

    if "time" not in data_array.coords:
        warnings.warn(
            "Cannot infer onset-relative SliceTiming from a 2D `slice_time` "
            "coordinate without a `time` coordinate. Omitting BIDS SliceTiming export.",
            stacklevel=find_stack_level(),
        )
        return {}

    time_values_seconds = convert_time_values(
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
    slice_time_seconds = convert_time_values(
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


def _build_nifti_affine(
    transform: npt.NDArray[np.floating] | None,
    T: npt.NDArray[np.floating],
    Z: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Reconstruct a NIfTI affine from a stored ConfUSIus physical-to-world transform.

    Reverses the ConfUSIus `(z, y, x)` permutation to recover the orientation matrix
    `D`, then combines it with the physical-coord origin `T` and spacing `Z` to rebuild
    the full NIfTI affine. Falls back to a diagonal affine when no transform is given.

    Parameters
    ----------
    transform : (4, 4) numpy.ndarray or None
        Physical-to-world affine in ConfUSIus `(z, y, x)` convention, or `None` to use a
        fallback diagonal affine.
    T : (3,) numpy.ndarray
        Origin of the physical coordinates for each NIfTI axis `(x, y, z)`, in the
        spatial units of the DataArray coordinates.
    Z : (3,) numpy.ndarray
        Voxel spacing for each NIfTI axis `(x, y, z)`, in the spatial units of the
        DataArray coordinates.

    Returns
    -------
    numpy.ndarray of shape (4, 4)
        Full NIfTI affine mapping voxel indices to world-space coordinates.
    """
    if transform is not None:
        A_nifti = np.asarray(transform)[[2, 1, 0, 3]][:, [2, 1, 0, 3]]
        D = A_nifti[:3, :3]
        out = np.eye(4)
        out[:3, :3] = D @ np.diag(Z)
        out[:3, 3] = T
    else:
        out = np.eye(4)
        out[:3, :3] = np.diag(Z)
        out[:3, 3] = T

    return out


def _prepare_data_for_nifti(
    data_array: xr.DataArray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return array data reordered to NIfTI axis order.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array to serialize.

    Returns
    -------
    data : numpy.ndarray
        Array reordered to NIfTI axis order `(x, y, z, time, ...)`, with singleton
        spatial axes inserted for any missing spatial dimensions. Boolean arrays are
        cast to `uint8` because NIfTI does not support `bool` payload dtypes.
    current_dims : tuple[str, ...]
        Original dimension names from `data_array`, preserved so later header logic can
        decide whether a time zoom should be written and where extra dimensions belong.
    """
    data = np.asarray(data_array)
    current_dims = tuple(str(dim) for dim in data_array.dims)

    target_order = []
    for dim in ("x", "y", "z", "time"):
        if dim in current_dims:
            target_order.append(current_dims.index(dim))

    for i, dim in enumerate(current_dims):
        if dim not in ("x", "y", "z", "time"):
            target_order.append(i)

    data = np.transpose(data, target_order)

    for insert_pos, dim in enumerate(("x", "y", "z")):
        if dim not in current_dims:
            data = np.expand_dims(data, axis=insert_pos)

    if np.issubdtype(data.dtype, np.bool_):
        data = data.astype(np.uint8, copy=False)

    return data, current_dims


def _get_spatial_zooms(data_array: xr.DataArray) -> list[float]:
    """Return spatial zooms for NIfTI header serialization.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.

    Returns
    -------
    list[float]
        Spatial voxel sizes for the NIfTI `x`, `y`, and `z` axes.

    Warns
    -----
    UserWarning
        If spatial spacing is non-uniform or undefined and a best-effort fallback is
        needed for the NIfTI header.
    """
    spatial_zooms: list[float] = []
    for dim in ("x", "y", "z"):
        spacing = get_coordinate_spacing_info(
            dim, data_array, uniformity_tolerance=1e-2
        )
        if spacing.value is not None:
            spatial_zooms.append(abs(spacing.value))
        elif spacing.median is not None:
            spatial_zooms.append(abs(spacing.median))
            warnings.warn(
                f"Coordinate '{dim}' has non-uniform spacing. NIfTI stores one "
                f"constant spacing per axis, so using the median step "
                f"{abs(spacing.median):.4g} for the header and affine; positions "
                "along this axis may be approximate.",
                stacklevel=find_stack_level(),
            )
        else:
            spatial_zooms.append(1.0)
            if spacing.warn_msg is not None:
                warnings.warn(
                    f"{spacing.warn_msg} Falling back to unit spacing for NIfTI "
                    f"axis '{dim}'.",
                    stacklevel=find_stack_level(),
                )

    return spatial_zooms


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

        time_values_seconds = convert_time_values(
            time_values_raw,
            time_unit,
            "s",
            raise_on_unknown=True,
        )
        time_values_seconds = np.atleast_1d(
            np.asarray(time_values_seconds, dtype=np.float64)
        )
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
    has_affine: bool,
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
    has_affine : bool
        Whether a stored affine was selected for this xform field.

    Returns
    -------
    int
        Resolved xform code. Explicit `code` takes precedence, then the corresponding
        `data_array.attrs["<form_name>_code"]` value when present. Otherwise qform
        defaults to `1`, while sform defaults to `1` only when an affine is available
        and `0` when no sform affine will be written.
    """
    if code is not None:
        return code

    attr_name = f"{form_name}_code"
    if attr_name in data_array.attrs:
        return int(data_array.attrs[attr_name])

    if form_name == "qform":
        return 1

    return 1 if has_affine else 0


def _build_selected_nifti_affine(
    data_array: xr.DataArray,
    *,
    spatial_zooms: list[float],
    stored_affines: dict[str, Any],
    affine_key: str | None,
) -> npt.NDArray[np.floating]:
    """Build a NIfTI header affine from a selected stored affine key.

    Parameters
    ----------
    data_array : xarray.DataArray
        Array being serialized.
    spatial_zooms : list[float]
        Spatial voxel sizes for the NIfTI `x`, `y`, and `z` axes.
    stored_affines : dict[str, Any]
        Affines stored in `data_array.attrs["affines"]`.
    affine_key : str, optional
        Key of the stored ConfUSIus physical-to-world affine to encode. When `None`, a
        diagonal NIfTI affine is built directly from coordinate origin and spacing.

    Returns
    -------
    (4, 4) numpy.ndarray
        NIfTI voxel-to-world affine in NIfTI axis order.
    """
    origin = np.array(
        [
            float(data_array.coords[dim][0]) if dim in data_array.coords else 0.0
            for dim in ("x", "y", "z")
        ]
    )
    zooms = np.array(spatial_zooms)
    transform = (
        _validate_affine_matrix(stored_affines[affine_key], name=affine_key)
        if affine_key is not None
        else None
    )
    return _build_nifti_affine(transform, origin, zooms)


def _prepare_nifti_xforms(
    data_array: xr.DataArray,
    *,
    spatial_zooms: list[float],
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
    spatial_zooms : list[float]
        Spatial voxel sizes for the NIfTI `x`, `y`, and `z` axes.
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
        "qform": "physical_to_qform",
        "sform": "physical_to_sform",
    }
    resolved_codes: dict[str, int] = {}
    header_affines: dict[str, npt.NDArray[np.floating] | None] = {}
    written_header_affine_keys: set[str] = set()

    for form_name in ("qform", "sform"):
        resolved_key = _resolve_nifti_affine_key(
            stored_affines,
            form_name=form_name,
            selected_key=selected_keys[form_name],
            default_key=default_affine_keys[form_name],
        )
        resolved_code = _resolve_nifti_xform_code(
            data_array,
            form_name=form_name,
            code=explicit_codes[form_name],
            has_affine=resolved_key is not None,
        )

        header_affines[form_name] = (
            _build_selected_nifti_affine(
                data_array,
                spatial_zooms=spatial_zooms,
                stored_affines=stored_affines,
                affine_key=resolved_key,
            )
            if form_name == "qform" or resolved_code > 0
            else None
        )
        resolved_codes[form_name] = resolved_code

        if resolved_code > 0 and resolved_key is not None:
            written_header_affine_keys.add(resolved_key)

    # We're guaranteed that qform isn't None since we always build a fallback affine for
    # it.
    assert header_affines["qform"] is not None
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
        if k not in ("sform_code", "qform_code", "affines")
    }
    if "time" in data_array.coords:
        from_unit = data_array.coords["time"].attrs.get("units")
        for key in _TIME_ATTRS_TO_SECONDS:
            value = sidecar_attrs.get(key)
            if isinstance(value, int | float | np.integer | np.floating):
                converted_value = convert_time_values(value, from_unit, "s")
                sidecar_attrs[key] = float(
                    np.asarray(converted_value, dtype=np.float64)
                )

    extra_affines = {
        k: np.asarray(v).tolist()
        for k, v in stored_affines.items()
        if k not in written_header_affine_keys
    }
    if extra_affines:
        sidecar_attrs["affines"] = extra_affines

    sidecar_attrs.update(timing_metadata)
    return sidecar_attrs


def _create_nifti_image(
    data_array: xr.DataArray,
    data: np.ndarray,
    *,
    nifti_version: NiftiVersion,
    zooms: list[float],
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
    zooms : list[float]
        Full NIfTI zoom vector, including temporal and extra-dimension entries when
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
        Configured image with zooms, qform/sform, and units written to the header.
    """
    import nibabel as nib

    img_class = nib.Nifti1Image if nifti_version == 1 else nib.Nifti2Image
    constructor_affine = sform_affine if sform_affine is not None else qform_affine
    nifti_img = img_class(data, constructor_affine)

    nifti_img.header.set_zooms(zooms)
    nifti_img.header.set_qform(qform_affine, code=resolved_qform_code)

    if sform_affine is not None:
        nifti_img.header.set_sform(sform_affine, code=resolved_sform_code)
    else:
        nifti_img.header.set_sform(None, code=0)

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
    """Save an Xarray DataArray to NIfTI format.

    Saves the DataArray to a NIfTI file and always writes a BIDS-style JSON sidecar
    alongside it. The data is transposed to NIfTI convention `(x, y, z, time)` before
    saving.

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to save.
    path : str or pathlib.Path
        Output path for the NIfTI file, with `.nii` or `.nii.gz` extension. If
        `.nii.gz` is used, the file will be saved in compressed format.
    nifti_version : {1, 2}, default: 1
        NIfTI format version to use. Version 2 is a simple extension to support
        larger files and arrays with dimension sizes greater than 32,767.
    qform : str, optional
        Key in `data_array.attrs["affines"]` to write into the NIfTI qform. When not
        provided, `"physical_to_qform"` is used if present; otherwise qform falls back
        to a diagonal affine derived from voxel spacing.
    sform : str, optional
        Key in `data_array.attrs["affines"]` to write into the NIfTI sform. When not
        provided, `"physical_to_sform"` is used if present; otherwise no sform is
        written.
    qform_code : int, optional
        NIfTI qform code to write. When provided, takes precedence over
        `data_array.attrs["qform_code"]`. When not provided, the value from
        `attrs["qform_code"]` is used if present; otherwise defaults to `1`.
    sform_code : int, optional
        NIfTI sform code to write. When provided, takes precedence over
        `data_array.attrs["sform_code"]`. When not provided, the value from
        `attrs["sform_code"]` is used if present; otherwise defaults to `1` when a
        sform affine is available, or `0` when no sform affine is written.

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
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(10, 32, 1, 64),
    ...                   dims=["time", "z", "y", "x"])
    >>> cf.io.save_nifti(da, "output.nii.gz")
    >>> cf.io.save_nifti(da, "output.nii.gz", sform="physical_to_template")
    """
    path = Path(path)
    if not path.name.endswith(".nii") and not path.name.endswith(".nii.gz"):
        raise ValueError("Output file must have .nii or .nii.gz extension.")

    data, current_dims = _prepare_data_for_nifti(data_array)
    spatial_zooms = _get_spatial_zooms(data_array)
    timing_metadata, tr_pixdim = _build_nifti_timing_metadata(data_array)

    zooms = spatial_zooms.copy()
    if "time" in current_dims:
        zooms.append(tr_pixdim if tr_pixdim is not None else 1.0)
    zooms.extend(1.0 for dim in current_dims if dim not in ("x", "y", "z", "time"))

    (
        stored_affines,
        qform_affine,
        sform_affine,
        resolved_qform_code,
        resolved_sform_code,
        written_header_affine_keys,
    ) = _prepare_nifti_xforms(
        data_array,
        spatial_zooms=spatial_zooms,
        qform=qform,
        sform=sform,
        qform_code=qform_code,
        sform_code=sform_code,
    )

    nifti_img = _create_nifti_image(
        data_array,
        data,
        nifti_version=nifti_version,
        zooms=zooms,
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
        except Exception as e:
            warnings.warn(
                f"fUSI-BIDS validation warning when saving: {e}",
                stacklevel=find_stack_level(),
            )

    with open(sidecar_path, "w") as f:
        json.dump(bids_attrs, f, indent=2, default=str)
