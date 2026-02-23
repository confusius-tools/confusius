"""Utilities for loading and saving NIfTI files.

This module provides functions to load NIfTI neuroimaging files as lazy Xarray
DataArrays using nibabel's proxy arrays and Dask for out-of-core processing. Following
ConfUSIus conventions, data is stored with dimensions ``(time, z, y, x)``.
"""

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import find_stack_level
from confusius.io.utils import check_path
from confusius.registration.affines import decompose_affine

if TYPE_CHECKING:
    import nibabel as nib

NiftiVersion: TypeAlias = Literal[1, 2]
"""Type alias for NIfTI file format version."""

# ConfUSIus and NIfTI dimension mapping.
_NIFTI_DIM_ORDER = ("x", "y", "z", "time", "dim4", "dim5", "dim6")

# Unit string mappings between NiBabel and ConfUSIus conventions.
_NIFTI_TO_CONFUSIUS_SPACE_UNITS: dict[str, str] = {
    "meter": "m",
    "mm": "mm",
    "micron": "um",
}
_NIFTI_TO_CONFUSIUS_TIME_UNITS: dict[str, str] = {
    "sec": "s",
    "msec": "ms",
    "usec": "us",
}
_CONFUSIUS_TO_NIFTI_SPACE_UNITS: dict[str, str] = {
    v: k for k, v in _NIFTI_TO_CONFUSIUS_SPACE_UNITS.items()
}
_CONFUSIUS_TO_NIFTI_TIME_UNITS: dict[str, str] = {
    v: k for k, v in _NIFTI_TO_CONFUSIUS_TIME_UNITS.items()
}


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
            Voxel dimensions keyed by ConfUSIus dimension name (``"x"``, ``"y"``,
            ``"z"``), in the units declared by the NIfTI header (see
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
            `get_unit_strings`), or ``None`` if not available.
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
            Spatial unit string (``"m"``, ``"mm"``, or ``"um"``), or ``None``
            if the header declares unknown units.
        time_unit : str or None
            Temporal unit string (``"s"``, ``"ms"``, or ``"us"``), or ``None``
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
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """Select primary and secondary affine matrices from a NiBabel header.

    Sform is preferred over qform when both codes are positive. When both are
    valid, the qform is returned as the secondary affine so that scanner-space
    coordinates can be stored alongside the template-space primary coordinates.
    When both codes are zero, ``(None, None)`` is returned.

    Parameters
    ----------
    header : nibabel.nifti1.Nifti1Header or nibabel.nifti2.Nifti2Header
        NiBabel NIfTI header.

    Returns
    -------
    primary_affine : (4, 4) numpy.ndarray or None
        Primary affine for computing spatial coordinates, or ``None`` when both
        sform and qform codes are zero.
    secondary_affine : (4, 4) numpy.ndarray or None
        Qform affine when both sform and qform codes are positive; ``None`` otherwise.
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


def _load_nifti_with_nibabel(
    path: Path,
) -> tuple[
    Union["nib.nifti1.Nifti1Image", "nib.nifti2.Nifti2Image"],
    dict[str, Any],
    dict[str, float],
    float | None,
    str | None,
    str | None,
]:
    """Load NIfTI file using NiBabel, extracting header metadata.

    Parameters
    ----------
    path : pathlib.Path
        Path to the NIfTI file (``.nii`` or ``.nii.gz``).

    Returns
    -------
    img : nib.nifti1.Nifti1Image or nib.nifti2.Nifti2Image
        NiBabel NIfTI image object with proxy data array.
    attrs : dict
        Dictionary of attributes extracted from the NIfTI header.
    voxel_sizes : dict[str, float]
        Voxel dimensions keyed by dimension name (``"x"``, ``"y"``, ``"z"``),
        in native header units.
    tr : float or None
        Repetition time in native header units, or ``None`` if not available.
    space_unit : str or None
        Spatial unit string in ConfUSIus conventions, or ``None`` if unknown.
    time_unit : str or None
        Temporal unit string in ConfUSIus conventions, or ``None`` if unknown.
    """
    import nibabel as nib

    img = nib.load(path)
    if not isinstance(img, nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image):
        raise ValueError(
            "Only NIfTI-1 and NIfTI-2 formats are supported when loading files with"
            " .nii or .nii.gz suffixes."
        )

    extractor = _NiftiHeaderExtractor(img.header)
    attrs = extractor.to_attrs()
    voxel_sizes = extractor.get_voxel_dimensions()
    tr = extractor.get_repetition_time()
    space_unit, time_unit = extractor.get_unit_strings()

    return img, attrs, voxel_sizes, tr, space_unit, time_unit


def _create_coords_from_nifti(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    primary_affine: npt.NDArray[np.float64] | None,
    tr: float | None = None,
    voxel_sizes: dict[str, float] | None = None,
    space_unit: str | None = None,
    time_unit: str | None = None,
    secondary_affine: npt.NDArray[np.float64] | None = None,
    primary_prefix: str = "sform",
    secondary_prefix: str = "qform",
) -> tuple[dict[str, xr.DataArray], dict[str, Any]]:
    """Create coordinate arrays and affine attributes from NIfTI affine matrices.

    When ``primary_affine`` is provided, spatial coordinates encode translation and zoom
    from the affine: each axis starts at ``affine[col, 3]`` (origin) and is stepped by
    the true voxel size along that axis (from ``decompose_affine``). The rotation and
    shear — the parts that cannot be encoded in 1D coordinates — are captured in a
    constant 4×4 affine ``A_physical`` that maps probe-space coordinates (in the units
    declared by the NIfTI header) to sform/qform world-space coordinates (same units).

    When ``primary_affine`` is ``None`` (both sform and qform codes are zero), spatial
    coordinates are built from ``voxel_sizes`` only: each axis starts at 0 and is stepped
    by the corresponding pixdim value (or 1 when the dimension is absent from
    ``voxel_sizes``). No ``"affines"`` entry is added to the returned attributes.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape in NIfTI order ``(x, y, z[, time])``.
    dims : tuple[str, ...]
        Dimension names in NIfTI order ``(x, y, z[, time])``.
    primary_affine : (4, 4) numpy.ndarray or None
        Primary affine matrix mapping voxel indices to world-space coordinates, or
        ``None`` when both sform and qform codes are zero (pixdim-only fallback).
    tr : float, optional
        Repetition time for the time coordinate, in the units given by
        ``time_unit``.
    voxel_sizes : dict[str, float] or None, optional
        Voxel dimensions keyed by dimension name (``"x"``, ``"y"``, ``"z"``),
        used to set the ``voxdim`` attribute on each spatial coordinate and as the
        coordinate step when ``primary_affine`` is ``None``.
    space_unit : str, optional
        Spatial unit string (e.g. ``"mm"``, ``"m"``, ``"um"``) set as the
        ``units`` attribute on each spatial coordinate. Omitted when not
        provided.
    time_unit : str, optional
        Temporal unit string (e.g. ``"s"``, ``"ms"``) set as the ``units``
        attribute on the time coordinate. Omitted when not provided.
    secondary_affine : (4, 4) numpy.ndarray, optional
        Optional secondary affine. Its physical transform is stored as an extra
        attribute keyed by ``secondary_prefix``. Ignored when ``primary_affine``
        is ``None``.
    primary_prefix : str, default: "sform"
        Attribute name prefix for the primary affine (``"sform"`` or
        ``"qform"``).
    secondary_prefix : str, default: "qform"
        Attribute name prefix for the secondary affine.

    Returns
    -------
    coords : dict[str, xarray.DataArray]
        Coordinate DataArrays keyed by name. When ``primary_affine`` is provided,
        spatial coordinates (``"x"``, ``"y"``, ``"z"``) start at
        ``affine[col, 3]`` and are stepped by the true voxel spacing. When
        ``primary_affine`` is ``None``, coordinates start at 0 and are stepped
        by ``voxel_sizes`` (defaulting to 1). A time coordinate is created if
        ``"time"`` is in ``dims`` and ``tr`` is provided.
    extra_attrs : dict[str, Any]
        Affine-derived DataArray attributes. When ``primary_affine`` is provided,
        contains ``"affines"``: a dict keyed by space name (``primary_prefix`` and
        optionally ``secondary_prefix``) mapping to a 4×4 probe→world affine in
        ConfUSIus (z, y, x) convention. Apply as ``A @ [pz, py, px, 1]`` to get
        ``[wz, wy, wx, 1]``. Both transforms share the same probe-space origin and
        spacing (NIfTI sform/qform describe the same voxel grid), so
        ``da.coords["z/y/x"]`` can be used with either transform. Empty when
        ``primary_affine`` is ``None``.
    """
    coords: dict[str, xr.DataArray] = {}
    extra_attrs: dict[str, Any] = {}

    if primary_affine is None:
        # Pixdim-only fallback: no affine available, no affines dict stored.
        for col, dim in enumerate(("x", "y", "z")):
            if dim not in dims:
                continue
            coord_attrs: dict[str, Any] = {}
            if space_unit is not None:
                coord_attrs["units"] = space_unit
            step = (
                voxel_sizes[dim]
                if voxel_sizes is not None and dim in voxel_sizes
                else 1.0
            )
            if voxel_sizes is not None and dim in voxel_sizes:
                coord_attrs["voxdim"] = voxel_sizes[dim]
            coords[dim] = xr.DataArray(
                step * np.arange(shape[col]),
                dims=[dim],
                attrs=coord_attrs,
            )
    else:
        affines_dict: dict[str, npt.NDArray[np.float64]] = {}

        nifti_affines: list[tuple[npt.NDArray[np.float64], str]] = [
            (primary_affine, primary_prefix),
        ]
        if secondary_affine is not None:
            nifti_affines.append((secondary_affine, secondary_prefix))

        for affine, prefix in nifti_affines:
            T, _, Z, _ = decompose_affine(affine)

            # Orientation matrix D satisfies D @ diag(Z) = RZS and captures rotation and
            # shear.
            D = affine[:3, :3] / Z

            # Build A_physical: the 4×4 affine mapping probe-space coordinates (in the
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
                if voxel_sizes is not None and dim in voxel_sizes:
                    coord_attrs_a["voxdim"] = voxel_sizes[dim]

                # Coordinates encode translation (T[col]) plus zoom (Z[col] from
                # decompose_affine, the true voxel spacing along this axis).
                coords[dim] = xr.DataArray(
                    T[col] + Z[col] * np.arange(shape[col]),
                    dims=[dim],
                    attrs=coord_attrs_a,
                )

        if affines_dict:
            extra_attrs["affines"] = affines_dict

    if "time" in dims:
        n_time = shape[dims.index("time")]
        time_values = np.arange(n_time) * tr if tr is not None else np.arange(n_time)
        time_attrs: dict[str, Any] = {}
        if time_unit is not None:
            time_attrs["units"] = time_unit
        coords["time"] = xr.DataArray(
            time_values,
            dims=["time"],
            attrs=time_attrs,
        )

    return coords, extra_attrs


def load_nifti(
    path: str | Path, chunks: int | tuple[int, ...] | str | None = "auto"
) -> xr.DataArray:
    """Load a NIfTI file as a lazy Xarray DataArray.

    Loads NIfTI files using nibabel's proxy arrays for memory-efficient access, wrapping
    the data in Dask arrays for chunked, parallel processing. The data is transposed to
    ConfUSIus conventions with dimensions ``(time, z, y, x)``.

    A BIDS-style JSON sidecar file (same name, ``.json`` extension) is loaded
    automatically when present.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NIfTI file (``.nii`` or ``.nii.gz``).
    chunks : int or tuple[int, ...] or str or None, default: "auto"
        How to chunk the array. Must be one of the following forms:

        - A blocksize like ``1000``.
        - A blockshape like ``(1000, 1000)``.
        - Explicit sizes of all blocks along all dimensions like ``((1000, 1000,
          500), (400, 400))``.
        - A size in bytes, like ``"100 MiB"`` which will choose a uniform block-like
          shape.
        - The word ``"auto"`` to let Dask choose chunk sizes based on heuristics. See
          `dask.array.normalize_chunks` for more details on how chunk sizes are
          determined.
        - ``-1`` or ``None`` as a blocksize indicate the size of the corresponding
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
    Physical-to-world affines are stored in ``da.attrs["affines"]``, a dict keyed by
    affine name. Each value is a 4×4 affine in ConfUSIus ``(z, y, x)`` convention that
    maps **physical coordinates** (as stored in ``da.coords``) to world-space
    coordinates. Apply as ``da.attrs["affines"]["physical_to_sform"] @ np.array([pz, py,
    px, 1.0])`` to get ``[wz, wy, wx, 1]``, where ``pz``, ``py``, ``px`` come from
    ``da.coords["z"]``, ``da.coords["y"]``, ``da.coords["x"]`` respectively.

    Unlike the NIfTI affine (which maps voxel *indices* to world space), the
    ``physical_to_*`` affines are invariant to any slicing or downsampling because they
    operate on physical positions, not grid indices.

    Affine selection follows NIfTI conventions:

    - If ``sform_code > 0``: sform is used as the primary affine; a
      ``"physical_to_sform"`` entry is written. When ``qform_code > 0`` as well, a
      ``"physical_to_qform"`` entry is also stored as secondary.
    - Else, if only ``qform_code > 0``: qform is used as the primary affine; only
      ``"physical_to_qform"`` is written.
    - If both codes are zero: a warning is emitted, coordinates are built from
      ``pixdim`` only (origin 0, step = voxel size), and no ``"affines"`` entry is
      stored in ``da.attrs``.

    The raw integer form codes are stored as ``da.attrs["qform_code"]`` and
    ``da.attrs["sform_code"]`` (only when > 0) so that a save/load roundtrip can
    reproduce the original NIfTI header codes.

    Voxel dimensions are stored in their native header units as a ``voxdim``
    attribute on each spatial coordinate array, consistent with the ``units``
    attribute of that coordinate.
    """
    path = check_path(path, type="file")

    img, attrs, voxel_sizes, tr, space_unit, time_unit = _load_nifti_with_nibabel(path)

    if path.suffix == ".gz" and path.stem.endswith(".nii"):
        sidecar_path = path.with_suffix("").with_suffix(".json")
    else:
        sidecar_path = path.with_suffix(".json")

    if sidecar_path.exists():
        with open(sidecar_path) as f:
            attrs.update(json.load(f))

    # We use np.asanyarray to get the memory-mapped array behind Nibabel's proxy. This
    # will allow Dask to create a lazy array without loading the entire dataset into
    # memory.
    data_obj = np.asanyarray(img.dataobj)

    # NIfTI stores data with shape (x, y, z, time) in column-major order.
    # ConfUSIus uses (time, z, y, x) in row-major order. These two conventions are
    # equivalent and one can be obtained from the other by transposing the array. This
    # will not copy data.
    data_obj = data_obj.T

    dask_arr = da.from_array(data_obj, chunks=chunks, asarray=False)

    ndim = dask_arr.ndim
    # NIfTI dim order is (x, y, z, time, ...); ConfUSIus order is the reverse.
    nifti_dims = _NIFTI_DIM_ORDER[:ndim]

    primary_affine, secondary_affine = _select_affines(img.header)

    if primary_affine is None:
        # Both sform_code and qform_code are 0: no spatial orientation is encoded in the
        # header. Coordinates are built from pixdim only (origin 0, step = voxel size).
        warnings.warn(
            "Both sform_code and qform_code are 0 in the NIfTI header. Coordinates "
            "will be computed from the voxel dimensions only, which may not reflect "
            "the true spatial orientation of the data.",
            stacklevel=find_stack_level(),
        )

    # The primary affine is the sform when sform_code > 0, otherwise qform. When
    # primary_affine is None (both codes 0), primary_prefix is unused.
    primary_prefix = "sform" if attrs.get("sform_code", 0) > 0 else "qform"

    coords, affine_attrs = _create_coords_from_nifti(
        shape=img.shape,
        dims=nifti_dims,
        primary_affine=primary_affine,
        tr=tr,
        voxel_sizes=voxel_sizes,
        space_unit=space_unit,
        time_unit=time_unit,
        secondary_affine=secondary_affine,
        primary_prefix=primary_prefix,
    )
    attrs.update(affine_attrs)

    # Override the pixdim-based time coordinate with sidecar timing fields when present.
    # Priority: VolumeTiming > RepetitionTime/DelayAfterTrigger > pixdim.
    if "time" in coords:
        time_attrs = coords["time"].attrs
        n_time = len(coords["time"])
        if "VolumeTiming" in attrs:
            coords["time"] = xr.DataArray(
                np.asarray(attrs.pop("VolumeTiming")),
                dims=["time"],
                attrs=time_attrs,
            )
        elif "RepetitionTime" in attrs:
            rep_time = float(attrs.pop("RepetitionTime"))
            delay = float(attrs.pop("DelayAfterTrigger", 0.0))
            if tr is not None and tr > 0 and not np.isclose(rep_time, tr, rtol=1e-3):
                warnings.warn(
                    f"Sidecar RepetitionTime ({rep_time}) does not match pixdim[4] "
                    f"({tr}) in the NIfTI header. Using sidecar value.",
                    stacklevel=find_stack_level(),
                )
            coords["time"] = xr.DataArray(
                delay + rep_time * np.arange(n_time),
                dims=["time"],
                attrs=time_attrs,
            )

    data_array = xr.DataArray(
        dask_arr,
        dims=nifti_dims[::-1],
        coords=coords,
        attrs=attrs,
        name="nifti_data",
    )

    return data_array


def _infer_repetition_time(
    times: npt.NDArray[np.float64],
) -> tuple[float | None, float]:
    """Infer the repetition time and onset delay from a time coordinate.

    Parameters
    ----------
    times : ndarray
        Time values, at least one element.

    Returns
    -------
    tr : float or None
        Uniform repetition time (spacing between volumes) when sampling is regular
        within ``rtol=1e-5``, or ``None`` for a single volume or irregular sampling.
    delay : float
        Onset time of the first volume (``times[0]``).
    """
    delay = float(times[0])
    if len(times) < 2:
        return None, delay
    diffs = np.diff(times)
    if np.allclose(diffs, diffs[0], rtol=1e-5):
        return float(diffs[0]), delay
    return None, delay


def _build_nifti_affine(
    transform: npt.NDArray[np.float64] | None,
    T: npt.NDArray[np.float64],
    Z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reconstruct a NIfTI affine from a stored ConfUSIus probe-to-world transform.

    Reverses the ConfUSIus ``(z, y, x)`` permutation to recover the orientation matrix
    ``D``, then combines it with the probe-coord origin ``T`` and spacing ``Z`` to
    rebuild the full NIfTI affine. Falls back to a diagonal affine when no transform is
    given.

    Parameters
    ----------
    transform : (4, 4) numpy.ndarray or None
        Probe-to-world affine in ConfUSIus ``(z, y, x)`` convention, or ``None`` to use
        a fallback diagonal affine.
    T : (3,) numpy.ndarray
        Origin of the probe coordinates for each NIfTI axis ``(x, y, z)``, in the
        spatial units of the DataArray coordinates.
    Z : (3,) numpy.ndarray
        Voxel spacing for each NIfTI axis ``(x, y, z)``, in the spatial units of the
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


def save_nifti(
    data_array: xr.DataArray,
    path: str | Path,
    nifti_version: NiftiVersion = 1,
    *,
    physical_to_qform: "npt.NDArray[np.float64] | None" = None,
    physical_to_sform: "npt.NDArray[np.float64] | None" = None,
    qform_code: int | None = None,
    sform_code: int | None = None,
) -> None:
    """Save an Xarray DataArray to NIfTI format.

    Saves the DataArray to a NIfTI file and always writes a BIDS-style JSON sidecar
    alongside it. The data is transposed to NIfTI convention ``(x, y, z, time)`` before
    saving.

    Parameters
    ----------
    data_array : xarray.DataArray
        DataArray to save.
    path : str or pathlib.Path
        Output path for the NIfTI file, with ``.nii`` or ``.nii.gz`` extension. If
        ``.nii.gz`` is used, the file will be saved in compressed format.
    nifti_version : {1, 2}, default: 1
        NIfTI format version to use. Version 2 is a simple extension to support
        larger files and arrays with dimension sizes greater than 32,767.
    physical_to_qform : (4, 4) numpy.ndarray, optional
        Affine mapping physical coordinates (as stored in the DataArray coordinates) to
        the NIfTI qform space, in ConfUSIus ``(z, y, x)`` convention. When provided,
        takes precedence over ``data_array.attrs["affines"]["physical_to_qform"]``. When
        not provided, the value from ``attrs["affines"]`` is used if present, otherwise
        a diagonal affine derived from voxel spacing is written.
    physical_to_sform : (4, 4) numpy.ndarray, optional
        Affine mapping physical coordinates (as stored in the DataArray coordinates) to
        the NIfTI sform space, in ConfUSIus ``(z, y, x)`` convention. When provided,
        takes precedence over ``data_array.attrs["affines"]["physical_to_sform"]``. When
        not provided, the value from ``attrs["affines"]`` is used if present; if neither
        is available, no sform is written.
    qform_code : int, optional
        NIfTI qform code to write. When provided, takes precedence over
        ``data_array.attrs["qform_code"]``. When not provided, the value from
        ``attrs["qform_code"]`` is used if present; otherwise defaults to ``1``.
    sform_code : int, optional
        NIfTI sform code to write. When provided, takes precedence over
        ``data_array.attrs["sform_code"]``. When not provided, the value from
        ``attrs["sform_code"]`` is used if present; otherwise defaults to ``1`` when a
        sform affine is available, or ``0`` when no sform affine is written.

    Examples
    --------
    >>> import confusius as cf
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(10, 32, 1, 64),
    ...                   dims=["time", "z", "y", "x"])
    >>> cf.io.save_nifti(da, "output.nii.gz")
    """
    import nibabel as nib

    path = Path(path)
    if path.suffix != ".nii" and path.suffixes[-2:] != [".nii", ".gz"]:
        raise ValueError("Output file must have .nii or .nii.gz extension.")

    data = np.asarray(data_array)

    current_dims = data_array.dims
    target_order = []
    for dim in ("x", "y", "z", "time"):
        if dim in current_dims:
            target_order.append(current_dims.index(dim))

    for i, dim in enumerate(current_dims):
        if dim not in ("x", "y", "z", "time"):
            target_order.append(i)

    # ConfUSIus order is (time, z, y, x), NIfTI order is (x, y, z, time).
    if len(target_order) == len(current_dims):
        data = np.transpose(data, target_order)

    # Insert singleton axes for any missing spatial dimensions so that the NIfTI axis
    # order (dim 0 = x, dim 1 = y, dim 2 = z) is always respected. Without this, a (z,
    # x) array would be saved as a 2D (x, z) NIfTI and reloaded as (x, y), corrupting
    # the coordinate labels.
    for insert_pos, dim in enumerate(("x", "y", "z")):
        if dim not in current_dims:
            data = np.expand_dims(data, axis=insert_pos)

    # .fusi.spacing handles voxdim fallback and warns on non-uniform/undefined dims.
    # NIfTI requires a concrete float; fall back to 1.0 for undefined spacing.
    spacing = data_array.fusi.spacing
    spatial_zooms = [
        abs(s) if (s := spacing.get(dim)) is not None else 1.0
        for dim in ("x", "y", "z")
    ]

    tr_pixdim: float | None = None
    time_sidecar: dict[str, Any] = {}
    if "time" in data_array.coords:
        time_values = data_array.coords["time"].values
        tr_spacing, delay = _infer_repetition_time(time_values)
        if tr_spacing is not None:
            tr_pixdim = tr_spacing
            time_sidecar["RepetitionTime"] = tr_spacing
            if not np.isclose(delay, 0.0):
                time_sidecar["DelayAfterTrigger"] = delay
        else:
            tr_pixdim = 0.0
            time_sidecar["VolumeTiming"] = time_values.tolist()

    zooms = spatial_zooms
    if "time" in current_dims:
        zooms.append(tr_pixdim if tr_pixdim is not None else 1.0)

    T = np.array(
        [
            float(data_array.coords[dim][0]) if dim in data_array.coords else 0.0
            for dim in ("x", "y", "z")
        ]
    )
    Z = np.array(spatial_zooms[:3])

    stored_affines: dict[str, Any] = data_array.attrs.get("affines", {})

    qform_matrix = (
        physical_to_qform
        if physical_to_qform is not None
        else stored_affines.get("physical_to_qform")
    )
    sform_matrix = (
        physical_to_sform
        if physical_to_sform is not None
        else stored_affines.get("physical_to_sform")
    )

    # Resolve xform codes: explicit kwargs > attrs > sensible defaults.
    # Default qform_code is 1 (scanner anatomical coordinates).
    # Default sform_code is 1 when a sform affine is present, 0 otherwise.
    resolved_qform_code: int
    if qform_code is not None:
        resolved_qform_code = qform_code
    elif "qform_code" in data_array.attrs:
        resolved_qform_code = int(data_array.attrs["qform_code"])
    else:
        resolved_qform_code = 1

    resolved_sform_code: int
    if sform_code is not None:
        resolved_sform_code = sform_code
    elif "sform_code" in data_array.attrs:
        resolved_sform_code = int(data_array.attrs["sform_code"])
    else:
        resolved_sform_code = 1 if sform_matrix is not None else 0

    qform_affine = _build_nifti_affine(qform_matrix, T, Z)
    sform_affine = (
        _build_nifti_affine(sform_matrix, T, Z) if resolved_sform_code > 0 else None
    )

    if nifti_version == 1:
        img_class = nib.Nifti1Image
    else:
        img_class = nib.Nifti2Image

    constructor_affine = sform_affine if sform_affine is not None else qform_affine
    nifti_img = img_class(data, constructor_affine)

    nifti_img.header.set_zooms(zooms)

    # qform is always written; it is the primary affine when no sform is stored.
    nifti_img.header.set_qform(qform_affine, code=resolved_qform_code)

    if sform_affine is not None:
        nifti_img.header.set_sform(sform_affine, code=resolved_sform_code)
    else:
        nifti_img.header.set_sform(None, code=0)

    space_unit_nib = None
    for dim in ("x", "y", "z"):
        if dim in data_array.coords and "units" in data_array.coords[dim].attrs:
            confusius_unit = data_array.coords[dim].attrs["units"]
            space_unit_nib = _CONFUSIUS_TO_NIFTI_SPACE_UNITS.get(confusius_unit)
            break

    time_unit_nib = None
    if "time" in data_array.coords and "units" in data_array.coords["time"].attrs:
        confusius_unit = data_array.coords["time"].attrs["units"]
        time_unit_nib = _CONFUSIUS_TO_NIFTI_TIME_UNITS.get(confusius_unit)

    if space_unit_nib is not None or time_unit_nib is not None:
        nifti_img.header.set_xyzt_units(
            xyz=space_unit_nib or "unknown",
            t=time_unit_nib or "unknown",
        )

    nifti_img.to_filename(path)

    sidecar_attrs = {
        k: v
        for k, v in data_array.attrs.items()
        # Exclude fields stored directly in the NIfTI header or derived from it.
        # "affines" is handled separately below.
        if k not in ("sform_code", "qform_code", "affines")
    }

    # Serialise affines that are not reconstructable from the NIfTI header (i.e.
    # everything except "sform" and "qform") as lists of lists.
    extra_affines = {
        k: np.asarray(v).tolist()
        for k, v in stored_affines.items()
        if k not in ("physical_to_sform", "physical_to_qform")
    }
    if extra_affines:
        sidecar_attrs["affines"] = extra_affines

    # Spatial coordinates and units are fully encoded in the NIfTI header (affine +
    # xyzt_units) and are not duplicated here. Time is stored as
    # RepetitionTime/DelayAfterTrigger (BIDS fields) for regular sampling, or as
    # VolumeTiming for irregular onset times.
    sidecar_attrs.update(time_sidecar)

    if path.suffix == ".gz":
        sidecar_path = path.with_suffix("").with_suffix(".json")
    else:
        sidecar_path = path.with_suffix(".json")

    with open(sidecar_path, "w") as f:
        json.dump(sidecar_attrs, f, indent=2, default=str)
