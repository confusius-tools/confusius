"""Xarray accessor for affine transform operations."""

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.spatial.transform import Rotation

from confusius._dims import POSE_DIM, VOXEL_DIMS, WORLD_DIMS
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_index_spacing,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
    has_voxel_to_world_index,
)

type TransformCenter = (
    Literal["origin", "geometry"]
    | Sequence[float]
    | Mapping[str, float]
    | npt.NDArray[np.floating]
)
"""Fixed point of a world-space transform: a named point or a world `(z, y, x)` point."""


def get_bounding_box(data: xr.DataArray) -> dict[str, tuple[float, float]]:
    """Return the world-space bounding box of a VoxelData array.

    The box encloses the full extent of every voxel, not just the voxel centers: the
    voxel-coordinate range of each `k`/`j`/`i` dim is extended by half a voxel at
    both ends before mapping through the voxel-to-world affine. The voxel size is
    [`fusi.spacing`][confusius.xarray.FUSIAccessor.spacing] (so a singleton dim spans
    one voxel, and irregular voxel coordinates are rejected). The result is exact for
    any affine, including oblique ones, without
    materializing the lazily derived world coordinates: each world coordinate is an
    affine function of the voxel coordinates, so its extremes over the grid are
    attained at the corners of the (extended) voxel-coordinate box. For pose-dependent
    geometry, the box encloses every pose; select a pose first for its own box.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, normalized with
        [ensure_voxeldata][confusius.validation.ensure_voxeldata] (scalar-indexed
        voxel dims are restored as singletons before computing the bounds).

    Returns
    -------
    dict[str, tuple[float, float]]
        World coordinate names `z`, `y`, `x` mapped to `(min, max)` bounds in the
        DataArray's world units (`.fusi.affine.units`).

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If `data` is not a valid VoxelData array, or if a voxel dim has irregular
        coordinates (no defined voxel extent).

    Examples
    --------
    >>> bbox = get_bounding_box(volume)
    >>> bbox["z"][0]
    """
    from confusius.validation import ensure_voxeldata

    data = ensure_voxeldata(data)
    voxel_dims = get_voxel_to_world_spatial_dims(data)
    affine = get_voxel_to_world_affine(data)

    # Half a voxel per dim in voxel-coordinate units: `fusi.spacing` (world length per
    # voxel, `None` for irregular coordinates) divided by the affine column norm
    # (world length per voxel-coordinate unit).
    scalings = get_affine_axis_scalings(affine, voxel_dims)
    half_steps: dict[str, float] = {}
    for dim, world_spacing in get_voxel_to_world_index_spacing(data).items():
        if world_spacing is None:
            raise ValueError(
                f"Voxel dim '{dim}' has irregular coordinates; the voxel extent, and "
                "therefore the bounding box, is undefined."
            )
        half_steps[dim] = 0.5 * world_spacing / scalings[dim]

    # The 8 corners of the voxel-coordinate box extended by half a voxel per dim, as
    # homogeneous column vectors.
    corners = np.array(
        list(
            itertools.product(
                *(
                    (
                        data.coords[dim].values.min() - half_steps[dim],
                        data.coords[dim].values.max() + half_steps[dim],
                    )
                    for dim in voxel_dims
                )
            )
        ),
        dtype=np.float64,
    )
    corners_h = np.hstack([corners, np.ones((len(corners), 1))]).T

    # (npose, 4, 4) @ (4, 8) broadcasts to one corner set per pose.
    world_corners = (affine @ corners_h)[..., :-1, :]
    lower = world_corners.min(axis=-1)
    upper = world_corners.max(axis=-1)
    if affine.ndim == 3:
        lower = lower.min(axis=0)
        upper = upper.max(axis=0)
    return {dim: (float(lower[i]), float(upper[i])) for i, dim in enumerate(WORLD_DIMS)}


def get_geometry_center(data: xr.DataArray) -> dict[str, float]:
    """Return the world-space center of a VoxelData array.

    The center is the midpoint of the box returned by
    [get_bounding_box][confusius.xarray.get_bounding_box], so it is exact for any
    affine, including oblique ones. Pose-dependent geometry yields a single point, the
    center of the box enclosing every pose. Select a pose first to get that pose's own
    center.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, normalized with
        [ensure_voxeldata][confusius.validation.ensure_voxeldata].

    Returns
    -------
    dict[str, float]
        Center keyed by `z`, `y`, `x`, in the DataArray's world units
        (`.fusi.affine.units`).

    Raises
    ------
    TypeError
        If `data` is not an `xarray.DataArray`.
    ValueError
        If `data` is not a valid VoxelData array, or if a voxel dim has irregular
        coordinates (no defined voxel extent).

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.xarray import create_voxeldata, get_geometry_center
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> get_geometry_center(data)
    {'z': 0.5, 'y': 1.0, 'x': 1.5}
    """
    return {
        dim: (lower + upper) / 2
        for dim, (lower, upper) in get_bounding_box(data).items()
    }


def get_relative_affine(
    da: xr.DataArray,
    other: xr.DataArray,
    via: str,
) -> "npt.NDArray[np.float64]":
    """Return the affine mapping `da`'s world space into `other`'s.

    Computes `inv(other.attrs["affines"][via]) @ da.attrs["affines"][via]`,
    giving the transform that takes coordinates expressed in `da`'s
    world frame and expresses them in `other`'s world frame.  Both
    arrays must carry an `"affines"` dict in their `attrs` with the key
    `via`.

    Parameters
    ----------
    da : xarray.DataArray
        The source scan (origin world space).
    other : xarray.DataArray
        The scan whose world space is the target.
    via : str
        Key into `attrs["affines"]` that names the shared intermediate
        coordinate space used to bridge the two world frames (e.g.
        `"world_to_lab"`).

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Homogeneous affine matrix mapping `da`'s world coordinates
        to `other`'s world coordinates.

    Raises
    ------
    KeyError
        If `via` is not present in `da.attrs["affines"]` or
        `other.attrs["affines"]`.
    ValueError
        If either array does not have an `"affines"` entry in its `attrs`.
    """
    if "affines" not in da.attrs:
        raise ValueError("self does not have an 'affines' entry in attrs.")
    if "affines" not in other.attrs:
        raise ValueError("other does not have an 'affines' entry in attrs.")
    self_affine: npt.NDArray[np.float64] = np.asarray(
        da.attrs["affines"][via], dtype=np.float64
    )
    other_affine: npt.NDArray[np.float64] = np.asarray(
        other.attrs["affines"][via], dtype=np.float64
    )
    return np.linalg.inv(other_affine) @ self_affine


def apply_affine(
    da: xr.DataArray,
    affine: "npt.NDArray[np.float64] | str",
    inplace: bool = False,
) -> xr.DataArray:
    """Apply a world-space affine to a DataArray's world coordinates.

    The transform is composed into the DataArray's `VoxelToWorldIndex`, derived
    world coordinates are regenerated, and existing `attrs["affines"]` entries are
    re-expressed against the new world frame.

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-to-world geometry (a `VoxelToWorldIndex`).
    affine : (4, 4) numpy.ndarray or str
        Homogeneous world-space affine matrix to apply. If a string, it is looked up as
        a key in `da.attrs["affines"]`.
    inplace : bool, default: False
        Whether to modify the DataArray in-place. This requires all native voxel
        dimensions `k`, `j` and `i`, since a DataArray modified in-place cannot get back
        dimensions that scalar indexing removed.

    Returns
    -------
    xarray.DataArray
        `da` with updated spatial coordinates and updated `attrs["affines"]`. No
        entry is removed: when `affine` is a string, that key is kept as the
        identity, since the world frame now is that named space. A voxel dim removed
        by scalar indexing is restored as a singleton dim.

    Raises
    ------
    ValueError
        If `da` lacks voxel-to-world geometry, if `affine` shape does not match the
        DataArray's voxel-to-world affine, if `affine` is a string and `da` has no
        `"affines"` entry in `attrs`, or if `inplace` is set and `da` has a voxel dim
        removed by scalar indexing.
    KeyError
        If `affine` is a string not present in `da.attrs["affines"]`.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.xarray import create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)),
    ...     dims=("k", "j", "i"),
    ...     spacing=(1, 1, 1),
    ...     origin=(0, 0, 0),
    ... )
    >>> shift = np.eye(4)
    >>> shift[:3, 3] = [10.0, 5.0, 0.0]
    >>> result = data.fusi.affine.apply(shift)
    >>> float(result.fusi.affine.voxel_to_world[0, 3])
    10.0
    """
    from confusius.validation import ensure_voxeldata

    applied_key = affine if isinstance(affine, str) else None
    if applied_key is not None:
        if "affines" not in da.attrs:
            raise ValueError("da does not have an 'affines' entry in attrs.")
        if applied_key not in da.attrs["affines"]:
            raise KeyError(f"'{applied_key}' not found in da.attrs['affines'].")
        affine = da.attrs["affines"][applied_key]
    affine_array = np.asarray(affine, dtype=np.float64)

    if not has_voxel_to_world_index(da):
        raise ValueError("DataArray must have a voxel-to-world index.")

    # A scalar-indexed array has a rectangular voxel-to-world affine that no world-space
    # affine can match, so restore its voxel dims first. Complete arrays skip this
    # because `ensure_voxeldata` returns a new object, which would break `inplace=True`.
    if not set(VOXEL_DIMS) <= set(da.dims):
        if inplace:
            raise ValueError(
                f"inplace=True needs all native voxel dims {VOXEL_DIMS}, got dims "
                f"{da.dims}. Use inplace=False."
            )
        da = ensure_voxeldata(da)

    voxel_to_world = get_voxel_to_world_affine(da)
    is_pose_stacked = voxel_to_world.ndim == 3
    # A single (4, 4) world-space affine broadcasts over every pose (numpy's `@`
    # already does this correctly: `affine_array @ voxel_to_world[p]` per pose); a
    # pose-dependent affine_array must match the stack length exactly instead.
    valid_shape = (
        affine_array.shape == voxel_to_world.shape[1:]
        or affine_array.shape == voxel_to_world.shape
        if is_pose_stacked
        else affine_array.shape == voxel_to_world.shape
    )
    if not valid_shape:
        expected = (
            f"{voxel_to_world.shape[1:]} or {voxel_to_world.shape}"
            if is_pose_stacked
            else f"{voxel_to_world.shape}"
        )
        raise ValueError(
            "voxel-to-world data requires an affine with shape matching "
            f"voxel_to_world ({expected}), got {affine_array.shape}."
        )

    stored = da.attrs.get("affines", {})
    new_affines: dict[str, npt.NDArray[np.float64]] = {}
    inv_affine = np.linalg.inv(affine_array)
    for stored_key, val in stored.items():
        if stored_key == applied_key:
            # The world frame now is this named space. Store an exact identity
            # instead of `arr @ inv(arr)`, which carries floating-point error. A
            # single `(4, 4)` identity is valid even for pose-stacked data.
            new_affines[stored_key] = np.eye(affine_array.shape[-1])
            continue
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim in (2, 3):
            new_affines[stored_key] = arr @ inv_affine
        else:
            new_affines[stored_key] = arr

    new_attrs = dict(da.attrs)
    if "affines" in da.attrs:
        new_attrs["affines"] = new_affines

    result = attach_voxel_to_world_index(
        da.assign_attrs(new_attrs),
        affine_array @ voxel_to_world,
        units=get_voxel_to_world_units(da),
    )
    if inplace:
        da.coords.update(result.coords)
        da.attrs.clear()
        da.attrs.update(result.attrs)
        return da
    return result


def _as_finite_array(
    value: npt.ArrayLike, name: str, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Validate a finite real array of a given shape.

    Parameters
    ----------
    value : array-like
        Candidate array.
    name : str
        Argument name used in error messages.
    shape : tuple[int, ...]
        Required shape.

    Returns
    -------
    numpy.ndarray
        The array as `float64`, with shape `shape`.

    Raises
    ------
    ValueError
        If `value` is not a finite integer or float array of shape `shape`.
    """
    msg = f"{name} must be finite real numbers with shape {shape}."
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(msg) from error
    # Casting straight to float64 would parse strings, silently accepting `"2"`, and
    # would turn booleans into numbers.
    if (
        array.dtype.kind not in "iuf"
        or array.shape != shape
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(msg)
    return array.astype(np.float64)


def apply_translation(
    data: xr.DataArray, translation: npt.ArrayLike, *, inplace: bool = False
) -> xr.DataArray:
    """Translate a VoxelData array in world space.

    Only the voxel-to-world geometry changes: data values and voxel coordinates are
    untouched, and any `attrs["affines"]` entries are re-expressed accordingly.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to translate.
    translation : (3,) array-like
        Translation in world `(z, y, x)` order, in units of `data.fusi.affine.units`.
    inplace : bool, default: False
        Whether to modify `data` in place instead of returning a copy.

    Returns
    -------
    xarray.DataArray
        The translated VoxelData array. This is `data` itself if `inplace` is `True`.

    Raises
    ------
    ValueError
        If `translation` is not a finite `(3,)` vector, or if `data` is not a VoxelData
        array.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.xarray import apply_translation, create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> shifted = apply_translation(data, (0.0, 0.0, 1.5))
    >>> shifted.fusi.affine.voxel_to_world[:3, 3]
    array([0. , 0. , 1.5])
    """
    matrix = np.eye(4)
    matrix[:3, 3] = _as_finite_array(translation, "translation", (3,))
    return apply_affine(data, matrix, inplace=inplace)


def _resolve_center(
    data: xr.DataArray, center: TransformCenter
) -> npt.NDArray[np.float64]:
    """Return the world point a transform pivots about.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array being transformed.
    center : {"origin", "geometry"} or (3,) array-like or mapping
        `"origin"` for the first sampled voxel's world position, `"geometry"` for the
        center of the bounding box, a world point in `(z, y, x)` order, or a mapping
        with `z`, `y`, `x` keys.

    Returns
    -------
    (3,) numpy.ndarray
        The pivot in world `(z, y, x)` order.

    Raises
    ------
    ValueError
        If `center` is neither `"origin"`, `"geometry"` nor a finite `(3,)` vector, or
        if `data` cannot provide the requested pivot.
    """
    if isinstance(center, Mapping):
        missing = [dim for dim in WORLD_DIMS if dim not in center]
        if missing:
            raise ValueError(f"center mapping is missing world coordinates: {missing}.")
        center = [center[dim] for dim in WORLD_DIMS]
    if not isinstance(center, str):
        return _as_finite_array(center, "center", (3,))
    if center == "geometry":
        point = get_geometry_center(data)
        return np.array([point[dim] for dim in WORLD_DIMS])
    if center == "origin":
        from confusius.validation import ensure_voxeldata

        point = get_voxel_to_world_index_origin(ensure_voxeldata(data))
        return np.array([point[dim] for dim in WORLD_DIMS])
    raise ValueError(
        f"center must be 'origin', 'geometry' or a (3,) world point, got {center!r}."
    )


def apply_rotation(
    data: xr.DataArray,
    angle: npt.ArrayLike,
    axis: str | npt.ArrayLike,
    *,
    center: TransformCenter = "geometry",
    degrees: bool = True,
    inplace: bool = False,
) -> xr.DataArray:
    """Rotate a VoxelData array in world space.

    Only the voxel-to-world geometry changes: data values and voxel coordinates are
    untouched, and any `attrs["affines"]` entries are re-expressed accordingly.
    Rotations follow the right-hand rule in ordered world `(z, y, x)` coordinates, so a
    rotation of +90 degrees about `x` maps `+z` onto `+y`.
    With `axis="zyx"`, the angles match the rotations reported by
    [extract_motion_parameters][confusius.registration.extract_motion_parameters].

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to rotate.
    angle : float or (N,) array-like
        Rotation angle, or one angle per name when `axis` names several axes.
    axis : str or (3,) array-like
        Rotation axis. Either one or more world dimension names among `"z"`, `"y"` and
        `"x"`, or a vector in world `(z, y, x)` order, which is normalized. Several
        names rotate in sequence from left to right about the fixed world axes, so
        `"zyx"` rotates about `z` first. Names may repeat, as in `"zyz"`, and their
        order changes the result.
    center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
        Fixed point of the rotation. `"origin"` is the first sampled voxel's world
        position and requires selecting a scalar pose for pose-dependent data.
        `"geometry"` is the center of the bounding box, see
        [get_geometry_center][confusius.xarray.get_geometry_center]. Otherwise a world
        point in `(z, y, x)` order, in units of `data.fusi.affine.units`, such as
        `(0, 0, 0)` for the world origin. A mapping must contain `z`, `y`, and `x`
        keys; other keys are ignored.
    degrees : bool, default: True
        Whether to read `angle` in degrees rather than radians.
    inplace : bool, default: False
        Whether to modify `data` in place instead of returning a copy.

    Returns
    -------
    xarray.DataArray
        The rotated VoxelData array. This is `data` itself if `inplace` is `True`.

    Raises
    ------
    ValueError
        If `angle` is not finite, if it does not hold one value per axis name, if
        `axis` is neither world dimension names nor a non-zero finite `(3,)` vector,
        if `center` is neither `"origin"`, `"geometry"` nor a finite `(3,)` vector, if
        `center` is `"geometry"` and a voxel dim has irregular coordinates, or if `data`
        is not a VoxelData array.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.registration import extract_motion_parameters
    >>> from confusius.xarray import apply_rotation, create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> rotated = apply_rotation(data, 90.0, "x")
    >>> np.round(rotated.fusi.affine.voxel_to_world[:3, :3])
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    >>> tilted = apply_rotation(data, (30.0, 20.0, 10.0), "zyx")
    >>> params = extract_motion_parameters([tilted.fusi.affine.voxel_to_world])
    >>> np.round(np.rad2deg(params[0, :3]))
    array([30., 20., 10.])
    """
    axes = dict(zip(WORLD_DIMS, np.eye(3), strict=True))
    if isinstance(axis, str):
        if not axis or not set(axis) <= axes.keys():
            raise ValueError(
                f"axis must be one or more of 'z', 'y', 'x' or a vector, got {axis!r}."
            )
        unit_axes = np.array([axes[name] for name in axis])
    else:
        unit_axis = _as_finite_array(axis, "axis", (3,))
        norm = math.hypot(*unit_axis)
        if not 0 < norm < math.inf:
            raise ValueError("axis must have a non-zero, finite length.")
        unit_axes = unit_axis[np.newaxis] / norm

    angles = _as_finite_array(np.atleast_1d(angle), "angle", (len(unit_axes),))
    if degrees:
        angles = np.deg2rad(angles)

    # The rotation vectors stay in (z, y, x) component order, so the right-hand rule
    # applies to that ordered triad. Using (x, y, z) components instead would reverse
    # the sense of every rotation, since the two orders differ by an odd permutation.
    rotation = np.eye(3)
    for step in Rotation.from_rotvec(angles[:, np.newaxis] * unit_axes).as_matrix():
        rotation = step @ rotation
    pivot = _resolve_center(data, center)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot - rotation @ pivot
    return apply_affine(data, matrix, inplace=inplace)


def apply_zoom(
    data: xr.DataArray,
    zoom: npt.ArrayLike,
    *,
    center: TransformCenter = "geometry",
    inplace: bool = False,
) -> xr.DataArray:
    """Rescale a VoxelData array in world space.

    Only the voxel-to-world geometry changes: data values and voxel coordinates are
    untouched, and any `attrs["affines"]` entries are re-expressed accordingly.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to rescale.
    zoom : float or (3,) array-like
        Scale factor, either a scalar applied to all axes or per-axis factors in world
        `(z, y, x)` order. Negative factors reflect the volume along that axis.
        Unequal factors shear the voxel grid when it is oblique to the world axes, so
        rotating then zooming differs from zooming then rotating.
    center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
        Fixed point of the rescaling. `"origin"` is the first sampled voxel's world
        position and requires selecting a scalar pose for pose-dependent data.
        `"geometry"` is the center of the bounding box, see
        [get_geometry_center][confusius.xarray.get_geometry_center]. Otherwise a world
        point in `(z, y, x)` order, in units of `data.fusi.affine.units`, such as
        `(0, 0, 0)` for the world origin. A mapping must contain `z`, `y`, and `x`
        keys; other keys are ignored.
    inplace : bool, default: False
        Whether to modify `data` in place instead of returning a copy.

    Returns
    -------
    xarray.DataArray
        The rescaled VoxelData array. This is `data` itself if `inplace` is `True`.

    Raises
    ------
    ValueError
        If `zoom` is not a finite scalar or `(3,)` vector, if any factor is zero
        (singular affine), if `center` is neither `"origin"`, `"geometry"` nor a finite
        `(3,)` vector, if `center` is `"geometry"` and a voxel dim has irregular
        coordinates, if `data` is not a VoxelData array, or if the transformed pose
        affines no longer share equal spatial scale magnitudes.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.xarray import apply_zoom, create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> zoomed = apply_zoom(data, 2.0)
    >>> np.diag(zoomed.fusi.affine.voxel_to_world)
    array([2., 2., 2., 1.])
    """
    if np.ndim(zoom) == 0:
        zoom = np.full(3, zoom)
    factors = _as_finite_array(zoom, "zoom", (3,))
    if np.any(factors == 0):
        raise ValueError("zoom factors must be non-zero, got a singular scaling.")
    pivot = _resolve_center(data, center)
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(factors)
    matrix[:3, 3] = pivot - factors * pivot
    return apply_affine(data, matrix, inplace=inplace)


def apply_flip(
    data: xr.DataArray,
    axis: str,
    *,
    center: TransformCenter = "geometry",
    inplace: bool = False,
) -> xr.DataArray:
    """Reflect a VoxelData array along world axes without resampling.

    Only the voxel-to-world geometry changes: data values and voxel coordinates are
    untouched, and any `attrs["affines"]` entries are re-expressed accordingly.
    This is equivalent to negative zoom factors along the selected world axes,
    not reversing the voxel array along native `k`, `j`, or `i` dimensions.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to reflect.
    axis : str
        One or more distinct world dimension names among `"z"`, `"y"` and `"x"`,
        such as `"x"` or `"yx"`. Order does not matter.
    center : {"origin", "geometry"} or (3,) array-like or mapping, default: "geometry"
        Fixed point of the reflection. `"geometry"` is the center of the bounding
        box; `"origin"` is the first sampled voxel's world position and requires a
        scalar pose. Otherwise a world point in `(z, y, x)` order, in units of `data.fusi.affine.units`, or
        a mapping containing `z`, `y`, and `x` keys; other keys are ignored.
    inplace : bool, default: False
        Whether to modify `data` in place instead of returning a copy.

    Returns
    -------
    xarray.DataArray
        The reflected VoxelData array. This is `data` itself if `inplace` is `True`.

    Raises
    ------
    ValueError
        If `axis` is not a non-empty string of distinct world dimension names,
        if `center` is invalid, if `center` is `"geometry"` and a voxel dim has
        irregular coordinates, or if `data` is not a VoxelData array.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.xarray import apply_flip, create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> flipped = apply_flip(data, "x")
    >>> flipped.fusi.affine.voxel_to_world[2]
    array([ 0.,  0., -1.,  3.])
    """
    if (
        not isinstance(axis, str)
        or not axis
        or not set(axis) <= set(WORLD_DIMS)
        or len(set(axis)) != len(axis)
    ):
        raise ValueError(
            f"axis must be one or more distinct names among 'z', 'y', 'x', got {axis!r}."
        )
    factors = [-1.0 if name in axis else 1.0 for name in WORLD_DIMS]
    return apply_zoom(data, factors, center=center, inplace=inplace)


def reindex_voxels(da: xr.DataArray) -> xr.DataArray:
    """Rebase voxel coordinates to dense positions without moving world coordinates.

    A VoxelData array's stored `voxel_to_world` affine is defined in
    terms of voxel *coordinate values*, which stay unchanged across cropping or
    striding by design (see
    `VoxelToWorldIndex`).
    Because of this, the affine generally does not describe where voxel *position*
    `(0, ..., 0)` sits in world space, or the world distance between
    consecutive positions, once `da` has been cropped or strided from a larger
    array. This replaces each voxel dimension's coordinate with `0, 1, ..., dim - 1`
    and rebuilds `voxel_to_world` so the resulting affine directly maps those dense
    positions to `da`'s existing world coordinates, producing a DataArray whose
    affine is directly usable by software that assumes dense, zero-based voxel
    indices (e.g. ITK, nilearn).

    Parameters
    ----------
    da : xarray.DataArray
        Input scan with voxel-to-world geometry.

    Returns
    -------
    xarray.DataArray
        `da` with voxel coordinates rebased to `0, 1, ..., dim - 1` and an updated
        `voxel_to_world` affine. World coordinates are unchanged. For pose-dependent
        geometry, `voxel_to_world` stays a per-pose stack: each pose keeps its own
        origin and direction (spacing is shared, per the equal-scale invariant), so
        rebasing is unambiguous per pose and every pose is reindexed at once. A voxel
        dim removed by scalar indexing is restored as a singleton dim.

    Raises
    ------
    ValueError
        If `da` lacks voxel-to-world geometry, or if world spacing is undefined for
        any voxel dimension.
    """
    from confusius.validation import ensure_voxeldata

    if not has_voxel_to_world_index(da):
        raise ValueError("DataArray must have a voxel-to-world index.")

    # The affine rebuilt below needs one column per native voxel dim, so restore any dim
    # removed by scalar indexing. Complete arrays skip this to avoid the extra validation
    # in `ensure_voxeldata`.
    if not set(VOXEL_DIMS) <= set(da.dims):
        da = ensure_voxeldata(da)

    voxel_dims = get_voxel_to_world_spatial_dims(da)

    spacing = get_voxel_to_world_index_spacing(da)
    missing_spacing = [dim for dim in voxel_dims if spacing[dim] is None]
    if missing_spacing:
        raise ValueError(
            f"Cannot reindex voxels because spacing is undefined for dimensions "
            f"{missing_spacing!r}."
        )
    spacing_diag = np.diag([spacing[dim] for dim in voxel_dims])
    ndim = len(voxel_dims)

    def _new_pose_affine(pose_da: xr.DataArray) -> "npt.NDArray[np.float64]":
        origin = get_voxel_to_world_index_origin(pose_da)
        # Already expressed in dense-position terms (accounts for a descending
        # voxel coordinate, e.g. after `.isel(dim=slice(None, None, -1))`),
        # matching how `spacing` (a magnitude) and dense array position combine
        # below.
        direction = get_voxel_to_world_direction_matrix(pose_da)
        affine = np.eye(ndim + 1, dtype=np.float64)
        affine[:ndim, :ndim] = direction @ spacing_diag
        affine[:ndim, ndim] = [origin[name] for name in WORLD_DIMS]
        return affine

    if POSE_DIM in da.dims and get_voxel_to_world_affine(da).ndim == 3:
        new_affine = np.stack(
            [
                _new_pose_affine(da.isel({POSE_DIM: pose}))
                for pose in range(da.sizes[POSE_DIM])
            ]
        )
    else:
        new_affine = _new_pose_affine(da)

    reindexed = da.assign_coords({dim: np.arange(da.sizes[dim]) for dim in voxel_dims})
    return attach_voxel_to_world_index(
        reindexed, new_affine, units=get_voxel_to_world_units(da)
    )


def reindex_voxels_like(
    data: xr.DataArray, reference: xr.DataArray, *, atol: float = 1e-6
) -> xr.DataArray:
    """Rebase voxel coordinates onto `reference`'s voxel labels.

    `data` and `reference`'s `voxel_to_world` affines can differ even when they describe
    the exact same world grid: the affine is defined in terms of voxel *coordinate
    values*, so two arrays occupying identical world positions can still carry
    different affines if their voxel dimensions happen to be labeled differently (e.g.
    `reference` was cropped or strided from a larger array, while `data` was freshly
    built with dense labels). This verifies the two occupy the same world grid, then
    relabels `data`'s voxel coordinates and affine to match `reference`'s exactly, so
    the two become directly alignable (`.sel()`, arithmetic, `xarray.align`, ...) by
    voxel label as well as by world position.

    Parameters
    ----------
    data : xarray.DataArray
        Input scan with voxel-to-world geometry, physically aligned with `reference`.
    reference : xarray.DataArray
        DataArray whose voxel labels and affine `data` should adopt.
    atol : float, default: 1e-6
        Absolute tolerance, in `reference`'s physical units, for the world-coordinate
        alignment check between `data` and `reference`.

    Returns
    -------
    xarray.DataArray
        `data` with voxel coordinates and `voxel_to_world` replaced by `reference`'s.
        World coordinates are unchanged, since `data` and `reference` are verified to
        already occupy the same world grid.

    Raises
    ------
    ValueError
        If `data` or `reference` lacks voxel-to-world geometry, if their voxel
        dimensions or shapes differ, or if their world coordinates do not match
        within `atol`.
    """
    if not has_voxel_to_world_index(data):
        raise ValueError("data must have a voxel-to-world index.")
    if not has_voxel_to_world_index(reference):
        raise ValueError("reference must have a voxel-to-world index.")

    voxel_dims = get_voxel_to_world_spatial_dims(data)
    reference_voxel_dims = get_voxel_to_world_spatial_dims(reference)
    if voxel_dims != reference_voxel_dims:
        raise ValueError(
            f"data and reference must have the same voxel dimensions; got "
            f"{voxel_dims!r} and {reference_voxel_dims!r}."
        )
    shape = tuple(data.sizes[dim] for dim in voxel_dims)
    reference_shape = tuple(reference.sizes[dim] for dim in voxel_dims)
    if shape != reference_shape:
        raise ValueError(
            f"data and reference must have the same voxel grid shape to reindex; "
            f"got {shape!r} and {reference_shape!r}."
        )

    for name in WORLD_DIMS:
        data_values = data.coords[name].transpose(*voxel_dims).values
        reference_values = reference.coords[name].transpose(*voxel_dims).values
        if not np.allclose(data_values, reference_values, atol=atol):
            raise ValueError(
                f"data and reference are not aligned in world space: coordinate "
                f"{name!r} differs by more than {atol}. reindex_voxels_like requires "
                "data to already occupy reference's exact world grid."
            )

    relabeled = data.assign_coords({dim: reference.coords[dim] for dim in voxel_dims})
    return attach_voxel_to_world_index(
        relabeled,
        get_voxel_to_world_affine(reference),
        units=get_voxel_to_world_units(reference),
    )


class FUSIAffineAccessor:
    """Accessor for affine transform operations on VoxelData arrays.

    Provides methods to compute relative transforms between scans and to
    apply affines to a scan's spatial coordinates.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The `DataArray` to wrap.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def voxel_to_world(self) -> "npt.NDArray[np.float64]":
        """Affine mapping native voxel coordinates to world coordinates.

        Returns
        -------
        numpy.ndarray
            Homogeneous voxel-to-world affine.
        """
        return get_voxel_to_world_affine(self._obj)

    def set_voxel_to_world(
        self,
        voxel_to_world: "npt.ArrayLike",
        *,
        units: str | None = None,
        inplace: bool = False,
    ) -> xr.DataArray:
        """Replace voxel-to-world geometry.

        Parameters
        ----------
        voxel_to_world : numpy.typing.ArrayLike
            Homogeneous affine mapping native voxel coordinates to world coordinates.
        units : str, optional
            New physical unit shared by every world coordinate. If not provided,
            the existing unit is kept, or `"mm"` if there is none yet (e.g. when
            attaching geometry for the first time onto a plain, not-yet-indexed
            DataArray).
        inplace : bool, default: False
            Whether to modify the wrapped DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with rebuilt VoxelToWorldIndex-backed coordinates.
        """
        if units is None:
            if has_voxel_to_world_index(self._obj):
                units = get_voxel_to_world_units(self._obj)
            else:
                units = "mm"
        result = attach_voxel_to_world_index(self._obj, voxel_to_world, units=units)
        if inplace:
            self._obj.coords.update(result.coords)
            self._obj.attrs.clear()
            self._obj.attrs.update(result.attrs)
            return self._obj
        return result

    @property
    def units(self) -> str:
        """World-space unit shared by every world coordinate (e.g. `"mm"`).

        Returns
        -------
        str
            Physical unit shared by `z`/`y`/`x`, since they're derived jointly from
            one affine.
        """
        return get_voxel_to_world_units(self._obj)

    def set_units(self, units: str, *, inplace: bool = False) -> xr.DataArray:
        """Replace the world-space unit, rebuilding the VoxelToWorldIndex.

        Parameters
        ----------
        units : str
            New physical unit shared by every world coordinate.
        inplace : bool, default: False
            Whether to modify the wrapped DataArray in-place.

        Returns
        -------
        xarray.DataArray
            DataArray with rebuilt VoxelToWorldIndex-backed coordinates.
        """
        result = attach_voxel_to_world_index(
            self._obj, self.voxel_to_world, units=units
        )
        if inplace:
            self._obj.coords.update(result.coords)
            self._obj.attrs.clear()
            self._obj.attrs.update(result.attrs)
            return self._obj
        return result

    @property
    def bounding_box(self) -> dict[str, tuple[float, float]]:
        """World-space bounding box of the wrapped VoxelData array.

        Encloses the full extent of every voxel (half a voxel beyond the outermost
        voxel centers). For pose-dependent geometry, the box encloses every pose;
        select a pose first for its own box. See
        [get_bounding_box][confusius.xarray.get_bounding_box] for details.

        Returns
        -------
        dict[str, tuple[float, float]]
            World coordinate names `z`, `y`, `x` mapped to `(min, max)` bounds in
            `.fusi.affine.units`.

        Raises
        ------
        ValueError
            If `self` is not a valid VoxelData array, or if a voxel dim has irregular
            coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> from confusius.xarray import create_voxeldata
        >>> data = create_voxeldata(
        ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), spacing=(1.0, 1.0, 1.0)
        ... )
        >>> data.fusi.affine.bounding_box["x"][1]
        (-2.0, 2.0)
        """
        return get_bounding_box(self._obj)

    @property
    def geometry_center(self) -> dict[str, float]:
        """World-space center of the wrapped VoxelData array.

        The midpoint of
        [bounding_box][confusius.xarray.FUSIAffineAccessor.bounding_box]. Pose-dependent
        geometry yields a single point, the center of the box enclosing every pose. See
        [get_geometry_center][confusius.xarray.get_geometry_center] for details.

        Returns
        -------
        dict[str, float]
            Center keyed by `z`, `y`, `x`, in `.fusi.affine.units`.

        Raises
        ------
        ValueError
            If `self` is not a valid VoxelData array, or if a voxel dim has irregular
            coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> from confusius.xarray import create_voxeldata
        >>> data = create_voxeldata(
        ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), spacing=(1.0, 1.0, 1.0)
        ... )
        >>> data.fusi.affine.geometry_center["y"]
        1.5
        """
        return get_geometry_center(self._obj)

    def to(self, other: xr.DataArray, via: str) -> "npt.NDArray[np.float64]":
        """Return the affine mapping `self`'s world space into `other`'s.

        Computes `inv(other.attrs["affines"][via]) @ self.attrs["affines"][via]`,
        giving the transform from `self`'s world frame to `other`'s.

        Parameters
        ----------
        other : xarray.DataArray
            The scan whose world space is the target.
        via : str
            Key into `attrs["affines"]` naming the shared intermediate
            coordinate space (e.g. `"world_to_lab"`).

        Returns
        -------
        numpy.ndarray, shape (4, 4)
            Homogeneous affine matrix mapping `self`'s world coordinates
            to `other`'s world coordinates.

        Raises
        ------
        KeyError
            If `via` is not present in either scan's `attrs["affines"]`.
        ValueError
            If either scan has no `"affines"` entry in `attrs`.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import confusius  # noqa: F401
        >>> eye = np.eye(4)
        >>> a = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"to_world": eye}})
        >>> b = xr.DataArray(np.zeros((2, 2)), attrs={"affines": {"to_world": eye}})
        >>> np.allclose(a.fusi.affine.to(b, via="to_world"), np.eye(4))
        True
        """
        return get_relative_affine(self._obj, other, via)

    def apply(
        self,
        affine: "npt.NDArray[np.float64] | str",
        inplace: bool = False,
    ) -> xr.DataArray:
        """Apply a world-space affine to a DataArray's world coordinates.

        The transform is composed into the DataArray's `VoxelToWorldIndex`, derived
        world coordinates are regenerated, and existing `attrs["affines"]` entries
        are re-expressed against the new world frame.

        Parameters
        ----------
        affine : (4, 4) numpy.ndarray or str
            Homogeneous world-space affine matrix to apply. If a string, it is
            looked up as a key in `self.attrs["affines"]`.
        inplace : bool, default: False
            Whether to modify the DataArray in-place. This requires all native voxel
            dimensions `k`, `j` and `i`, since a DataArray modified in-place cannot get
            back dimensions that scalar indexing removed.

        Returns
        -------
        xarray.DataArray
            The DataArray with updated spatial coordinates and `attrs["affines"]`.
            No entry is removed: when `affine` is a string, that key is kept as the
            identity, since the world frame now is that named space. A voxel dim
            removed by scalar indexing is restored as a singleton dim.

        Raises
        ------
        ValueError
            If `self` lacks voxel-to-world geometry, if `affine` shape does not match
            the DataArray's voxel-to-world affine, if `affine` is a string and `self`
            has no `"affines"` entry in `attrs`, or if `inplace` is set and `self` has
            a voxel dim removed by scalar indexing.
        KeyError
            If `affine` is a string not present in `self.attrs["affines"]`.

        Examples
        --------
        >>> import numpy as np
        >>> from confusius.xarray import create_voxeldata
        >>> data = create_voxeldata(
        ...     np.zeros((2, 3, 4)),
        ...     dims=("k", "j", "i"),
        ...     spacing=(1, 1, 1),
        ...     origin=(0, 0, 0),
        ... )
        >>> shift = np.eye(4)
        >>> shift[:3, 3] = [10.0, 5.0, 0.0]
        >>> result = data.fusi.affine.apply(shift)
        >>> float(result.fusi.affine.voxel_to_world[0, 3])
        10.0
        """
        return apply_affine(self._obj, affine, inplace=inplace)

    def reindex_voxels(self, *, inplace: bool = False) -> xr.DataArray:
        """Rebase voxel coordinates to dense positions without moving world coordinates.

        See
        [reindex_voxels][confusius.xarray.affine.reindex_voxels]
        for details.

        Parameters
        ----------
        inplace : bool, default: False
            Whether to modify the wrapped DataArray in-place. This requires all native
            voxel dimensions `k`, `j` and `i`, since a DataArray modified in-place
            cannot get back dimensions that scalar indexing removed.

        Returns
        -------
        xarray.DataArray
            DataArray with voxel coordinates rebased to `0, 1, ..., dim - 1` and an
            updated `voxel_to_world` affine. World coordinates are unchanged. A voxel
            dim removed by scalar indexing is restored as a singleton dim.

        Raises
        ------
        ValueError
            If `self` lacks voxel-to-world geometry, if world spacing is undefined for
            any voxel dimension, or if `inplace` is set and `self` has a voxel dim
            removed by scalar indexing.

        Examples
        --------
        >>> import numpy as np
        >>> import confusius  # noqa: F401
        >>> from confusius.xarray import create_voxeldata
        >>> base = create_voxeldata(
        ...     np.zeros((5, 5)), dims=("j", "i"), voxel_to_world=np.eye(4)
        ... )
        >>> data = base.isel(j=slice(2, 5), i=slice(1, 5))
        >>> reindexed = data.fusi.affine.reindex_voxels()
        >>> reindexed.coords["j"].values
        array([0, 1, 2])
        >>> float(reindexed.coords["y"].isel(j=0, i=0, k=0))
        2.0
        """
        if inplace and not set(VOXEL_DIMS) <= set(self._obj.dims):
            raise ValueError(
                f"inplace=True needs all native voxel dims {VOXEL_DIMS}, got dims "
                f"{self._obj.dims}. Use inplace=False."
            )
        result = reindex_voxels(self._obj)
        if inplace:
            self._obj.coords.update(result.coords)
            self._obj.attrs.clear()
            self._obj.attrs.update(result.attrs)
            return self._obj
        return result

    def reindex_voxels_like(
        self, reference: xr.DataArray, *, atol: float = 1e-6, inplace: bool = False
    ) -> xr.DataArray:
        """Rebase voxel coordinates onto `reference`'s voxel labels.

        See
        [reindex_voxels_like][confusius.xarray.affine.reindex_voxels_like]
        for details.

        Parameters
        ----------
        reference : xarray.DataArray
            DataArray whose voxel labels and affine `self` should adopt.
        atol : float, default: 1e-6
            Absolute tolerance, in `reference`'s physical units, for the
            world-coordinate alignment check between `self` and `reference`.
        inplace : bool, default: False
            Whether to modify the wrapped DataArray in-place.

        Returns
        -------
        xarray.DataArray
            `self` with voxel coordinates and `voxel_to_world` replaced by
            `reference`'s. World coordinates are unchanged.

        Raises
        ------
        ValueError
            If `self` or `reference` lacks voxel-to-world geometry, if their voxel
            dimensions or shapes differ, or if their world coordinates do not
            match within `atol`.
        """
        result = reindex_voxels_like(self._obj, reference, atol=atol)
        if inplace:
            self._obj.coords.update(result.coords)
            self._obj.attrs.clear()
            self._obj.attrs.update(result.attrs)
            return self._obj
        return result
