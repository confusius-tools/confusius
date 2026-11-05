"""Affine matrix utilities and world-space translate, rotate and zoom helpers."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.spatial.transform import Rotation

from confusius._dims import POSE_DIM, WORLD_DIMS
from confusius._utils.geometry import get_voxel_to_world_index_origin
from confusius.xarray.affine import apply_affine, get_geometry_center

if TYPE_CHECKING:
    import SimpleITK as sitk
    from SimpleITK import (
        AffineTransform,
        Euler2DTransform,
        Euler3DTransform,
        Similarity3DTransform,
        VersorRigid3DTransform,
    )

    _MatrixTransform = (
        AffineTransform
        | Euler2DTransform
        | Euler3DTransform
        | Similarity3DTransform
        | VersorRigid3DTransform
    )

type TransformCenter = (
    Literal["origin", "geometry"]
    | Sequence[float]
    | npt.NDArray[np.floating]
    | xr.DataArray
)
"""Fixed point of a world-space transform: a named point or a world `(z, y, x)` point."""

_MATRIX_TRANSFORMS = {
    "AffineTransform",
    "Euler2DTransform",
    "Euler3DTransform",
    "Similarity3DTransform",
    "VersorRigid3DTransform",
}
"""SimpleITK linear transform types that expose `GetMatrix`, `GetCenter`, and `GetTranslation`."""


def _striu2mat(
    striu: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Construct a shear matrix from the upper triangular vector.

    Parameters
    ----------
    striu : (N,) numpy.ndarray
        Vector giving triangle above diagonal of shear matrix. `N` must be a
        triangular number (1, 3, 6, 10, …).

    Returns
    -------
    SM : (M, M) numpy.ndarray
        Shear matrix, where `M` is the integer satisfying `M*(M-1)/2 == N`.

    Raises
    ------
    ValueError
        If `len(striu)` is not a triangular number.

    Notes
    -----
    Adapted from `transforms3d.affines.striu2mat` by Matthew Brett et al.
    (BSD-2-Clause License). See the `NOTICE` and `LICENSE-BSD-2-Clause`
    files for details.
    Source: https://github.com/matthew-brett/transforms3d
    """
    n = len(striu)
    N = (-1 + math.sqrt(8 * n + 1)) / 2.0 + 1
    if N != math.floor(N):
        raise ValueError(f"{n} is a strange number of shear elements")
    N = int(N)
    M = np.eye(N)
    inds = np.triu(np.ones((N, N)), 1).astype(bool)
    M[inds] = striu
    return M


def compose_affine(
    T: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating],
    Z: npt.NDArray[np.floating],
    S: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.float64]:
    """Compose translations, rotations, zooms, and shears into an affine matrix.

    Parameters
    ----------
    T : (N,) numpy.ndarray
        Translation vector, where `N` is usually 3 (3D case).
    R : (N, N) numpy.ndarray
        Rotation matrix, where `N` is usually 3 (3D case).
    Z : (N,) numpy.ndarray
        Zoom (scale) vector, where `N` is usually 3 (3D case).
    S : (P,) numpy.ndarray, optional
        Shear vector filling the upper triangle above the diagonal of the shear
        matrix. `P` is the `(N-2)`-th triangular number (3 for the 3D case).
        If not provided, no shear is applied.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Homogeneous affine transformation matrix.

    Notes
    -----
    Adapted from `transforms3d.affines.compose` by Matthew Brett _et al_.
    (BSD-2-Clause License). See the `NOTICE` and `LICENSE-BSD-2-Clause`
    files for details.
    Source: https://github.com/matthew-brett/transforms3d
    """
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError(f"Expected shape ({n},{n}) for R, got {R.shape}")

    A = np.eye(n + 1)
    ZS = np.diag(Z)

    if S is not None:
        ZS = ZS @ _striu2mat(S)

    A[:n, :n] = R @ ZS
    A[:n, n] = T
    return A


def decompose_affine(
    A44: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Decompose a 4x4 homogeneous affine into translation, rotation, zoom, shear.

    Decomposes `A44` into `T, R, Z, S` such that::

        Smat = np.array([[1, S[0], S[1]],
                         [0,    1, S[2]],
                         [0,    0,    1]])
        RZS = R @ np.diag(Z) @ Smat
        A44[:3, :3] = RZS
        A44[:3,  3] = T

    Parameters
    ----------
    A44 : (4, 4) numpy.ndarray
        Homogeneous affine matrix.

    Returns
    -------
    T : (3,) numpy.ndarray
        Translation vector.
    R : (3, 3) numpy.ndarray
        Rotation matrix.
    Z : (3,) numpy.ndarray
        Zoom (scale) vector. May have one negative zoom to avoid a negative
        determinant in `R`.
    S : (3,) numpy.ndarray
        Shear vector `[sxy, sxz, syz]` filling the upper triangle of the
        shear matrix. Zero for pure rotation/zoom affines.

    Notes
    -----
    Adapted from `transforms3d.affines.decompose44` by Matthew Brett et al.
    (BSD-2-Clause License). See the `NOTICE` and `LICENSE-BSD-2-Clause`
    files for details.
    Source: https://github.com/matthew-brett/transforms3d

    See also: Spencer W. Thomas, "Decomposing a matrix into simple
    transformations", pp 320-323 in *Graphics Gems II*, James Arvo (ed.),
    Academic Press, 1991.
    """
    A44 = np.asarray(A44)

    T = A44[:-1, -1]

    RZS = A44[:-1, :-1]
    M0, M1, M2 = np.array(RZS).T

    sx = math.sqrt(np.sum(M0**2))
    M0 /= sx

    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0

    sy = math.sqrt(np.sum(M1**2))
    M1 /= sy

    sxy = sx_sxy / sx
    sx_sxz = M0 @ M2
    sy_syz = M1 @ M2
    M2 -= sx_sxz * M0 + sy_syz * M1

    sz = math.sqrt(np.sum(M2**2))
    M2 /= sz

    sxz = sx_sxz / sx
    syz = sy_syz / sy
    Rmat = np.array([M0, M1, M2]).T

    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:, 0] *= -1

    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])


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
    >>> from confusius.registration import apply_translation
    >>> from confusius.xarray import create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> shifted = apply_translation(data, (0.0, 0.0, 1.5))
    >>> shifted.fusi.affine.voxel_to_world[:3, 3]
    array([0. , 0. , 1.5])
    """
    vector = _as_finite_array(translation, "translation", (3,))
    matrix = compose_affine(vector, np.eye(3), np.ones(3))
    return apply_affine(data, matrix, inplace=inplace)


def _resolve_center(
    data: xr.DataArray, center: TransformCenter
) -> npt.NDArray[np.float64]:
    """Return the world point a transform pivots about.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array being transformed.
    center : {"origin", "geometry"} or (3,) array-like
        `"origin"` for the world position of the first sampled voxel, `"geometry"` for
        the center of the bounding box, or a world point in `(z, y, x)` order.

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
    if not isinstance(center, str):
        return _as_finite_array(center, "center", (3,))
    if center == "geometry":
        return get_geometry_center(data).to_numpy()
    if center == "origin":
        # Pose-stacked geometry gets one matrix, so one pivot: the first pose's.
        first = data.isel({POSE_DIM: 0}) if POSE_DIM in data.dims else data
        origin = get_voxel_to_world_index_origin(first)
        return np.array([origin[name] for name in WORLD_DIMS])
    raise ValueError(
        f"center must be 'origin', 'geometry' or a (3,) world point, got {center!r}."
    )


def apply_rotation(
    data: xr.DataArray,
    angle: npt.ArrayLike,
    axis: str | npt.ArrayLike,
    *,
    center: TransformCenter = "origin",
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
    center : {"origin", "geometry"} or (3,) array-like, default: "origin"
        Fixed point of the rotation. `"origin"` is the grid origin, the world position
        of the first sampled voxel (of the first pose for pose-stacked geometry).
        `"geometry"` is the center of the bounding box, see
        [get_geometry_center][confusius.xarray.get_geometry_center]. Otherwise a world
        point in `(z, y, x)` order, in units of `data.fusi.affine.units`, such as
        `(0, 0, 0)` for the world origin.
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
    >>> from confusius.registration import apply_rotation, extract_motion_parameters
    >>> from confusius.xarray import create_voxeldata
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
    matrix = compose_affine(pivot - rotation @ pivot, rotation, np.ones(3))
    return apply_affine(data, matrix, inplace=inplace)


def apply_zoom(
    data: xr.DataArray,
    zoom: npt.ArrayLike,
    *,
    center: TransformCenter = "origin",
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
    center : {"origin", "geometry"} or (3,) array-like, default: "origin"
        Fixed point of the rescaling. `"origin"` is the grid origin, the world position
        of the first sampled voxel (of the first pose for pose-stacked geometry).
        `"geometry"` is the center of the bounding box, see
        [get_geometry_center][confusius.xarray.get_geometry_center]. Otherwise a world
        point in `(z, y, x)` order, in units of `data.fusi.affine.units`, such as
        `(0, 0, 0)` for the world origin.
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
        coordinates, if `data` is not a VoxelData array, or if a per-axis `zoom` is
        applied to pose-stacked geometry whose poses are not parallel (all pose affines
        must share equal spatial scale magnitudes).

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.registration import apply_zoom
    >>> from confusius.xarray import create_voxeldata
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
    matrix = compose_affine(pivot - factors * pivot, np.eye(3), factors)
    return apply_affine(data, matrix, inplace=inplace)


def affine_to_sitk_linear_transform(
    affine: npt.NDArray[np.floating],
) -> "AffineTransform":
    """Convert a homogeneous affine matrix to a SimpleITK `AffineTransform`.

    Parameters
    ----------
    affine : (4, 4) numpy.ndarray
        Homogeneous 3D affine matrix. The last row is expected to be `[0, 0, 0, 1]`.

    Returns
    -------
    AffineTransform
        SimpleITK affine transform with the matrix and translation set from
        `affine`. The rotation centre is left at the origin.
    """
    import SimpleITK as sitk

    ndim = affine.shape[0] - 1
    tx = sitk.AffineTransform(ndim)
    tx.SetMatrix(affine[:ndim, :ndim].flatten().tolist())
    tx.SetTranslation(affine[:ndim, ndim].tolist())
    return tx


def sitk_linear_transform_to_affine(
    transform: "sitk.Transform",
) -> npt.NDArray[np.floating]:
    """Convert a SimpleITK linear transform to a homogeneous affine matrix.

    Handles `TranslationTransform`, `Euler3DTransform`, `AffineTransform`,
    `VersorRigid3DTransform`, and `CompositeTransform` (by composing each
    sub-transform in order). Non-linear transforms (`BSplineTransform`,
    `DisplacementField`) are not supported.

    Parameters
    ----------
    transform : SimpleITK.Transform
        Estimated registration transform returned by SimpleITK.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Homogeneous affine matrix in world space mapping fixed-space points
        to moving-space points (pull/inverse convention used by SimpleITK's
        `Resample`).

    Raises
    ------
    AssertionError
        If called with a non-linear transform (`BSplineTransform`, `DisplacementField`).

    Notes
    -----
    SimpleITK uses a pull (inverse-mapping) convention: `Resample` iterates
    over fixed-grid points and applies the transform to look up the
    corresponding moving-image location. For linear transforms, the stored
    parameters encode:

    ```python
    p_moving = A @ (p_fixed - center) + center + translation
    ```

    where `A` is the `(N, N)` matrix returned by `GetMatrix()` and
    `center` is the rotation centre. This is equivalent to the homogeneous
    form::

        A_full[:N, :N] = A
        A_full[:N,  N] = -A @ center + center + translation
    """
    ndim = transform.GetDimension()

    name = transform.GetName()

    # Non-linear transforms cannot be represented as affine matrices. This
    # branch is unreachable from the current codebase: volume.py only calls
    # this function on the non-bspline path.
    assert "BSpline" not in name and "DisplacementField" not in name, (
        f"_sitk_linear_transform_to_affine called with non-linear transform '{name}'"
    )

    if name == "CompositeTransform":
        import SimpleITK as sitk

        # SimpleITK's CompositeTransform applies the last-added transform first
        # (outermost first). GetNthTransform(0) is the outermost, so the full
        # composite is T_0(T_1(...(T_n(p)))) = A_0 @ A_1 @ ... @ A_n @ p.
        # Right-multiply in forward order to build A_total = A_0 @ A_1 @ ... @ A_n.
        assert isinstance(transform, sitk.CompositeTransform)
        A = np.eye(ndim + 1)
        for i in range(transform.GetNumberOfTransforms()):
            A = A @ sitk_linear_transform_to_affine(transform.GetNthTransform(i))
        return A

    if name == "IdentityTransform":
        return np.eye(ndim + 1)

    if "Translation" in name:
        # TranslationTransform has no matrix, only an offset vector.
        A = np.eye(ndim + 1)
        A[:ndim, ndim] = np.array(transform.GetParameters())
        return A

    # All remaining supported linear transforms expose GetMatrix(), GetCenter(), and
    # GetTranslation(). Raise on anything unexpected. Cast to the _MatrixTransform
    # union: SimpleITK's stubs don't expose MatrixOffsetTransformBase as a Python class
    # so there is no common supertype to narrow to.
    if name not in _MATRIX_TRANSFORMS:
        raise ValueError(
            f"_sitk_linear_transform_to_affine: unsupported transform type {name!r}."
        )
    linear = cast("_MatrixTransform", transform)
    mat = np.array(linear.GetMatrix()).reshape(ndim, ndim)
    center = np.array(linear.GetCenter())
    translation = np.array(linear.GetTranslation())

    A = np.eye(ndim + 1)
    A[:ndim, :ndim] = mat
    # Combine rotation-centre offset with translation:
    # t_full = -mat @ center + center + translation
    A[:ndim, ndim] = -mat @ center + center + translation
    return A
