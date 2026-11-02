"""Affine matrix utilities and world-space translate, rotate and zoom helpers."""

import math
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.spatial.transform import Rotation

from confusius.xarray.affine import apply_affine

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
) -> npt.NDArray[np.floating]:
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


def _as_finite_scalar(value: object, name: str) -> float:
    """Validate a finite real scalar.

    Parameters
    ----------
    value : object
        Candidate scalar.
    name : str
        Argument name used in error messages.

    Returns
    -------
    float
        The value as a finite Python float.

    Raises
    ------
    ValueError
        If `value` cannot be converted to a finite float.
    """
    # `float` parses strings, which would silently accept `"2"`.
    try:
        result = math.nan if isinstance(value, str) else float(cast(float, value))
    except (TypeError, ValueError):
        result = math.nan
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return result


def _as_world_vector(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Validate a world-space 3-vector given in `z/y/x` order.

    Parameters
    ----------
    value : array-like
        Candidate vector.
    name : str
        Argument name used in error messages.

    Returns
    -------
    (3,) numpy.ndarray
        The vector as a finite `float64` array.

    Raises
    ------
    ValueError
        If `value` is not convertible to a finite float array of shape `(3,)`.
    """
    msg = f"{name} must be a (3,) array-like of finite floats in world z/y/x order."
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(msg) from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(msg)
    return vector


def apply_translate(
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
        If `translation` is not a finite `(3,)` vector, or if `data` is not a
        VoxelData array.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.registration import apply_translate
    >>> from confusius.xarray import create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> shifted = apply_translate(data, (0.0, 0.0, 1.5))
    >>> shifted.fusi.affine.voxel_to_world[:3, 3]
    array([0. , 0. , 1.5])
    """
    matrix = np.eye(4)
    matrix[:3, 3] = _as_world_vector(translation, "translation")
    return apply_affine(data, matrix, inplace=inplace)


def _about_center(
    linear: npt.NDArray[np.float64], center: npt.ArrayLike | None
) -> npt.NDArray[np.float64]:
    """Embed a linear map in a homogeneous matrix pivoting about `center`.

    This is the closed form of `T(center) @ L @ T(-center)`.

    Parameters
    ----------
    linear : (3, 3) numpy.ndarray
        Linear part, in world `(z, y, x)` order.
    center : (3,) array-like, optional
        Fixed point of the map, in world `(z, y, x)` order. If not provided, the map
        pivots about the world origin.

    Returns
    -------
    (4, 4) numpy.ndarray
        Homogeneous matrix.

    Raises
    ------
    ValueError
        If `center` is not a finite `(3,)` vector.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    if center is not None:
        c = _as_world_vector(center, "center")
        matrix[:3, 3] = c - linear @ c
    return matrix


def apply_rotate(
    data: xr.DataArray,
    angle: float,
    axis: Literal["z", "y", "x"] | npt.ArrayLike,
    *,
    center: npt.ArrayLike | None = None,
    degrees: bool = True,
    inplace: bool = False,
) -> xr.DataArray:
    """Rotate a VoxelData array in world space.

    Only the voxel-to-world geometry changes: data values and voxel coordinates are
    untouched, and any `attrs["affines"]` entries are re-expressed accordingly.
    Rotations follow the right-hand rule in ordered world `(z, y, x)` coordinates, the
    frame napari displays, so a rotation of +90 degrees about `x` maps `+z` onto `+y`.
    Rotating about `z`, then `y`, then `x` gives the angles reported by
    [extract_motion_parameters][confusius.registration.extract_motion_parameters].

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to rotate.
    angle : float
        Rotation angle.
    axis : {"z", "y", "x"} or (3,) array-like
        Rotation axis, either a world dimension name or a vector in world
        `(z, y, x)` order. Vectors are normalized.
    center : (3,) array-like, optional
        Fixed point of the rotation in world `(z, y, x)` order, in units of
        `data.fusi.affine.units`. If not provided, the rotation is about the world
        origin. For single-pose data, use
        `data.fusi.affine.bounding_box.mean("bound").values` to rotate about the
        center of the volume.
    degrees : bool, default: True
        Whether `angle` is in degrees rather than radians.
    inplace : bool, default: False
        Whether to modify `data` in place instead of returning a copy.

    Returns
    -------
    xarray.DataArray
        The rotated VoxelData array. This is `data` itself if `inplace` is `True`.

    Raises
    ------
    ValueError
        If `angle` is not finite, if `axis` is not a world dimension name or a
        non-zero finite `(3,)` vector, if `center` is not a finite `(3,)` vector, or
        if `data` is not a VoxelData array.

    Examples
    --------
    >>> import numpy as np
    >>> from confusius.registration import apply_rotate
    >>> from confusius.xarray import create_voxeldata
    >>> data = create_voxeldata(
    ...     np.zeros((2, 3, 4)), dims=("k", "j", "i"), voxel_to_world=np.eye(4)
    ... )
    >>> rotated = apply_rotate(data, 90.0, "x")
    >>> np.round(rotated.fusi.affine.voxel_to_world[:3, :3])
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    """
    angle = _as_finite_scalar(angle, "angle")

    axes = {"z": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "x": (0.0, 0.0, 1.0)}
    if isinstance(axis, str):
        if axis not in axes:
            raise ValueError(
                f"axis must be one of 'z', 'y', 'x' or a vector, got {axis!r}."
            )
        unit_axis = np.array(axes[axis])
    else:
        unit_axis = _as_world_vector(axis, "axis")
        norm = np.linalg.norm(unit_axis)
        if norm == 0:
            raise ValueError("axis must have non-zero length.")
        unit_axis = unit_axis / norm

    angle_rad = np.deg2rad(angle) if degrees else float(angle)
    # The rotation vector stays in (z, y, x) component order, so the right-hand rule
    # applies to that ordered triad. Using (x, y, z) components instead would reverse
    # the sense of every rotation, since the two orders differ by an odd permutation.
    rotation = Rotation.from_rotvec(angle_rad * unit_axis).as_matrix()
    return apply_affine(data, _about_center(rotation, center), inplace=inplace)


def apply_zoom(
    data: xr.DataArray,
    zoom: float | npt.ArrayLike,
    *,
    center: npt.ArrayLike | None = None,
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
    center : (3,) array-like, optional
        Fixed point of the rescaling in world `(z, y, x)` order, in units of
        `data.fusi.affine.units`. If not provided, the volume is rescaled about the
        world origin. For single-pose data, use
        `data.fusi.affine.bounding_box.mean("bound").values` to rescale about the
        center of the volume.
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
        (singular affine), if `center` is not a finite `(3,)` vector, if `data` is not
        a VoxelData array, or if a per-axis `zoom` is applied to pose-stacked geometry
        whose poses are not parallel (all pose affines must share equal spatial scale
        magnitudes).

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
        factors = np.full(3, _as_finite_scalar(zoom, "zoom"), dtype=np.float64)
    else:
        factors = _as_world_vector(zoom, "zoom")
    if np.any(factors == 0):
        raise ValueError("zoom factors must be non-zero, got a singular scaling.")
    return apply_affine(data, _about_center(np.diag(factors), center), inplace=inplace)


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
