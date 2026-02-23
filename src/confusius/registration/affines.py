"""Affine matrix decomposition and composition utilities."""

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import SimpleITK as sitk

__all__ = [
    "compose_affine",
    "decompose_affine",
    "sitk_transform_to_affine",
]


def _striu2mat(
    striu: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Construct a shear matrix from the upper triangular vector.

    Parameters
    ----------
    striu : (N,) numpy.ndarray
        Vector giving triangle above diagonal of shear matrix. ``N`` must be a
        triangular number (1, 3, 6, 10, …).

    Returns
    -------
    SM : (M, M) numpy.ndarray
        Shear matrix, where ``M`` is the integer satisfying ``M*(M-1)/2 == N``.

    Raises
    ------
    ValueError
        If ``len(striu)`` is not a triangular number.

    Notes
    -----
    Adapted from ``transforms3d.affines.striu2mat`` by Matthew Brett et al.
    (BSD-2-Clause License). See the ``NOTICE`` and ``LICENSE-BSD-2-Clause``
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
    T: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    Z: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Compose translations, rotations, zooms, and shears into an affine matrix.

    Parameters
    ----------
    T : (N,) numpy.ndarray
        Translation vector, where ``N`` is usually 3 (3D case).
    R : (N, N) numpy.ndarray
        Rotation matrix, where ``N`` is usually 3 (3D case).
    Z : (N,) numpy.ndarray
        Zoom (scale) vector, where ``N`` is usually 3 (3D case).
    S : (P,) numpy.ndarray, optional
        Shear vector filling the upper triangle above the diagonal of the shear
        matrix. ``P`` is the ``(N-2)``-th triangular number (3 for the 3D case).
        If ``None``, no shear is applied.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Homogeneous affine transformation matrix.

    Notes
    -----
    Adapted from ``transforms3d.affines.compose`` by Matthew Brett _et al_.
    (BSD-2-Clause License). See the ``NOTICE`` and ``LICENSE-BSD-2-Clause``
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
    A44: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Decompose a 4x4 homogeneous affine into translation, rotation, zoom, shear.

    Decomposes ``A44`` into ``T, R, Z, S`` such that::

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
        determinant in ``R``.
    S : (3,) numpy.ndarray
        Shear vector ``[sxy, sxz, syz]`` filling the upper triangle of the
        shear matrix. Zero for pure rotation/zoom affines.

    Notes
    -----
    Adapted from ``transforms3d.affines.decompose44`` by Matthew Brett et al.
    (BSD-2-Clause License). See the ``NOTICE`` and ``LICENSE-BSD-2-Clause``
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
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
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


def sitk_transform_to_affine(
    transform: "sitk.Transform",
) -> npt.NDArray[np.float64] | None:
    """Convert a SimpleITK linear transform to a homogeneous affine matrix.

    Handles ``TranslationTransform``, ``Euler2DTransform``, ``Euler3DTransform``,
    ``AffineTransform``, ``VersorRigid3DTransform``, and ``CompositeTransform``
    (by composing each sub-transform in order). Returns ``None`` for non-linear
    transforms such as ``BSplineTransform``, which cannot be represented as a
    finite affine matrix.

    Parameters
    ----------
    transform : SimpleITK.Transform
        Estimated registration transform returned by SimpleITK.

    Returns
    -------
    (N+1, N+1) numpy.ndarray or None
        Homogeneous affine matrix in physical space mapping fixed-space points
        to moving-space points (pull/inverse convention used by SimpleITK's
        ``Resample``), or ``None`` if the transform is non-linear (e.g. B-spline).

    Notes
    -----
    SimpleITK uses a pull (inverse-mapping) convention: ``Resample`` iterates
    over fixed-grid points and applies the transform to look up the
    corresponding moving-image location. For linear transforms, the stored
    parameters encode::

        p_moving = A @ (p_fixed - center) + center + translation

    where ``A`` is the ``(N, N)`` matrix returned by ``GetMatrix()`` and
    ``center`` is the rotation centre. This is equivalent to the homogeneous
    form::

        A_full[:N, :N] = A
        A_full[:N,  N] = -A @ center + center + translation
    """
    ndim = transform.GetDimension()

    name = transform.GetName()

    # Non-linear transforms cannot be represented as affine matrices.
    if "BSpline" in name or "DisplacementField" in name:
        return None

    if name == "CompositeTransform":
        # Compose sub-transforms in order (first applied last in matrix product). Type
        # stubs declare composite-specific methods only on CompositeTransform, so use
        # getattr to avoid type-checker errors.
        A = np.eye(ndim + 1)
        n_sub = getattr(transform, "GetNumberOfTransforms")()
        for i in range(n_sub):
            sub = getattr(transform, "GetNthTransform")(i)
            sub_affine = sitk_transform_to_affine(sub)
            if sub_affine is None:
                return None
            A = sub_affine @ A
        return A

    if name == "IdentityTransform":
        return np.eye(ndim + 1)

    if "Translation" in name:
        # TranslationTransform has no matrix, only an offset vector.
        A = np.eye(ndim + 1)
        A[:ndim, ndim] = np.array(transform.GetParameters())
        return A

    # All remaining linear transforms (Euler2D, Euler3D, Affine, Similarity, Versor…)
    # expose GetMatrix(), GetCenter(), and GetTranslation() directly. The type stubs
    # only declare these on concrete subclasses, so we use getattr to satisfy the
    # type checker without a runtime cast.
    mat = np.array(getattr(transform, "GetMatrix")()).reshape(ndim, ndim)
    center = np.array(getattr(transform, "GetCenter")())
    translation = np.array(getattr(transform, "GetTranslation")())

    A = np.eye(ndim + 1)
    A[:ndim, :ndim] = mat
    # Combine rotation-centre offset with translation:
    # t_full = -mat @ center + center + translation
    A[:ndim, ndim] = -mat @ center + center + translation
    return A
