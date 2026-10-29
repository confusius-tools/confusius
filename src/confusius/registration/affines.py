"""Affine matrix decomposition and composition utilities."""

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import itk


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


def affine_to_itk_linear_transform(
    affine: npt.NDArray[np.floating],
) -> "itk.Transform":
    """Convert a homogeneous affine matrix to an ITK affine transform.

    Parameters
    ----------
    affine : (4, 4) numpy.ndarray
        Homogeneous 3D affine matrix. The last row is expected to be `[0, 0, 0, 1]`.

    Returns
    -------
    itk.Transform
        ITK affine transform with the matrix and translation set from
        `affine`. The rotation centre is left at the origin.
    """
    import itk

    ndim = affine.shape[0] - 1
    tx = itk.AffineTransform[itk.D, ndim].New()
    tx.SetMatrix(
        itk.matrix_from_array(
            np.ascontiguousarray(affine[:ndim, :ndim], dtype=np.float64)
        )
    )
    tx.SetTranslation(tuple(float(v) for v in affine[:ndim, ndim]))
    return tx


def itk_linear_transform_to_affine(
    transform: "itk.Transform",
) -> npt.NDArray[np.floating]:
    """Convert a ITK linear transform to a homogeneous affine matrix.

    Handles `TranslationTransform`, `Euler3DTransform`, `AffineTransform`,
    `VersorRigid3DTransform`, and `CompositeTransform` (by composing each
    sub-transform in order). Non-linear transforms (`BSplineTransform`,
    `DisplacementField`) are not supported.

    Parameters
    ----------
    transform : itk.Transform
        Estimated registration transform returned by ITK.

    Returns
    -------
    (N+1, N+1) numpy.ndarray
        Homogeneous affine matrix in world space mapping fixed-space points
        to moving-space points (pull/inverse convention used by ITK's
        `Resample`).

    Raises
    ------
    AssertionError
        If called with a non-linear transform (`BSplineTransform`, `DisplacementField`).

    Notes
    -----
    ITK uses a pull (inverse-mapping) convention: `Resample` iterates
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
    import itk

    transform = itk.down_cast(transform)
    if not transform.IsLinear():
        raise ValueError(
            f"Cannot represent nonlinear transform {transform.GetNameOfClass()!r} as an affine."
        )
    ndim = transform.GetInputSpaceDimension()
    affine = np.eye(ndim + 1)
    name = transform.GetNameOfClass()
    if name == "CompositeTransform":
        for index in range(transform.GetNumberOfTransforms()):
            affine = affine @ itk_linear_transform_to_affine(
                transform.GetNthTransform(index)
            )
    elif name == "TranslationTransform":
        affine[:ndim, ndim] = np.asarray(transform.GetParameters())
    elif name != "IdentityTransform":
        # Read matrix/offset directly; subtracting transformed points loses precision at large origins.
        affine[:ndim, :ndim] = itk.array_from_matrix(transform.GetMatrix())
        affine[:ndim, ndim] = transform.GetOffset()
    return affine
