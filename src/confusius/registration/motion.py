"""Motion parameter estimation and framewise displacement computation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from confusius.registration.affines import decompose_affine

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


def extract_motion_parameters(
    affines: Sequence[NDArray[np.float64] | None],
) -> NDArray[np.floating]:
    """Extract motion parameters from affine matrices.

    Decomposes each ``(N+1, N+1)`` homogeneous affine into translation and
    rotation parameters.

    For 2D transforms, extracts: ``[rotation, translation_x, translation_y]``.
    For 3D transforms, extracts:
    ``[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]``.

    Parameters
    ----------
    affines : list[numpy.ndarray | None]
        List of affine matrices from registration. ``None`` entries (e.g. from
        B-spline transforms) are treated as identity transforms.

    Returns
    -------
    (n_frames, n_params) numpy.ndarray
        Motion parameters array.

        - For 2D: ``n_params = 3 (rotation, tx, ty)``.
        - For 3D: ``n_params = 6 (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)``.

    Raises
    ------
    ValueError
        If any affine does not have shape ``(3, 3)`` (2D) or ``(4, 4)`` (3D).
    """
    import math

    params_list = []

    for affine in affines:
        if affine is None:
            # Non-linear (e.g. B-spline): treat as identity (zero motion). Dimension is
            # unknown; append a placeholder and fix after loop.
            params_list.append(None)
            continue

        ndim = affine.shape[0] - 1
        if ndim not in (2, 3):
            msg = f"Expected 2D or 3D affine, got shape {affine.shape}"
            raise ValueError(msg)

        T, R, _Z, _S = (
            decompose_affine(affine) if ndim == 3 else _decompose_affine_2d(affine)
        )

        if ndim == 2:
            # Rotation angle from the 2D rotation matrix.
            rotation = math.atan2(R[1, 0], R[0, 0])
            params_list.append([rotation, float(T[0]), float(T[1])])
        else:
            # Decompose the 3D rotation matrix into Euler angles (XYZ convention).
            rot_x, rot_y, rot_z = _rotation_matrix_to_euler_xyz(R)
            params_list.append(
                [rot_x, rot_y, rot_z, float(T[0]), float(T[1]), float(T[2])]
            )

    # Resolve None placeholders: infer dimensionality from the first non-None entry.
    ndim_inferred: int | None = None
    for p in params_list:
        if p is not None:
            # 3 params → 2D, 6 params → 3D.
            ndim_inferred = 2 if len(p) == 3 else 3
            break

    if ndim_inferred is None:
        # All transforms are None; default to 3D identity.
        ndim_inferred = 3

    zero_2d = [0.0, 0.0, 0.0]
    zero_3d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # ndim_inferred is spatial dimensionality (2 or 3), not n_params.
    n_params = 3 if ndim_inferred == 2 else 6
    resolved = [
        p if p is not None else (zero_2d if n_params == 3 else zero_3d)
        for p in params_list
    ]

    return np.array(resolved)


def _decompose_affine_2d(
    A33: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Decompose a (3, 3) 2D homogeneous affine into T, R, Z, S.

    Parameters
    ----------
    A33 : (3, 3) numpy.ndarray
        2D homogeneous affine.

    Returns
    -------
    T : (2,) numpy.ndarray
    R : (2, 2) numpy.ndarray
    Z : (2,) numpy.ndarray
    S : (1,) numpy.ndarray
    """
    import math

    T = A33[:2, 2].copy()
    RZS = A33[:2, :2]
    c0 = RZS[:, 0].copy()
    sx = math.sqrt(float(np.dot(c0, c0)))
    c0 /= sx
    sxy_sx = float(np.dot(c0, RZS[:, 1]))
    c1 = RZS[:, 1] - sxy_sx * c0
    sy = math.sqrt(float(np.dot(c1, c1)))
    c1 /= sy
    sxy = sxy_sx / sx
    R = np.array([[c0[0], c1[0]], [c0[1], c1[1]]])
    if np.linalg.det(R) < 0:
        sx *= -1
        R[:, 0] *= -1
    return T, R, np.array([sx, sy]), np.array([sxy])


def _rotation_matrix_to_euler_xyz(
    R: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Extract XYZ Euler angles from a (3, 3) rotation matrix.

    Parameters
    ----------
    R : (3, 3) numpy.ndarray
        Rotation matrix.

    Returns
    -------
    rot_x, rot_y, rot_z : float
        Rotation angles around X, Y, Z axes (radians).
    """
    import math

    # XYZ convention: R = Rz @ Ry @ Rx
    # R[2,0] = -sin(rot_y)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    rot_y = math.asin(sy)

    cos_y = math.cos(rot_y)
    if abs(cos_y) > 1e-6:
        rot_x = math.atan2(R[2, 1], R[2, 2])
        rot_z = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: set rot_x = 0 and compute rot_z.
        rot_x = 0.0
        rot_z = math.atan2(-R[0, 1], R[1, 1])

    return rot_x, rot_y, rot_z


def compute_framewise_displacement(
    affines: Sequence[NDArray[np.float64] | None],
    reference: "xr.DataArray",
    mask: NDArray[np.bool_] | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Compute framewise displacement from affine transforms.

    Framewise displacement measures how much voxels move between consecutive
    frames after registration. For each voxel, we compute the Euclidean distance
    between its position at frame t and frame t+1 after applying the affine
    transforms.

    Parameters
    ----------
    affines : list[numpy.ndarray | None]
        List of affine matrices, one per frame. ``None`` entries are treated as identity
        transforms.
    reference : xarray.DataArray
        Spatial DataArray defining the physical grid (spacing and origin derived from
        its coordinates).
    mask : numpy.ndarray, optional
        Boolean mask indicating which voxels to include. If not provided, uses all
        voxels.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"mean_fd"``: Mean framewise displacement per frame.
        - ``"max_fd"``: Maximum framewise displacement per frame.
        - ``"rms_fd"``: RMS framewise displacement per frame.
    """
    from confusius._utils import _compute_origin, _compute_spacing

    n_frames = len(affines)
    ndim = reference.ndim

    spacing_dict = _compute_spacing(reference)
    origin_dict = _compute_origin(reference)

    coords_1d = []
    for dim in (str(d) for d in reference.dims):
        sp = spacing_dict.get(dim) or 1.0
        orig = origin_dict.get(dim, 0.0)
        n = reference.sizes[dim]
        coords_1d.append(orig + np.arange(n) * sp)

    grids = np.meshgrid(*coords_1d, indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)  # (n_voxels, ndim)

    if mask is not None:
        mask_flat = mask.ravel()
        points = points[mask_flat]

    eye = np.eye(ndim + 1)
    transformed = []
    for affine in affines:
        A = affine if affine is not None else eye
        # Apply affine: p_out = A[:ndim, :ndim] @ p.T + A[:ndim, ndim]
        pts_out = (A[:ndim, :ndim] @ points.T).T + A[:ndim, ndim]
        transformed.append(pts_out)

    mean_fd = np.zeros(n_frames)
    max_fd = np.zeros(n_frames)
    rms_fd = np.zeros(n_frames)

    for t in range(n_frames - 1):
        displacements = np.linalg.norm(transformed[t + 1] - transformed[t], axis=1)
        mean_fd[t] = np.mean(displacements)
        max_fd[t] = np.max(displacements)
        rms_fd[t] = np.sqrt(np.mean(displacements**2))

    # Last frame has no next frame, so FD is 0.
    mean_fd[-1] = 0.0
    max_fd[-1] = 0.0
    rms_fd[-1] = 0.0

    return {
        "mean_fd": mean_fd,
        "max_fd": max_fd,
        "rms_fd": rms_fd,
    }


def create_motion_dataframe(
    affines: Sequence[NDArray[np.float64] | None],
    reference: "xr.DataArray",
    mask: NDArray[np.bool_] | None = None,
    time_coords: NDArray[np.floating] | None = None,
) -> "pd.DataFrame":
    """Create a DataFrame with motion parameters and framewise displacement.

    Parameters
    ----------
    affines : list[numpy.ndarray | None]
        List of affine matrices from registration. ``None`` entries (e.g. from B-spline
        transforms) are treated as identity.
    reference : xarray.DataArray
        Spatial DataArray defining the physical grid for framewise displacement
        computation.
    mask : numpy.ndarray, optional
        Boolean mask for FD computation.
    time_coords : numpy.ndarray, optional
        Time coordinates for each frame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        For 2D:

        - ``rotation``: Rotation angle in radians.
        - ``trans_x``: Translation in x (mm).
        - ``trans_y``: Translation in y (mm).

        For 3D:

        - ``rot_x, rot_y, rot_z``: Rotation angles in radians.
        - ``trans_x, trans_y, trans_z``: Translations (mm).

        Both:
        - ``mean_fd``: Mean framewise displacement (mm).
        - ``max_fd``: Maximum framewise displacement (mm).
        - ``rms_fd``: RMS framewise displacement (mm).
    """
    import pandas as pd

    params = extract_motion_parameters(affines)

    fd_dict = compute_framewise_displacement(affines, reference, mask)

    if params.shape[1] == 3:
        df = pd.DataFrame(
            {
                "rotation": params[:, 0],
                "trans_x": params[:, 1],
                "trans_y": params[:, 2],
                "mean_fd": fd_dict["mean_fd"],
                "max_fd": fd_dict["max_fd"],
                "rms_fd": fd_dict["rms_fd"],
            }
        )
    else:
        df = pd.DataFrame(
            {
                "rot_x": params[:, 0],
                "rot_y": params[:, 1],
                "rot_z": params[:, 2],
                "trans_x": params[:, 3],
                "trans_y": params[:, 4],
                "trans_z": params[:, 5],
                "mean_fd": fd_dict["mean_fd"],
                "max_fd": fd_dict["max_fd"],
                "rms_fd": fd_dict["rms_fd"],
            }
        )

    if time_coords is not None:
        df.index = time_coords
        df.index.name = "time"
    else:
        df.index.name = "frame"

    return df
