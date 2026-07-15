"""Motion parameter estimation and framewise displacement computation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from confusius.registration.affines import decompose_affine

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


def _validate_affines(
    affines: Sequence[NDArray[np.floating]],
) -> list[NDArray[np.floating]]:
    """Return validated affine matrices for motion diagnostics.

    Parameters
    ----------
    affines : list[numpy.ndarray]
        Candidate affine matrices.

    Returns
    -------
    list[numpy.ndarray]
        Validated affine matrices.

    Raises
    ------
    ValueError
        If `affines` is empty or contains an array that is not a `(4, 4)` homogeneous
        3D affine.
    TypeError
        If any entry is not a `numpy.ndarray`.
    """
    if len(affines) == 0:
        raise ValueError("affines must contain at least one affine matrix.")

    validated: list[NDArray[np.floating]] = []

    for i, affine in enumerate(affines):
        if not isinstance(affine, np.ndarray):
            raise TypeError(
                f"affines[{i}] must be a numpy.ndarray, got {type(affine).__name__}."
            )
        if affine.shape != (4, 4):
            raise ValueError(
                f"affines[{i}] must be a (4, 4) 3D homogeneous affine, got shape "
                f"{affine.shape}."
            )
        validated.append(affine)

    return validated


def extract_motion_parameters(
    affines: Sequence[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Extract motion parameters from affine matrices.

    Decomposes each `(4, 4)` homogeneous 3D affine into translation and rotation
    parameters, extracting
    `[rotation_0, rotation_1, rotation_2, translation_0, translation_1, translation_2]`.

    Parameters
    ----------
    affines : list[numpy.ndarray]
        List of `(4, 4)` affine matrices from registration.

    Returns
    -------
    (n_frames, 6) numpy.ndarray
        Motion parameters array in raw transform-component order
        `(r0, r1, r2, t0, t1, t2)`.

    Raises
    ------
    ValueError
        If `affines` is empty or contains an array that is not a `(4, 4)` homogeneous
        3D affine.
    TypeError
        If any affine entry is not a `numpy.ndarray`.
    """
    affines_validated = _validate_affines(affines)
    params_list = []

    for affine in affines_validated:
        T, R, _Z, _S = decompose_affine(affine)
        rot_0, rot_1, rot_2 = _get_euler_xyz_from_rotation_matrix(R)
        params_list.append([rot_0, rot_1, rot_2, float(T[0]), float(T[1]), float(T[2])])

    return np.array(params_list)


def _get_euler_xyz_from_rotation_matrix(
    R: NDArray[np.floating],
) -> tuple[float, float, float]:
    """Get XYZ Euler angles from a 3D rotation matrix.

    Parameters
    ----------
    R : (3, 3) numpy.ndarray
        Rotation matrix.

    Returns
    -------
    rot_0 : float
        First XYZ Euler angle in radians.
    rot_1 : float
        Second XYZ Euler angle in radians.
    rot_2 : float
        Third XYZ Euler angle in radians.

    Raises
    ------
    ValueError
        If `R` is not a `(3, 3)` array.
    """
    import math

    if R.shape != (3, 3):
        raise ValueError(f"Expected a (3, 3) rotation matrix, got {R.shape}.")

    # XYZ convention: R = Rz @ Ry @ Rx.
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    rot_1 = math.asin(sy)

    cos_y = math.cos(rot_1)
    if abs(cos_y) > 1e-6:
        rot_0 = math.atan2(R[2, 1], R[2, 2])
        rot_2 = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: set the first angle to zero and recover the third.
        rot_0 = 0.0
        rot_2 = math.atan2(-R[0, 1], R[1, 1])

    return rot_0, rot_1, rot_2


def _get_motion_parameter_columns(
    reference: "xr.DataArray", params: NDArray[np.floating]
) -> list[tuple[str, int]]:
    """Return `(column_name, param_index)` pairs for `create_motion_dataframe`.

    Parameters
    ----------
    reference : xarray.DataArray
        Spatial reference DataArray whose dims and sizes define the named axes.
    params : numpy.ndarray
        Motion parameter array returned by `extract_motion_parameters`.

    Returns
    -------
    list[tuple[str, int]]
        Output motion columns paired with the source index in `params`.

    Raises
    ------
    ValueError
        If `params` does not have 6 columns.
    """
    spatial_dims = tuple(str(dim) for dim in reference.dims)
    axis_index = {dim: i for i, dim in enumerate(spatial_dims)}

    if params.shape[1] != 6:
        raise ValueError(
            f"Expected motion parameters with 6 columns, got {params.shape[1]}."
        )

    columns: list[tuple[str, int]] = []
    for dim in ("x", "y", "z"):
        columns.append((f"rot_{dim}", axis_index[dim]))
    for dim in ("x", "y", "z"):
        columns.append((f"trans_{dim}", 3 + axis_index[dim]))
    return columns


def compute_framewise_displacement(
    affines: Sequence[NDArray[np.floating]],
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
    affines : list[numpy.ndarray]
        List of affine matrices, one per frame.
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

        - `"mean_fd"`: Mean framewise displacement per frame.
        - `"max_fd"`: Maximum framewise displacement per frame.
        - `"rms_fd"`: RMS framewise displacement per frame.

    Raises
    ------
    ValueError
        If `reference` fails fUSI validation, or if `affines` is empty or contains an
        array that is not a `(4, 4)` homogeneous 3D affine.
    TypeError
        If any affine entry is not a `numpy.ndarray`.

    References
    ----------
    [^1]:
        Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L. & Petersen, S. E.
        Spurious but systematic correlations in functional connectivity MRI networks
        arise from subject motion. Neuroimage 59, 2142-2154
    """
    from confusius.validation import ensure_fusi_dataarray

    reference = ensure_fusi_dataarray(
        reference,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
        require_regular_spacing=True,
        regular_spacing_dims="space",
    )

    affines_validated = _validate_affines(affines)
    n_frames = len(affines_validated)
    # fUSI references and affines are both 3D after validation.
    ndim = 3

    coords_1d = [
        np.asarray(reference.coords[str(dim)].values, dtype=float)
        for dim in reference.dims
    ]

    grids = np.meshgrid(*coords_1d, indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)

    if mask is not None:
        points = points[mask.ravel()]

    mean_fd = np.zeros(n_frames)
    max_fd = np.zeros(n_frames)
    rms_fd = np.zeros(n_frames)

    # Framewise displacement is strictly pairwise, so transform one frame at a time and
    # retain only the previous frame's points instead of every frame's transformed grid.
    first = affines_validated[0]
    prev_pts = (first[:ndim, :ndim] @ points.T).T + first[:ndim, ndim]
    for t in range(n_frames - 1):
        nxt = affines_validated[t + 1]
        next_pts = (nxt[:ndim, :ndim] @ points.T).T + nxt[:ndim, ndim]
        displacements = np.linalg.norm(next_pts - prev_pts, axis=1)
        mean_fd[t] = np.mean(displacements)
        max_fd[t] = np.max(displacements)
        rms_fd[t] = np.sqrt(np.mean(displacements**2))
        prev_pts = next_pts

    mean_fd[-1] = 0.0
    max_fd[-1] = 0.0
    rms_fd[-1] = 0.0

    return {
        "mean_fd": mean_fd,
        "max_fd": max_fd,
        "rms_fd": rms_fd,
    }


def create_motion_dataframe(
    affines: Sequence[NDArray[np.floating]],
    reference: "xr.DataArray",
    mask: NDArray[np.bool_] | None = None,
    time_coords: NDArray[np.floating] | None = None,
) -> "pd.DataFrame":
    """Create a DataFrame with motion parameters and framewise displacement.

    Parameters
    ----------
    affines : list[numpy.ndarray]
        List of affine matrices from registration.
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
        DataFrame with the following columns:

        - `rot_x, rot_y, rot_z`: Rotation angles around the DataArray's
          `x`, `y`, and `z` axes, in radians.
        - `trans_x, trans_y, trans_z`: Translations along the DataArray's
          `x`, `y`, and `z` axes (mm), even when one spatial axis is singleton.
        - `mean_fd`: Mean framewise displacement (mm).
        - `max_fd`: Maximum framewise displacement (mm).
        - `rms_fd`: RMS framewise displacement (mm).

        Motion columns are reported in named-axis order (`x`, `y`, `z`), even when
        the input DataArray stores its spatial dimensions in a different order such as
        `(z, y, x)`.

    Raises
    ------
    ValueError
        If `reference` fails fUSI validation, or if `affines` is empty or contains an
        array that is not a `(4, 4)` homogeneous 3D affine.
    TypeError
        If any affine entry is not a `numpy.ndarray`.
    """
    import pandas as pd

    params = extract_motion_parameters(affines)
    fd_dict = compute_framewise_displacement(affines, reference, mask)

    motion_data = {
        name: params[:, index]
        for name, index in _get_motion_parameter_columns(reference, params)
    }
    df = pd.DataFrame(
        {
            **motion_data,
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
