"""Motion parameter estimation and framewise displacement computation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from confusius.registration.affines import (
    decompose_affine,
    get_euler_xyz_from_rotation_matrix,
)

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
        If `affines` is empty, contains non-square arrays, unsupported affine shapes,
        or mixes 2D and 3D affines.
    TypeError
        If any entry is not a `numpy.ndarray`.
    """
    if len(affines) == 0:
        raise ValueError("affines must contain at least one affine matrix.")

    validated: list[NDArray[np.floating]] = []
    ndim: int | None = None

    for i, affine in enumerate(affines):
        if not isinstance(affine, np.ndarray):
            raise TypeError(
                f"affines[{i}] must be a numpy.ndarray, got {type(affine).__name__}."
            )
        if affine.ndim != 2 or affine.shape[0] != affine.shape[1]:
            raise ValueError(
                f"affines[{i}] must be a square 2D array, got shape {affine.shape}."
            )

        affine_ndim = affine.shape[0] - 1
        if affine_ndim not in (2, 3):
            raise ValueError(
                f"affines[{i}] must have shape (3, 3) or (4, 4), got {affine.shape}."
            )

        if ndim is None:
            ndim = affine_ndim
        elif affine_ndim != ndim:
            raise ValueError(
                "affines must all have the same dimensionality, got both "
                f"{ndim}D and {affine_ndim}D affines."
            )
        validated.append(affine)

    return validated


def extract_motion_parameters(
    affines: Sequence[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Extract motion parameters from affine matrices.

    Decomposes each `(N+1, N+1)` homogeneous affine into translation and
    rotation parameters.

    For 2D transforms, extracts: `[rotation, translation_0, translation_1]`.
    For 3D transforms, extracts:
    `[rotation_0, rotation_1, rotation_2, translation_0, translation_1, translation_2]`.

    Parameters
    ----------
    affines : list[numpy.ndarray]
        List of affine matrices from registration.

    Returns
    -------
    (n_frames, n_params) numpy.ndarray
        Motion parameters array in raw transform-component order.

        - For 2D: `n_params = 3 (rotation, t0, t1)`.
        - For 3D: `n_params = 6 (r0, r1, r2, t0, t1, t2)`.
    """
    import math

    affines_validated = _validate_affines(affines)
    params_list = []

    for affine in affines_validated:
        ndim = affine.shape[0] - 1

        T, R, _Z, _S = (
            decompose_affine(affine) if ndim == 3 else _decompose_affine_2d(affine)
        )

        if ndim == 2:
            rotation = math.atan2(R[1, 0], R[0, 0])
            params_list.append([rotation, float(T[0]), float(T[1])])
        else:
            rot_x, rot_y, rot_z = get_euler_xyz_from_rotation_matrix(R)
            params_list.append(
                [rot_x, rot_y, rot_z, float(T[0]), float(T[1]), float(T[2])]
            )

    return np.array(params_list)


# TODO: Temporary, to be removed once we enforce minimum dimensionality of 3D for all
# fUSI DataArrays.
def _decompose_affine_2d(
    A33: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
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
        If `params` does not have 3 or 6 columns.
    """
    spatial_dims = tuple(str(dim) for dim in reference.dims)
    axis_index = {dim: i for i, dim in enumerate(spatial_dims)}

    if params.shape[1] == 3:
        columns: list[tuple[str, int]] = [("rotation", 0)]
        for dim in ("x", "y", "z"):
            if dim in axis_index:
                columns.append((f"trans_{dim}", 1 + axis_index[dim]))
        return columns

    if params.shape[1] != 6:
        raise ValueError(
            f"Expected motion parameters with 3 or 6 columns, got {params.shape[1]}."
        )

    columns = []
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

    References
    ----------
    [^1]:
        Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L. & Petersen, S. E.
        Spurious but systematic correlations in functional connectivity MRI networks
        arise from subject motion. Neuroimage 59, 2142-2154
    """
    from confusius.validation import validate_fusi_dataarray

    validate_fusi_dataarray(
        reference,
        require_time=False,
        allow_pose=False,
        allow_extra_dims=False,
        minimum_spatial_dims=2,
        require_regular_spacing=True,
        regular_spacing_dims="space",
    )

    affines_validated = _validate_affines(affines)
    n_frames = len(affines_validated)
    ndim = affines_validated[0].shape[0] - 1
    if reference.ndim != ndim:
        raise ValueError(
            "reference dimensionality must match affine dimensionality, got "
            f"reference.ndim={reference.ndim} and affine ndim={ndim}."
        )

    coords_1d = [
        np.asarray(reference.coords[str(dim)].values, dtype=float)
        for dim in reference.dims
    ]

    grids = np.meshgrid(*coords_1d, indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)

    if mask is not None:
        points = points[mask.ravel()]

    transformed = []
    for affine in affines_validated:
        pts_out = (affine[:ndim, :ndim] @ points.T).T + affine[:ndim, ndim]
        transformed.append(pts_out)

    mean_fd = np.zeros(n_frames)
    max_fd = np.zeros(n_frames)
    rms_fd = np.zeros(n_frames)

    for t in range(n_frames - 1):
        displacements = np.linalg.norm(transformed[t + 1] - transformed[t], axis=1)
        mean_fd[t] = np.mean(displacements)
        max_fd[t] = np.max(displacements)
        rms_fd[t] = np.sqrt(np.mean(displacements**2))

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
        DataFrame with columns determined by the affine dimensionality.

        2D affines:

        - `rotation`: In-plane rotation angle in radians.
        - `trans_<dim>`: Translations along the DataArray's two named spatial axes
          (mm), reported in named-axis order.

        3D affines:

        - `rot_x, rot_y, rot_z`: Rotation angles around the DataArray's
          `x`, `y`, and `z` axes, in radians.
        - `trans_x, trans_y, trans_z`: Translations along the DataArray's
          `x`, `y`, and `z` axes (mm), even when one spatial axis is singleton.

        Both:
        - `mean_fd`: Mean framewise displacement (mm).
        - `max_fd`: Maximum framewise displacement (mm).
        - `rms_fd`: RMS framewise displacement (mm).

        Motion columns are reported in named-axis order (`x`, `y`, `z`), even when
        the input DataArray stores its spatial dimensions in a different order such as
        `(z, y, x)`.
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
