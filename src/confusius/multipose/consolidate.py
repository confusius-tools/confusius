"""Multi-pose volume consolidation.

This module provides functions for consolidating multi-pose acquisitions into
single volumes by merging the pose dimension into a spatial dimension.
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils import find_stack_level


def consolidate_poses(
    da: xr.DataArray,
    affines_key: str = "physical_to_lab",
    sweep_dim: str = "z",
    rtol: float = 0.01,
) -> xr.DataArray:
    """Merge `pose` and `sweep_dim` dimensions into a single axis ordered by position.

    For each `(pose, sweep_dim)` voxel, the position is computed using the sweep-dim
    column and translation of the per-pose affine (other spatial dims are zero at voxel
    centres along the sweep):

    ```python
    pos[p, i] = affine[p, :3, sweep_col] * sweep_mm[i] + affine[p, :3, 3]
    ```

    where `sweep_col` is the column index of `sweep_dim` in the spatial dim ordering
    `(z, y, x)` → columns `(0, 1, 2)`.

    The primary sweep direction is found via singular value decomposition of all
    positions. Each voxel is projected onto that axis, the positions are checked for
    regularity, then the data is reindexed in ascending order along the consolidated
    sweep axis.

    This function is primarily intended for consolidating multi-pose fUSI volumes
    acquired with an Iconeus system using a purely translational probe sweep. In that
    workflow, each pose corresponds to one probe position along the elevation axis
    (`z`), and the DataArray is produced by [`load_scan`][confusius.io.load_scan]:

    ```python
    scan_3d = load_scan("recording.scan")       # dims: (pose, z, y, x)
    volume  = consolidate_poses(scan_3d)        # dims: (z, y, x)

    scan_4d = load_scan("recording_4d.scan")    # dims: (time, pose, z, y, x)
    volume  = consolidate_poses(scan_4d)        # dims: (time, z, y, x)
    ```

    The function also works on any DataArray that carries a `(npose, 4, 4)` affine stack
    in `da.attrs["affines"][affines_key]` with columns ordered as `(z, y, x,
    translation)`. The `sweep_dim` parameter selects which spatial dimension is being
    swept across poses. For example, a stack of NIfTI DataArrays concatenated along a
    new `pose` dimension with their `physical_to_qform` affines stacked accordingly.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with a `pose` dimension and a `(npose, 4, 4)` affine stack stored
        in `da.attrs["affines"][affines_key]`. Typically produced by
        [`load_scan`][confusius.io.load_scan] for `3Dscan` or `4Dscan` files.
    affines_key : str, default: "physical_to_lab"
        Key into `da.attrs["affines"]` that holds the `(npose, 4, 4)` affine stack.
        Column order must be `(z, y, x, translation)`.
    sweep_dim : str, default: "z"
        Name of the spatial dimension being swept across poses. Must be one of the
        spatial dimensions in `da.dims`. The column index in the affine is determined
        by the order of spatial dimensions in the DataArray (e.g., if spatial dims are
        `["z", "y", "x"]`, then `"z"` → column 0, `"y"` → column 1, `"x"` → column 2).
    rtol : float, default: 0.01
        Relative tolerance for the regularity check (fraction of mean spacing).

    Returns
    -------
    xarray.DataArray
        DataArray with `pose` merged into `sweep_dim`, sorted by physical position.
        The consolidated `sweep_dim` coordinate holds the projection of each voxel's
        lab position (mm) onto the sweep axis. For inputs that carry a `pose_time`
        coordinate, a consolidated `pose_time` with dims `("time", sweep_dim)` is
        included: each slice inherits the timestamp of the pose it came from.

    Raises
    ------
    ValueError
        If `da` has no `pose` dimension, if `sweep_dim` is not one of the spatial
        dimensions in `da.dims`, if the rotation block of the affine is not constant
        across poses (non-translation sweep), or if the consolidated positions are not
        regularly spaced within `rtol`.

    Warns
    -----
    UserWarning
        If the sweep is not purely 1D (secondary/primary singular value ratio > 0.01).
    """
    # Determine spatial dimensions (non-time, non-pose) and their column indices.
    spatial_dims = [d for d in da.dims if d not in ("time", "pose")]
    if sweep_dim not in spatial_dims:
        raise ValueError(
            f"sweep_dim must be one of the spatial dimensions {spatial_dims!r}; "
            f"got {sweep_dim!r}."
        )
    sweep_col = spatial_dims.index(sweep_dim)

    if "pose" not in da.dims:
        raise ValueError("DataArray has no 'pose' dimension.")

    affine: npt.NDArray[np.float64] = np.asarray(
        da.attrs["affines"][affines_key]
    )  # (npose, 4, 4)
    sweep_mm: npt.NDArray[np.float64] = da.coords[sweep_dim].values
    n_sweep = len(sweep_mm)

    rotations: npt.NDArray[np.float64] = affine[:, :3, :3]  # (npose, 3, 3)
    if not np.allclose(rotations, rotations[0], rtol=1e-6, atol=0):
        raise ValueError(
            f"Rotation block of affines[{affines_key!r}] is not constant across poses. "
            "consolidate_poses only supports pure translation sweeps."
        )

    # Lab position for each (pose, sweep_dim): only the sweep column of the affine
    # contributes when we evaluate at the other spatial dims = 0 at voxel centres.
    # Rotation is constant across poses; only the translation T changes.
    r_sweep: npt.NDArray[np.float64] = affine[
        :, :3, sweep_col
    ]  # (npose, 3), sweep_dim direction in lab.
    t: npt.NDArray[np.float64] = affine[
        :, :3, 3
    ]  # (npose, 3), per-pose translation (mm).

    # Shape (npose, n_sweep, 3) -> (npose*n_sweep, 3).
    lab_pos: npt.NDArray[np.float64] = (
        r_sweep[:, np.newaxis, :] * sweep_mm[np.newaxis, :, np.newaxis]
        + t[:, np.newaxis, :]
    )
    lab_pos_flat = lab_pos.reshape(-1, 3)

    centered = lab_pos_flat - lab_pos_flat.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    if sv[0] > 0 and sv[1] / sv[0] > 0.01:
        warnings.warn(
            f"Sweep is not purely 1D: secondary/primary singular value ratio = "
            f"{sv[1] / sv[0]:.4f}. Projecting onto primary axis anyway.",
            stacklevel=find_stack_level(),
        )

    # Orient the sweep axis so the dominant component is positive.
    sweep_axis: npt.NDArray[np.float64] = vt[0]
    if sweep_axis[np.argmax(np.abs(sweep_axis))] < 0:
        sweep_axis = -sweep_axis

    proj: npt.NDArray[np.float64] = lab_pos_flat @ sweep_axis  # (npose*n_sweep,)
    sorted_flat: npt.NDArray[np.intp] = np.argsort(proj)
    proj_sorted = proj[sorted_flat]

    diffs: npt.NDArray[np.float64] = np.diff(proj_sorted)
    mean_spacing = float(np.mean(diffs))
    if not np.allclose(diffs, mean_spacing, rtol=rtol, atol=0):
        raise ValueError(
            f"Consolidated {sweep_dim} positions are not regularly spaced "
            f"(spacing range: [{diffs.min():.4f}, {diffs.max():.4f}] mm, "
            f"mean: {mean_spacing:.4f} mm, rtol={rtol})."
        )

    pose_idx = sorted_flat // n_sweep
    sweep_idx = sorted_flat % n_sweep

    # Replace the SVD-derived projections with a regular arithmetic grid anchored at
    # proj_sorted[0]. The regularity check above ensures this is within rtol of the
    # actual positions; using a perfect grid avoids floating-point accumulation errors
    # that would otherwise cause napari (which renders voxel k at origin + k * scale)
    # and resample_like (which reconstructs coords as origin + k * spacing) to disagree
    # with the coordinate array values.
    n_consolidated = len(proj_sorted)
    proj_regular: npt.NDArray[np.float64] = (
        proj_sorted[0] + np.arange(n_consolidated) * mean_spacing
    )

    new_sweep = xr.Variable(
        sweep_dim,
        proj_regular,
        attrs={"units": "mm", "voxdim": float(abs(mean_spacing))},
    )

    # After merging, sweep_dim is the projection along sweep_axis (already in mm lab
    # space), so the sweep column becomes sweep_axis. The other spatial columns are
    # constant across poses; the translation is the perpendicular component of the first
    # sorted pose's translation.
    t0 = t[pose_idx[0]]
    t_perp = t0 - np.dot(t0, sweep_axis) * sweep_axis
    other_cols = [c for c in range(3) if c != sweep_col]

    # Assemble a candidate rotation matrix and orthogonalise it via QR decomposition.
    # This guarantees a non-singular result: sweep_axis (SVD-derived) may not be
    # perfectly orthogonal to the columns taken from affine[0], which would otherwise
    # produce a singular matrix.  QR preserves the column ordering, so we place
    # sweep_axis in the sweep column and the pose-0 vectors in the remaining columns,
    # then decompose.  We fix signs afterward so each QR column points in the same
    # half-space as the corresponding candidate column, preserving the orientation of
    # the original per-pose affines.
    candidate: npt.NDArray[np.float64] = np.empty((3, 3), dtype=np.float64)
    candidate[:, sweep_col] = sweep_axis
    # affine[0, :3, other_cols] has shape (len(other_cols), 3) due to advanced
    # indexing; transpose to (3, len(other_cols)) before assigning.
    candidate[:, other_cols] = affine[0, :3, other_cols].T
    q, _ = np.linalg.qr(candidate)
    # Fix column signs: each QR column should agree in direction with the original
    # candidate column (positive dot product).
    for col in range(3):
        if np.dot(q[:, col], candidate[:, col]) < 0:
            q[:, col] = -q[:, col]

    new_affine: npt.NDArray[np.float64] = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = q
    new_affine[:3, 3] = t_perp

    other_dims = [d for d in spatial_dims if d != sweep_dim]
    new_attrs = {**da.attrs, "affines": {affines_key: new_affine}}
    base_coords: dict[str, Any] = {
        str(sweep_dim): new_sweep,
        **{str(d): da.coords[d] for d in other_dims},
    }

    # Build axis-agnostic fancy index tuples so that pose_idx selects the pose axis and
    # sweep_idx selects the sweep_dim axis, regardless of which spatial dim is being
    # swept.  For 3D inputs dims are (pose, z, y, x); for 4D (time, pose, z, y, x).
    # `sweep_dim_axis` is the axis of sweep_dim in da.values.
    sweep_dim_axis = da.dims.index(sweep_dim)
    pose_axis = da.dims.index("pose")

    # Build a full-array index with slice(None) for every axis, then replace the pose
    # and sweep axes with fancy indices. Broadcasting pose_idx and sweep_idx against
    # each other gives the consolidated axis of length npose*n_sweep.
    idx: list[Any] = [slice(None)] * da.values.ndim
    idx[pose_axis] = pose_idx
    idx[sweep_dim_axis] = sweep_idx

    if "time" in da.dims:
        coords: dict[str, Any] = {"time": da.coords["time"]}
        # Propagate per-slice timestamps if present: each consolidated slice inherits
        # the timestamp of the pose it came from.
        if "pose_time" in da.coords:
            coords["pose_time"] = xr.DataArray(
                da.coords["pose_time"].values[:, pose_idx],
                dims=["time", sweep_dim],
                attrs=da.coords["pose_time"].attrs,
            )
        coords.update(base_coords)
        out_dims = ["time", sweep_dim] + other_dims
        return xr.DataArray(
            da.values[tuple(idx)],
            dims=out_dims,
            coords=coords,
            attrs=new_attrs,
            name="scan_data",
        )

    out_dims = [sweep_dim] + other_dims
    return xr.DataArray(
        da.values[tuple(idx)],
        dims=out_dims,
        coords=base_coords,
        attrs=new_attrs,
        name="scan_data",
    )
