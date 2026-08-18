"""Multi-pose volume consolidation.

This module provides functions for consolidating multi-pose acquisitions into
single volumes by merging the pose dimension into a spatial dimension.
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_affine_axis_scalings,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
    get_voxel_to_world_spatial_dims,
)
from confusius._utils.stack import find_stack_level
from confusius.multipose._utils import build_consolidated_time_coordinate
from confusius.validation import ensure_fusi


def _consolidate_linked_affines(
    affines: dict[str, Any],
    main_per_pose: npt.NDArray[np.float64],
    main_consolidated: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Propagate secondary per-pose affines through pose consolidation.

    Primary geometry always lives in `da`'s `VoxelToWorldIndex` (`main_per_pose`,
    `main_consolidated`), never in `affines` -- so every entry here is a secondary,
    named affine (e.g. `world_to_brain`). One shaped like the main per-pose stack is
    assumed to be a constant left-link of it, i.e. there exists a constant `(4, 4)`
    matrix `L` such that `A[p] = L @ main_per_pose[p]` for all poses. The
    consolidated counterpart is then `L @ main_consolidated`. Affines that already
    have shape `(4, 4)` are passed through unchanged.

    Parameters
    ----------
    affines : dict[str, Any]
        Original `da.attrs["affines"]` mapping.
    main_per_pose : (npose, 4, 4) numpy.ndarray
        Per-pose stack of the primary affine, prior to consolidation.
    main_consolidated : (4, 4) numpy.ndarray
        Consolidated form of the primary affine.

    Returns
    -------
    dict[str, numpy.ndarray]
        Updated `affines` mapping with every per-pose-shaped entry consolidated
        accordingly.

    Raises
    ------
    ValueError
        If a per-pose-shaped affine does not satisfy `A[p] = L @ main_per_pose[p]`
        for a constant `L` to within numerical tolerance.
    """
    new_affines: dict[str, npt.NDArray[np.float64]] = {}
    main_inv0 = np.linalg.inv(main_per_pose[0])
    for key, value in affines.items():
        arr = np.asarray(value)
        if arr.shape == main_per_pose.shape:
            link = arr[0] @ main_inv0
            if not np.allclose(arr, link @ main_per_pose, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"Affine {key!r} is not a constant left-link of the primary "
                    "voxel-to-world geometry; cannot consolidate."
                )
            new_affines[key] = link @ main_consolidated
        else:
            new_affines[key] = arr
    return new_affines


def consolidate_poses(
    da: xr.DataArray,
    sweep_dim: str = "k",
    rtol: float = 0.01,
) -> xr.DataArray:
    """Merge `pose` and `sweep_dim` dimensions into a single axis ordered by position.

    Per-`(pose, sweep_dim)` world positions are read directly from `da`'s own world
    coordinates, which requires `da`'s primary voxel-to-world geometry to itself be
    pose-dependent (a `(npose, 4, 4)` affine stack — see
    [VoxelToWorldIndex.is_pose_dependent][confusius._utils.geometry.VoxelToWorldIndex.is_pose_dependent]).
    This is the case for SCAN data, where lab space (a fixed scanner frame shared by
    every pose) is the canonical world frame — see
    [`load_scan`][confusius.io.load_scan] — and for
    [`stack_poses`][confusius.multipose.stack_poses] output. If `da`'s primary
    geometry is instead driven by a *secondary*, named affine in
    `da.attrs["affines"]` (e.g. a stack of NIfTI DataArrays with their
    `world_to_qform` affines stacked, no pose-dependent primary geometry of their
    own), rebase onto it first with
    [`.fusi.affine.apply`][confusius.xarray.FUSIAffineAccessor.apply], e.g.
    `da.fusi.affine.apply("world_to_qform")`, before calling this function.

    The primary sweep direction is found via singular value decomposition of all
    positions. Each voxel is projected onto that axis, the positions are checked for
    regularity, then the data is reindexed in ascending order along the consolidated
    sweep axis.

    This function is primarily intended for consolidating multi-pose fUSI volumes
    acquired with an Iconeus system using a purely translational probe sweep. In that
    workflow, each pose corresponds to one probe position along the elevation axis
    (`k`/world `z`), and the VoxelData array is produced by
    [`load_scan`][confusius.io.load_scan]:

    ```python
    scan_3d = load_scan("recording.scan")       # dims: (pose, k, j, i)
    volume  = consolidate_poses(scan_3d)        # dims: (k, j, i)

    scan_4d = load_scan("recording_4d.scan")    # dims: (time, pose, k, j, i)
    volume  = consolidate_poses(scan_4d)        # dims: (time, k, j, i)
    ```

    Parameters
    ----------
    da : xarray.DataArray
        VoxelData array with a `pose` dimension and pose-dependent primary
        voxel-to-world geometry. Typically produced by
        [`load_scan`][confusius.io.load_scan] for `3Dscan` or `4Dscan` files, or by
        [`stack_poses`][confusius.multipose.stack_poses].
    sweep_dim : str, default: "k"
        Name of the voxel dimension being swept across poses. Must be one of the
        spatial dimensions in `da.dims`. The column index in the affine is determined
        by the voxel dimension order (`"k"` → column 0, `"j"` → column 1,
        `"i"` → column 2).
    rtol : float, default: 0.01
        Relative tolerance for the regularity check (fraction of mean spacing).

    Returns
    -------
    xarray.DataArray
        VoxelData array with `pose` merged into `sweep_dim`, sorted by world
        position. The consolidated `sweep_dim` coordinate holds the projection of
        each voxel's world position onto the sweep axis, expressed in the same
        units as the input `sweep_dim` coordinate. For inputs whose `time`
        coordinate is itself
        pose-dependent (`(time, pose)`-shaped -- see
        [stack_poses][confusius.multipose.stack_poses]), a consolidated `slice_time`
        with dims `("time", sweep_dim)` is included: each slice inherits the
        timestamp of the pose it came from.

    Raises
    ------
    ValueError
        If `da` has no `pose` dimension, if `sweep_dim` is not one of the spatial
        dimensions in `da.dims`, if `da`'s primary geometry is not pose-dependent, if
        the rotation block of the affine is not constant across poses
        (non-translation sweep), or if the consolidated positions are not regularly
        spaced within `rtol`.

    Warns
    -----
    UserWarning
        If the sweep is not purely 1D (secondary/primary singular value ratio > 0.01).
    """
    da = ensure_fusi(da)

    # Determine spatial dimensions (non-time, non-pose) and their column indices.
    spatial_dims = [d for d in da.dims if d not in ("time", "pose")]
    if sweep_dim not in spatial_dims:
        raise ValueError(
            f"sweep_dim must be one of the spatial dimensions {spatial_dims!r}; "
            f"got {sweep_dim!r}."
        )

    if "pose" not in da.dims:
        raise ValueError("DataArray has no 'pose' dimension.")

    # ensure_fusi above guarantees da carries a VoxelToWorldIndex, so the voxel dims
    # (and their derived world coordinates) are always available here.
    voxel_dims = list(get_voxel_to_world_spatial_dims(da))
    world_dims = list(get_voxel_to_world_coord_names(da))
    voxel_to_world = dict(zip(voxel_dims, world_dims, strict=True))
    if sweep_dim not in voxel_dims:
        raise ValueError(
            f"sweep_dim must be one of the spatial dimensions {voxel_dims!r}; "
            f"got {sweep_dim!r}."
        )
    sweep_data_dim = sweep_dim
    spatial_dims = voxel_dims
    output_spatial_dims = spatial_dims
    world_sweep_dim = voxel_to_world[sweep_dim]
    other_voxel_dims = [d for d in voxel_dims if d != sweep_dim]
    sweep_coord_attrs = dict(da.coords[world_sweep_dim].attrs)
    sweep_col = spatial_dims.index(sweep_dim)

    affine = get_voxel_to_world_affine(da)  # (npose, 4, 4) when pose-dependent.
    if affine.ndim != 3:
        raise ValueError(
            "DataArray has no pose-dependent primary geometry; consolidate_poses "
            "requires a pose-dependent VoxelToWorldIndex (see "
            "VoxelToWorldIndex.is_pose_dependent). If per-pose geometry instead "
            "lives in a secondary, named affine in da.attrs['affines'], rebase onto "
            "it first with da.fusi.affine.apply(<key>)."
        )

    # Read exact per-(pose, sweep) world positions directly from da's own world
    # coordinates rather than reconstructing them from a separately stored affine.
    n_sweep = da.sizes[sweep_dim]
    world_positions = []
    for world_dim in world_dims:
        coord = da.coords[world_dim]
        if other_voxel_dims:
            coord = coord.isel(dict.fromkeys(other_voxel_dims, 0))
        world_positions.append(np.asarray(coord.values, dtype=np.float64))
    # Each entry is (npose, n_sweep); stacked along a new last axis gives
    # (npose, n_sweep, 3).
    lab_pos: npt.NDArray[np.float64] = np.stack(world_positions, axis=-1)

    rotations: npt.NDArray[np.float64] = affine[:, :3, :3]  # (npose, 3, 3)
    t: npt.NDArray[np.float64] = affine[:, :3, 3]  # (npose, 3), per-pose translation.
    if not np.allclose(rotations, rotations[0], rtol=1e-6, atol=0):
        raise ValueError(
            "Rotation block of the primary voxel-to-world geometry is not constant "
            "across poses. consolidate_poses only supports pure translation sweeps."
        )

    # Shape (npose, n_sweep, 3) -> (npose*n_sweep, 3).
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

    new_sweep = xr.Variable(sweep_dim, proj_regular, attrs=sweep_coord_attrs)

    # After merging, sweep_dim is the projection along sweep_axis (already in affine/
    # world-space units), so the sweep column becomes sweep_axis. The other spatial
    # columns are
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

    other_output_dims = [d for d in output_spatial_dims if d != sweep_dim]
    new_affines = _consolidate_linked_affines(
        da.attrs.get("affines", {}), affine, new_affine
    )
    new_attrs = {**da.attrs, "affines": new_affines}
    base_coords: dict[str, Any] = {str(sweep_dim): new_sweep}
    for output_dim in other_output_dims:
        coord_name = voxel_to_world.get(output_dim, output_dim)
        if coord_name in da.coords:
            coord = da.coords[coord_name]
            other_coord_dims = [str(d) for d in coord.dims if d != output_dim]
            if other_coord_dims:
                # World coordinates are (k, j, i)-shaped (backed by a single joint
                # VoxelToWorldIndex covering all voxel dims), but only genuinely vary
                # along output_dim for axis-aligned geometry; reduce the other voxel
                # dims to get a 1D coordinate for the output.
                coord = coord.isel(dict.fromkeys(other_coord_dims, 0))
            base_coords[str(output_dim)] = xr.DataArray(
                np.asarray(coord.values),
                dims=[str(output_dim)],
                attrs=dict(coord.attrs),
            )

    output_spatial_dim_names = tuple(str(dim) for dim in output_spatial_dims)
    world_coord_names = tuple(
        str(voxel_to_world.get(dim, dim)) for dim in output_spatial_dims
    )

    # Non-sweep axes keep their pre-consolidation affine columns untouched, so their
    # true world spacing is that column's Euclidean norm (get_affine_axis_scalings) --
    # the affine is ground truth regardless of axis length, unlike diffing a single
    # named world coordinate. For an oblique geometry (e.g. SCAN data where a voxel
    # axis is not aligned with its "own" named world axis), diffing only that one
    # coordinate component picks up a near-zero cross term instead of the axis's real
    # magnitude, and a singleton axis has no diff to take at all.
    other_axis_scalings = get_affine_axis_scalings(affine[0], tuple(spatial_dims))

    def _attach_output_cti(result: xr.DataArray) -> xr.DataArray:
        world_attrs: dict[str, dict[str, Any]] = {}
        origins: list[float] = []
        spacings: list[float] = []
        for voxel_dim, world_dim in zip(
            output_spatial_dims, world_coord_names, strict=True
        ):
            coord = result.coords[voxel_dim]
            values = np.asarray(coord.values, dtype=np.float64)
            origins.append(float(values[0]))
            if voxel_dim == sweep_dim:
                # mean_spacing is the SVD-projected, pose-merged spacing -- finer
                # than any single pre-consolidation affine column, and well-defined
                # even when the consolidated axis ends up with a single sample.
                spacing = mean_spacing
            else:
                spacing = other_axis_scalings[voxel_dim]
            spacings.append(spacing)
            world_attrs[world_dim] = dict(coord.attrs)
            result = result.assign_coords(
                {voxel_dim: np.arange(result.sizes[voxel_dim])}
            )
        voxel_to_world_affine = np.eye(len(output_spatial_dim_names) + 1)
        voxel_to_world_affine[:-1, :-1] = np.diag(spacings)
        voxel_to_world_affine[:-1, -1] = origins
        return attach_voxel_to_world_index(
            result, voxel_to_world_affine, world_coord_attrs=world_attrs
        )

    # Use xarray's vectorized isel to select (pose, sweep_dim) pairs simultaneously.
    # This stays dask-backed; dask does not support multi-axis fancy indexing via
    # da.data[...] (raises NotImplementedError for N-d fancy indexing).
    # The temporary dimension replaces both pose and sweep_dim; its position in the
    # output matches out_dims, so we can use .data directly without renaming.
    _consolidated = "__consolidated__"
    data = da.isel(
        {
            "pose": xr.DataArray(pose_idx, dims=[_consolidated]),
            sweep_data_dim: xr.DataArray(sweep_idx, dims=[_consolidated]),
        }
    ).data

    if "time" in da.dims:
        time_coord = da.coords["time"]
        is_pose_dependent_time = time_coord.dims == ("time", "pose")

        if is_pose_dependent_time:
            # "time" is genuinely (time, pose)-shaped (see
            # confusius.multipose.stack_poses / confusius.io.load_scan's 4Dscan
            # output), holding each pose's own real timestamp directly. Build a 1D
            # base coordinate (pose 0's own column) to inherit units/reference
            # attrs from when building the consolidated "time" below.
            base_time_coord = xr.DataArray(
                np.asarray(time_coord.isel(pose=0).values),
                dims=["time"],
                attrs=dict(time_coord.attrs),
            )
        else:
            base_time_coord = time_coord

        coords: dict[str, Any] = {"time": base_time_coord}
        # Propagate per-slice timestamps: each consolidated slice inherits the
        # timestamp of the pose it came from.
        if is_pose_dependent_time:
            slice_time_attrs = dict(time_coord.attrs)
            slice_time_values = np.asarray(time_coord.values)[:, pose_idx]
            coords["slice_time"] = xr.DataArray(
                slice_time_values,
                dims=["time", sweep_dim],
                attrs=slice_time_attrs,
            )
            coords["time"] = build_consolidated_time_coordinate(
                base_time_coord,
                slice_time_values,
                slice_time_attrs,
            )
        coords.update(base_coords)
        out_dims = ["time", sweep_dim] + other_output_dims
        return _attach_output_cti(
            xr.DataArray(
                data,
                dims=out_dims,
                coords=coords,
                attrs=new_attrs,
                name="scan_data",
            )
        )

    out_dims = [sweep_dim] + other_output_dims
    return _attach_output_cti(
        xr.DataArray(
            data,
            dims=out_dims,
            coords=base_coords,
            attrs=new_attrs,
            name="scan_data",
        )
    )
