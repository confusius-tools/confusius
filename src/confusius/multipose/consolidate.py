"""Multi-pose volume consolidation.

This module provides functions for consolidating multi-pose acquisitions into
single volumes by merging the pose dimension into a spatial dimension.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import WORLD_DIMS
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_spatial_dims,
    get_voxel_to_world_units,
)
from confusius._utils.stack import find_stack_level
from confusius.multipose._utils import build_consolidated_time_coordinate
from confusius.validation import ensure_voxeldata


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


def _detect_sweep_dim(
    t: npt.NDArray[np.float64],
    rotation: npt.NDArray[np.float64],
    voxel_dims: Sequence[str],
) -> str:
    """Infer which voxel dimension is swept across poses from world-space geometry.

    The per-pose translation `t` isolates pure pose-stepping, independent of any
    voxel dimension (unlike diffing world positions along a candidate voxel
    dimension, which conflates pose-stepping with intra-pose stepping along that
    same dimension). Its dominant direction, found via SVD to stay robust to more
    than two poses, is dotted against each (normalized) column of the shared
    rotation block -- each column is the world-space direction of a unit step along
    the corresponding voxel dimension -- and the best-aligned voxel dimension is
    returned.

    Parameters
    ----------
    t : (npose, 3) numpy.ndarray
        Per-pose translation (world origin), read from the primary voxel-to-world
        affine stack.
    rotation : (3, 3) numpy.ndarray
        Shared rotation block of the primary voxel-to-world affine (constant across
        poses).
    voxel_dims : sequence[str]
        Voxel dimension names, in the same column order as `rotation`.

    Returns
    -------
    str
        Name of the voxel dimension best aligned with the pose-translation
        direction. A genuinely ambiguous choice (e.g. a raster sweep stepping along
        two voxel axes at once) is not flagged here: no single voxel dimension can
        span such a sweep at all, so [`consolidate_poses`][confusius.multipose.consolidate_poses]'s
        own regularity check rejects it downstream regardless of which dimension is
        picked.

    Raises
    ------
    ValueError
        If fewer than two poses are given, if poses share the same world position,
        or if the voxel-to-world geometry has a degenerate voxel axis.
    """
    if t.shape[0] < 2:
        raise ValueError("Cannot detect the swept voxel dimension from a single pose.")
    centered = t - t.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    if sv[0] == 0:
        raise ValueError(
            "Cannot detect the swept voxel dimension: poses have identical world "
            "positions."
        )
    pose_axis = vt[0]

    col_norms = np.linalg.norm(rotation, axis=0)
    if np.any(col_norms == 0):
        raise ValueError(
            "Cannot detect the swept voxel dimension: primary voxel-to-world "
            "geometry has a degenerate (zero-length) voxel axis."
        )
    alignments = np.abs(pose_axis @ rotation) / col_norms
    return voxel_dims[int(np.argmax(alignments))]


def consolidate_poses(
    da: xr.DataArray,
    rtol: float = 0.01,
    atol: float = 0.005,
) -> xr.DataArray:
    """Merge the `pose` dimension into the swept voxel dimension, ordered by position.

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

    The swept voxel dimension is detected from `da`'s own geometry: the per-pose
    translation's dominant direction (via SVD) is matched against each voxel
    dimension's world-space direction, and the best-aligned dimension is used. The
    primary sweep direction is then found via singular value decomposition of all
    positions along that dimension. Each voxel is projected onto that axis, the
    positions are checked for regularity, then the data is reindexed in ascending
    order along the consolidated sweep axis.

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
    rtol : float, default: 0.01
        Relative tolerance for the regularity check (fraction of mean spacing).
        Combined with `atol` as `abs(spacing - mean_spacing) <= atol + rtol *
        abs(mean_spacing)` (`numpy.isclose` convention), so `rtol` alone dominates at
        large step sizes.
    atol : float, default: 0.005
        Absolute tolerance in mm for the regularity check, combined with `rtol` as
        described above. The default (5 um) matches typical repeatability of
        stepper-motor-driven linear stages used for probe positioning, so it
        dominates at small step sizes (e.g. a 100 um step) where a pure relative
        tolerance would be unrealistically tight.

    Returns
    -------
    xarray.DataArray
        VoxelData array with `pose` merged into the swept voxel dimension, sorted by
        world position. Every voxel keeps its world position: the output
        voxel-to-world affine carries over the input's non-swept columns (including
        any rotation), uses one regular step along the detected sweep axis for the
        swept column, and is anchored at the first sorted voxel. For inputs whose
        `time` coordinate is itself
        pose-dependent (`(time, pose)`-shaped -- see
        [stack_poses][confusius.multipose.stack_poses]), a consolidated `slice_time`
        with dims `("time", <sweep_dim>)` is included: each slice inherits the
        timestamp of the pose it came from.

    Raises
    ------
    ValueError
        If `da` has no `pose` dimension, if `da`'s primary geometry is not
        pose-dependent, if the rotation block of the affine is not constant across
        poses (non-translation sweep), if the swept voxel dimension cannot be
        detected (fewer than two poses, identical pose positions, or a degenerate
        voxel axis), or if the consolidated positions are not regularly spaced
        within `atol`/`rtol` -- which also rejects a sweep that steps along more than
        one voxel axis at once, since no single voxel dimension can span it.

    Warns
    -----
    UserWarning
        If the sweep is not purely 1D (secondary/primary singular value ratio > 0.01).
    """
    da = ensure_voxeldata(da)

    if "pose" not in da.dims:
        raise ValueError("DataArray has no 'pose' dimension.")

    # ensure_voxeldata above guarantees da carries a VoxelToWorldIndex, so the voxel dims
    # (and their derived world coordinates) are always available here.
    voxel_dims = list(get_voxel_to_world_spatial_dims(da))

    affine = get_voxel_to_world_affine(da)  # (npose, 4, 4) when pose-dependent.
    if affine.ndim != 3:
        raise ValueError(
            "DataArray has no pose-dependent primary geometry; consolidate_poses "
            "requires a pose-dependent VoxelToWorldIndex (see "
            "VoxelToWorldIndex.is_pose_dependent). If per-pose geometry instead "
            "lives in a secondary, named affine in da.attrs['affines'], rebase onto "
            "it first with da.fusi.affine.apply(<key>)."
        )

    rotations: npt.NDArray[np.float64] = affine[:, :3, :3]  # (npose, 3, 3)
    t: npt.NDArray[np.float64] = affine[:, :3, 3]  # (npose, 3), per-pose translation.
    if not np.allclose(rotations, rotations[0], rtol=1e-6, atol=0):
        raise ValueError(
            "Rotation block of the primary voxel-to-world geometry is not constant "
            "across poses. consolidate_poses only supports pure translation sweeps."
        )

    sweep_dim = _detect_sweep_dim(t, rotations[0], voxel_dims)
    sweep_col = voxel_dims.index(sweep_dim)
    other_voxel_dims = [d for d in voxel_dims if d != sweep_dim]

    # Read exact per-(pose, sweep) world positions directly from da's own world
    # coordinates rather than reconstructing them from a separately stored affine.
    n_sweep = da.sizes[sweep_dim]
    world_positions = []
    for world_dim in WORLD_DIMS:
        coord = da.coords[world_dim]
        if other_voxel_dims:
            coord = coord.isel(dict.fromkeys(other_voxel_dims, 0))
        world_positions.append(np.asarray(coord.values, dtype=np.float64))
    # Each entry is (npose, n_sweep); stacked along a new last axis gives
    # (npose, n_sweep, 3).
    lab_pos: npt.NDArray[np.float64] = np.stack(world_positions, axis=-1)

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
    # A pure relative tolerance is unrealistically tight for small steps (e.g. 1% of
    # a 100 um step is 1 um -- below real stage repeatability); combine with a small
    # absolute tolerance so it dominates at small step sizes instead. See #363.
    if not np.allclose(diffs, mean_spacing, rtol=rtol, atol=atol):
        raise ValueError(
            f"Consolidated {sweep_dim} positions are not regularly spaced "
            f"(spacing range: [{diffs.min():.4f}, {diffs.max():.4f}] mm, "
            f"mean: {mean_spacing:.4f} mm, atol={atol}, rtol={rtol})."
        )

    pose_idx = sorted_flat // n_sweep
    sweep_idx = sorted_flat % n_sweep

    # Output voxel-to-world affine. The swept column is one regular step along the
    # sweep axis (the regularity check above ensures this is within rtol of the
    # actual positions, and a perfect grid avoids floating-point accumulation
    # errors); the other columns are constant across poses (checked above) and carry
    # over from the input; the origin is the world position of the first sorted
    # (pose, sweep) voxel. Every voxel thus keeps its world position through the
    # merge, including any rotation of the input geometry.
    new_affine: npt.NDArray[np.float64] = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = rotations[0]
    new_affine[:3, sweep_col] = sweep_axis * mean_spacing
    new_affine[:3, 3] = lab_pos_flat[sorted_flat[0]]

    new_affines = _consolidate_linked_affines(
        da.attrs.get("affines", {}), affine, new_affine
    )
    new_attrs = {**da.attrs, "affines": new_affines}
    base_coords: dict[str, Any] = {
        sweep_dim: np.arange(len(sorted_flat)),
        **{dim: np.arange(da.sizes[dim]) for dim in other_voxel_dims},
    }

    # Use xarray's vectorized isel to select (pose, sweep_dim) pairs simultaneously.
    # This stays dask-backed; dask does not support multi-axis fancy indexing via
    # da.data[...] (raises NotImplementedError for N-d fancy indexing).
    # The temporary dimension replaces both pose and sweep_dim; its position in the
    # output matches out_dims, so we can use .data directly without renaming.
    _consolidated = "__consolidated__"
    data = da.isel(
        {
            "pose": xr.DataArray(pose_idx, dims=[_consolidated]),
            sweep_dim: xr.DataArray(sweep_idx, dims=[_consolidated]),
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
        out_dims = ["time", sweep_dim] + other_voxel_dims
        return attach_voxel_to_world_index(
            xr.DataArray(
                data,
                dims=out_dims,
                coords=coords,
                attrs=new_attrs,
                name="scan_data",
            ),
            new_affine,
            units=get_voxel_to_world_units(da),
        )

    out_dims = [sweep_dim] + other_voxel_dims
    return attach_voxel_to_world_index(
        xr.DataArray(
            data,
            dims=out_dims,
            coords=base_coords,
            attrs=new_attrs,
            name="scan_data",
        ),
        new_affine,
        units=get_voxel_to_world_units(da),
    )
