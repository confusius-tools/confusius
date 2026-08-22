"""Assembling single-pose VoxelData arrays into one pose-dependent DataArray."""

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, TIME_DIM
from confusius._utils.geometry import (
    attach_voxel_to_world_index,
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
)
from confusius.validation import ensure_voxeldata


def stack_poses(
    poses: Sequence[xr.DataArray],
    pose: Sequence[Hashable] | None = None,
) -> xr.DataArray:
    """Stack independently loaded single-pose VoxelData arrays into one pose-dependent array.

    `xr.concat` cannot combine `N` single-grid `VoxelToWorldIndex` objects into one
    joint pose-dependent index by itself: xarray's own pre-concat alignment step only
    excludes a coordinate from its equality check when that coordinate's *existing*
    index already spans the concat dimension, which a single-grid array's `z`/`y`/`x`
    index does not (it only spans `k`/`j`/`i`). This function closes that gap by first
    promoting each input to a genuinely pose-dependent index of length 1 (see
    [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]) via
    [attach_voxel_to_world_index][confusius._utils.geometry.attach_voxel_to_world_index]
    -- once every input's `z`/`y`/`x` index already spans `pose`, alignment correctly
    excludes it and dispatches to `VoxelToWorldIndex.concat`, which merges the pose
    labels and affine stacks in order exactly as
    [`xr.concat(..., dim="pose")`][xarray.concat] expects.

    Parameters
    ----------
    poses : sequence[xarray.DataArray]
        VoxelData arrays to stack, one per pose, in pose order. Each
        must have no existing `pose` dimension (see
        [ensure_voxeldata][confusius.validation.ensure_voxeldata]'s `allow_pose` parameter).
        Voxel dimensions, shape, voxel-space (`k`/`j`/`i`) coordinate values, and any
        non-core dimensions must otherwise agree across poses, exactly as required to
        merge non-concatenated variables in any `xr.concat` call.
    pose : sequence[hashable], optional
        Pose coordinate labels, one per entry of `poses`. If not provided, defaults to
        `0, 1, ..., len(poses) - 1`.

    Returns
    -------
    xarray.DataArray
        Stacked DataArray with a new `pose` dimension and pose-dependent
        voxel-to-world geometry (an `(npose, 4, 4)` affine stack, one affine per
        input). If a `time` dimension is present and every pose shares identical
        `time` values, `time` stays an ordinary 1D dimension coordinate. If per-pose
        `time` values differ (poses acquired sequentially rather than
        simultaneously), `time` instead becomes a genuine `(time, pose)`-shaped
        coordinate holding each pose's own real timestamp directly -- there is no
        single answer for "the" time of a `(pose, k, j, i)` voxel any more than
        there is a single answer for its `z`/`y`/`x` position, so `time` requires a
        scalar `pose` selection first, exactly like world coordinates already do.
        A 2D `time` is not itself an index (xarray dimension coordinates must be
        1D), so `.sel(time=...)` is unavailable until a pose is selected; after
        that, `.set_xindex("time")` promotes the resulting 1D `time` back into a
        real, selectable index.

    Raises
    ------
    ValueError
        If `poses` is empty, if `pose` does not have one label per entry of `poses`,
        if any pose already has a `pose` dimension, or if poses have mismatched
        `time` lengths.
    """
    if not poses:
        raise ValueError("poses must contain at least one DataArray.")
    if pose is not None and len(pose) != len(poses):
        raise ValueError(
            f"pose must have one label per entry of poses; got {len(pose)} labels "
            f"for {len(poses)} poses."
        )

    poses = [ensure_voxeldata(p, allow_pose=False) for p in poses]
    pose_labels = list(range(len(poses))) if pose is None else list(pose)
    has_time = TIME_DIM in poses[0].dims

    per_pose_time: np.ndarray | None = None
    time_attrs: dict[str, object] = {}
    if has_time:
        time_lengths = {da.sizes[TIME_DIM] for da in poses}
        if len(time_lengths) != 1:
            raise ValueError(
                f"All poses must have the same {TIME_DIM!r} length; got "
                f"{sorted(time_lengths)!r}."
            )
        time_values = np.stack(
            [np.asarray(da.coords[TIME_DIM].values) for da in poses], axis=1
        )  # (time, pose)
        if not np.all(time_values == time_values[:, [0]]):
            per_pose_time = time_values
            time_attrs = dict(poses[0].coords[TIME_DIM].attrs)

    promoted = []
    for da, label in zip(poses, pose_labels, strict=True):
        affine = get_voxel_to_world_affine(da)[np.newaxis]
        world_coord_attrs = {
            name: dict(da.coords[name].attrs)
            for name in get_voxel_to_world_coord_names(da)
        }
        expanded = da.expand_dims({POSE_DIM: [label]})
        if per_pose_time is not None:
            expanded = expanded.drop_vars(TIME_DIM)
        promoted.append(
            attach_voxel_to_world_index(
                expanded, affine, world_coord_attrs=world_coord_attrs
            )
        )

    stacked = xr.concat(promoted, dim=POSE_DIM, join="exact")
    # expand_dims prepends "pose" at axis 0; restore canonical core dim order
    # (...extra, time, pose, k, j, i).
    extra_dims = [d for d in stacked.dims if d not in CORE_DIMS]
    ordered_core = [d for d in CORE_DIMS if d in stacked.dims]
    stacked = stacked.transpose(*extra_dims, *ordered_core)
    if per_pose_time is not None:
        stacked = stacked.assign_coords(
            {TIME_DIM: ((TIME_DIM, POSE_DIM), per_pose_time, time_attrs)}
        )
    return stacked
