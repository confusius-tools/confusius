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
    get_voxel_to_world_affine,
    get_voxel_to_world_coord_names,
    get_voxel_to_world_spatial_dims,
)
from confusius._utils.stack import find_stack_level
from confusius.timing import convert_time_reference
from confusius.validation import ensure_fusi


def _reduce_world_coord_along_voxel_dim(
    coord: xr.DataArray, dim: str, other_voxel_dims: list[str]
) -> npt.NDArray[np.float64]:
    """Reduce a world coordinate to a 1D array of positions along one voxel dim.

    World coordinates are `(k, j, i)`-shaped (backed by a single joint
    `VoxelToWorldIndex` covering all voxel dims, see `confusius._utils.geometry`),
    but a world coordinate paired with `dim` only genuinely varies along `dim` for
    axis-aligned geometry; for oblique geometry this takes a representative slice at
    a fixed reference position in the other voxel dims.

    Parameters
    ----------
    coord : xarray.DataArray
        World coordinate to reduce.
    dim : str
        Voxel dimension to keep.
    other_voxel_dims : list[str]
        Voxel dimensions to reduce out, at index 0.

    Returns
    -------
    numpy.ndarray
        1D array of positions along `dim`.
    """
    if other_voxel_dims:
        coord = coord.isel(dict.fromkeys(other_voxel_dims, 0))
    return np.asarray(coord.values, dtype=np.float64)


def _build_consolidated_time_coordinate(
    time_coord: xr.DataArray,
    slice_time_values: npt.NDArray[np.floating],
    slice_time_attrs: dict[str, Any],
) -> xr.DataArray:
    """Build consolidated volume timings from per-slice timing metadata.

    Parameters
    ----------
    time_coord : xarray.DataArray
        Original volume-level time coordinate.
    slice_time_values : numpy.ndarray
        Consolidated per-slice timestamps with dims `(time, sweep_dim)`.
    slice_time_attrs : dict[str, Any]
        Attributes carried by the consolidated `slice_time` coordinate.

    Returns
    -------
    xarray.DataArray
        Replacement `time` coordinate for the consolidated volume. If per-slice timing
        metadata are insufficient to infer a whole-volume duration, the original
        `time_coord` is returned unchanged.

    Warns
    -----
    UserWarning
        If per-slice timing metadata do not include a usable
        `volume_acquisition_duration`. In that case, the original `time` coordinate is
        kept unchanged.
    UserWarning
        If inferred consolidated volume durations vary across time points. In that case,
        the returned coordinate omits `volume_acquisition_duration`.
    """
    slice_duration = slice_time_attrs.get("volume_acquisition_duration")
    if not isinstance(slice_duration, int | float) or slice_duration <= 0:
        warnings.warn(
            "Cannot infer consolidated volume timing from `slice_time` because "
            "`volume_acquisition_duration` is missing or non-positive. Keeping the "
            "original `time` coordinate.",
            stacklevel=find_stack_level(),
        )
        return time_coord

    slice_reference = slice_time_attrs.get(
        "volume_acquisition_reference",
        time_coord.attrs.get("volume_acquisition_reference", "start"),
    )
    volume_reference = time_coord.attrs.get(
        "volume_acquisition_reference", slice_reference
    )
    slice_onsets = convert_time_reference(
        slice_time_values,
        float(slice_duration),
        from_reference=slice_reference,
        to_reference="start",
    )
    volume_onsets = slice_onsets.min(axis=1)
    volume_durations = slice_onsets.max(axis=1) - volume_onsets + float(slice_duration)
    volume_times = convert_time_reference(
        volume_onsets,
        volume_durations,
        from_reference="start",
        to_reference=volume_reference,
    )

    time_attrs = dict(time_coord.attrs)
    time_attrs["volume_acquisition_reference"] = volume_reference
    if np.allclose(volume_durations, volume_durations[0], rtol=1e-5, atol=0):
        time_attrs["volume_acquisition_duration"] = float(volume_durations[0])
    else:
        time_attrs.pop("volume_acquisition_duration", None)
        warnings.warn(
            "Consolidated volume acquisition durations vary across time points. "
            "Omitting `time.attrs['volume_acquisition_duration']`.",
            stacklevel=find_stack_level(),
        )

    return xr.DataArray(volume_times, dims=["time"], attrs=time_attrs)


def _consolidate_linked_affines(
    affines: dict[str, Any],
    affines_key: str,
    main_per_pose: npt.NDArray[np.float64],
    main_consolidated: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Propagate per-pose affines through pose consolidation.

    The main affine (`affines[affines_key]`) is replaced by `main_consolidated`.
    Any other affine in `affines` that is shaped like the main per-pose stack is
    assumed to be a constant left-link of the main affine, i.e. there exists a
    constant `(4, 4)` matrix `L` such that `A[p] = L @ main_per_pose[p]` for all
    poses. The consolidated counterpart is then `L @ main_consolidated`. Affines
    that already have shape `(4, 4)` are passed through unchanged.

    Parameters
    ----------
    affines : dict[str, Any]
        Original `da.attrs["affines"]` mapping.
    affines_key : str
        Key of the affine driving the consolidation.
    main_per_pose : (npose, 4, 4) numpy.ndarray
        Per-pose stack of the main affine, prior to consolidation.
    main_consolidated : (4, 4) numpy.ndarray
        Consolidated form of the main affine.

    Returns
    -------
    dict[str, numpy.ndarray]
        Updated `affines` mapping with the main key replaced by
        `main_consolidated` and every other linked per-pose affine consolidated
        accordingly.

    Raises
    ------
    ValueError
        If a per-pose affine other than `affines_key` does not satisfy
        `A[p] = L @ main_per_pose[p]` for a constant `L` to within numerical
        tolerance.
    """
    new_affines: dict[str, npt.NDArray[np.float64]] = {affines_key: main_consolidated}
    main_inv0 = np.linalg.inv(main_per_pose[0])
    for key, value in affines.items():
        if key == affines_key:
            continue
        arr = np.asarray(value)
        if arr.shape == main_per_pose.shape:
            link = arr[0] @ main_inv0
            if not np.allclose(arr, link @ main_per_pose, rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"Affine {key!r} is not a constant left-link of "
                    f"{affines_key!r}; cannot consolidate."
                )
            new_affines[key] = link @ main_consolidated
        else:
            new_affines[key] = arr
    return new_affines


def consolidate_poses(
    da: xr.DataArray,
    affines_key: str = "world_to_lab",
    sweep_dim: str = "k",
    rtol: float = 0.01,
) -> xr.DataArray:
    """Merge `pose` and `sweep_dim` dimensions into a single axis ordered by position.

    Per-`(pose, sweep_dim)` world positions come from whichever source describes
    `da`'s per-pose geometry:

    - If `da`'s primary voxel-to-world geometry is itself pose-dependent (a
      `(npose, 4, 4)` affine stack — see
      [VoxelToWorldIndex.is_pose_dependent][confusius._utils.geometry.VoxelToWorldIndex.is_pose_dependent]),
      exact per-`(pose, sweep_dim)` world positions are read directly from `da`'s
      own world coordinates. This is the case for SCAN data, where lab space (a
      fixed scanner frame shared by every pose) is the canonical world frame — see
      [`load_scan`][confusius.io.load_scan].
    - Otherwise, positions are reconstructed from a separately stored `(npose, 4,
      4)` affine stack in `da.attrs["affines"][affines_key]` (other spatial dims
      are zero at voxel centres along the sweep):

      ```python
      pos[p, i] = affine[p, :3, sweep_col] * sweep_mm[i] + affine[p, :3, 3]
      ```

      where `sweep_col` is the column index of `sweep_dim` in the voxel dim
      ordering `(k, j, i)` → affine columns `(z, y, x)`. This covers, for example,
      a stack of NIfTI DataArrays concatenated along a new `pose` dimension with
      their `world_to_qform` affines stacked accordingly (no pose-dependent
      primary geometry of their own).

    The primary sweep direction is found via singular value decomposition of all
    positions. Each voxel is projected onto that axis, the positions are checked for
    regularity, then the data is reindexed in ascending order along the consolidated
    sweep axis.

    This function is primarily intended for consolidating multi-pose fUSI volumes
    acquired with an Iconeus system using a purely translational probe sweep. In that
    workflow, each pose corresponds to one probe position along the elevation axis
    (`k`/world `z`), and the DataArray is produced by
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
        DataArray with a `pose` dimension and per-pose geometry, either as
        pose-dependent primary voxel-to-world geometry or as a `(npose, 4, 4)`
        affine stack in `da.attrs["affines"][affines_key]`. Typically produced by
        [`load_scan`][confusius.io.load_scan] for `3Dscan` or `4Dscan` files.
    affines_key : str, default: "world_to_lab"
        Key into `da.attrs["affines"]` that holds the `(npose, 4, 4)` affine stack,
        used only when `da`'s primary geometry is not itself pose-dependent. Column
        order must be `(z, y, x, translation)`.
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
        DataArray with `pose` merged into `sweep_dim`, sorted by world position. The
        consolidated `sweep_dim` coordinate holds the projection of each voxel's
        world position onto the sweep axis, expressed in the same units as the input
        `sweep_dim` coordinate. For inputs that carry a `pose_time`
        coordinate, a consolidated `slice_time` with dims `("time", sweep_dim)` is
        included: each slice inherits the timestamp of the pose it came from.

    Raises
    ------
    ValueError
        If `da` has no `pose` dimension, if `sweep_dim` is not one of the spatial
        dimensions in `da.dims`, if `da`'s primary geometry is not pose-dependent and
        `affines_key` is missing from `da.attrs["affines"]`, if the rotation block of
        the affine is not constant across poses (non-translation sweep), or if the
        consolidated positions are not regularly spaced within `rtol`.

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

    primary_affine = get_voxel_to_world_affine(da)
    is_pose_dependent = primary_affine.ndim == 3

    if is_pose_dependent:
        # The primary geometry is itself a per-pose affine stack (e.g. SCAN data,
        # where lab space is the canonical world frame): read exact per-(pose,
        # sweep) world positions directly from da's own world coordinates rather
        # than reconstructing them from a separately stored affine. This needs no
        # assumption about a "local" pre-fold frame, unlike the affines_key path
        # below.
        affine = primary_affine  # (npose, 4, 4)
        n_sweep = da.sizes[sweep_dim]
        world_positions = []
        for world_dim in world_dims:
            coord = da.coords[world_dim]
            if other_voxel_dims:
                coord = coord.isel(dict.fromkeys(other_voxel_dims, 0))
            world_positions.append(np.asarray(coord.values, dtype=np.float64))
        # Each entry is (npose, n_sweep); stacked along a new last axis gives
        # (npose, n_sweep, 3), matching the affines_key path's lab_pos shape.
        lab_pos: npt.NDArray[np.float64] = np.stack(world_positions, axis=-1)
    else:
        if affines_key not in da.attrs.get("affines", {}):
            raise ValueError(
                f"DataArray has no pose-dependent primary geometry and no "
                f"{affines_key!r} entry in da.attrs['affines']; cannot determine "
                "per-pose positions."
            )
        affine = np.asarray(da.attrs["affines"][affines_key])  # (npose, 4, 4)
        sweep_mm = _reduce_world_coord_along_voxel_dim(
            da.coords[world_sweep_dim], sweep_dim, other_voxel_dims
        )
        n_sweep = len(sweep_mm)

        # Lab position for each (pose, sweep_dim): only the sweep column of the
        # affine contributes when we evaluate at the other spatial dims = 0 at
        # voxel centres. Rotation is constant across poses (checked below); only
        # the translation T changes.
        r_sweep: npt.NDArray[np.float64] = affine[
            :, :3, sweep_col
        ]  # (npose, 3), sweep_dim direction in lab.
        t: npt.NDArray[np.float64] = affine[
            :, :3, 3
        ]  # (npose, 3), per-pose translation in the affine/world-space units.
        lab_pos = (
            r_sweep[:, np.newaxis, :] * sweep_mm[np.newaxis, :, np.newaxis]
            + t[:, np.newaxis, :]
        )

    rotations: npt.NDArray[np.float64] = affine[:, :3, :3]  # (npose, 3, 3)
    # (npose, 3), per-pose translation, used below (t_perp) regardless of branch.
    t = affine[:, :3, 3]
    if not np.allclose(rotations, rotations[0], rtol=1e-6, atol=0):
        raise ValueError(
            f"Rotation block of affines[{affines_key!r}] is not constant across poses. "
            "consolidate_poses only supports pure translation sweeps."
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
        da.attrs.get("affines", {}), affines_key, affine, new_affine
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
            if values.size == 1:
                spacing = float(coord.attrs.get("voxdim", 1.0))
            else:
                spacing = float(np.median(np.diff(values)))
            spacings.append(spacing)
            world_attrs[world_dim] = {
                **coord.attrs,
                "voxdim": np.float64(spacing).item(),
            }
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
        coords: dict[str, Any] = {"time": da.coords["time"]}
        # Propagate per-slice timestamps if present: each consolidated slice inherits the
        # timestamp of the pose it came from. The consolidated slice_time keeps the same
        # acquisition reference metadata as the original pose_time coordinate.
        if "pose_time" in da.coords:
            slice_time_attrs = {
                **da.coords["pose_time"].attrs,
                "volume_acquisition_reference": da.coords["pose_time"].attrs.get(
                    "volume_acquisition_reference",
                    da.coords["time"].attrs.get(
                        "volume_acquisition_reference",
                        "start",
                    ),
                ),
            }
            if "volume_acquisition_duration" not in slice_time_attrs:
                slice_duration = da.coords["time"].attrs.get(
                    "volume_acquisition_duration"
                )
                if isinstance(slice_duration, int | float) and slice_duration > 0:
                    slice_time_attrs["volume_acquisition_duration"] = float(
                        slice_duration
                    )

            slice_time_values = np.asarray(da.coords["pose_time"].values)[:, pose_idx]
            coords["slice_time"] = xr.DataArray(
                slice_time_values,
                dims=["time", sweep_dim],
                attrs=slice_time_attrs,
            )
            coords["time"] = _build_consolidated_time_coordinate(
                da.coords["time"],
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
