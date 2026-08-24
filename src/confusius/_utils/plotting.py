"""Plotting helpers shared across the plotting/napari modules and registration views."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from confusius._dims import VOXEL_DIMS, WORLD_DIMS
from confusius._utils.geometry import (
    get_voxel_to_world_index_spacing,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_scalar_pose_affine,
)
from confusius.validation import ensure_voxeldata


def scale_min_max(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Linearly scale `arr` to [0, 1], handling flat arrays gracefully.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Float array with the same shape as `arr`, rescaled to `[0, 1]`. Returns an
        all-zero array when `arr` is flat (`arr.min() == arr.max()`). `-inf`/`inf` values
        (e.g. from [`db_scale`][confusius.xarray.scale.db_scale] on zero-valued voxels)
        are excluded when computing the scaling bounds and clipped to `0`/`1` in the
        output. `nan` elements are likewise excluded from the bounds but remain `nan`
        in the output, since there is no position on the `[0, 1]` scale to clip them to.

    Raises
    ------
    ValueError
        If `arr` contains no finite values.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot scale an array with no finite values.")
    lo, hi = finite.min(), finite.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def blend_red_cyan(
    fixed: NDArray[np.floating], moving: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Blend two 2D arrays as red (fixed) and cyan (moving) channels.

    Parameters
    ----------
    fixed : numpy.ndarray
        2D reference array, normalized to [0, 1].
    moving : numpy.ndarray
        2D moving array, normalized to [0, 1].

    Returns
    -------
    numpy.ndarray
        RGB image of shape `(*fixed.shape, 3)`.
    """
    h, w = fixed.shape
    rgb = np.zeros((h, w, 3))
    # Red channel: fixed only.
    rgb[..., 0] = fixed
    # Green + blue channels: cyan = moving.
    rgb[..., 1] = moving
    rgb[..., 2] = moving
    return rgb


def make_mosaic(
    fixed_vol: NDArray[np.floating], moving_vol: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Assemble a mosaic of per-slice red/cyan blends along the first axis.

    Parameters
    ----------
    fixed_vol : numpy.ndarray
        3D reference volume `(n_slices, H, W)`.
    moving_vol : numpy.ndarray
        3D moving volume `(n_slices, H, W)`.

    Returns
    -------
    numpy.ndarray
        RGB mosaic image.
    """
    n = fixed_vol.shape[0]
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    h, w = fixed_vol.shape[1], fixed_vol.shape[2]

    mosaic = np.zeros((n_rows * h, n_cols * w, 3))
    for i in range(n):
        r, c = divmod(i, n_cols)
        blend = blend_red_cyan(
            scale_min_max(fixed_vol[i]),
            scale_min_max(moving_vol[i]),
        )
        mosaic[r * h : (r + 1) * h, c * w : (c + 1) * w] = blend
    return mosaic


def qr_axis_spacing(
    linear: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Return each world row's dominant voxel axis and its spacing, via QR.

    Shared by `compute_oblique_axis_aligned_grid_geometry` and
    [`compute_slice_axis_aligned_grid_geometry`][confusius.plotting.image.compute_slice_axis_aligned_grid_geometry],
    both of which need the same nilearn `reorder_img`-style per-world-row
    spacing (see `compute_oblique_axis_aligned_grid_geometry`'s Notes for the
    rationale).

    Parameters
    ----------
    linear : (3, 3) numpy.ndarray
        Voxel-to-world affine's linear (non-translation) block.

    Returns
    -------
    row_to_axis : (3,) numpy.ndarray
        Index of the orthonormal QR basis vector each world row is most aligned
        to.
    axis_spacing : (3,) numpy.ndarray
        Scale of each orthonormal QR basis vector.
    """
    q_basis, r_scale = np.linalg.qr(linear)
    row_to_axis = np.abs(q_basis).argmax(axis=1)
    axis_spacing = np.abs(np.diag(r_scale))
    return row_to_axis, axis_spacing


def compute_oblique_axis_aligned_grid_geometry(
    data: xr.DataArray, world_dims: Sequence[str]
) -> tuple[list[int], list[float], list[float]]:
    """Compute the axis-aligned grid shape/spacing/origin `data` would resample onto.

    Pure geometry from `data`'s world-coordinate bounds and voxel spacing -- no
    interpolation. Callers that only need to know the resampled shape (e.g. to
    reject a panel that won't collapse to a 2D plane) can use this without paying
    for `resample_volume`'s actual interpolation.

    Parameters
    ----------
    data : xarray.DataArray
        Oblique (non-axis-aligned) VoxelData array. May have a spatial dim already
        squeezed to a scalar coordinate (e.g. a single-slice panel with `k` isel'd
        away) -- `ensure_voxeldata` restores it, affine and all, before computing
        geometry, so the result doesn't depend on whether the caller's dim was
        literal or scalar (see Notes).
    world_dims : collections.abc.Sequence of str
        World coordinate names (`"z"`/`"y"`/`"x"`) to compute grid geometry for, in
        order.

    Returns
    -------
    shape : list of int
        Number of voxels along each output world axis.
    spacing : list of float
        World distance between consecutive voxels along each output world axis.
    origin : list of float
        World location of output voxel position 0 along each output world axis.

    Raises
    ------
    ValueError
        If `data`'s geometry is pose-dependent, if any native voxel dimension has
        no well-defined (regular) spacing, or if the voxel-to-world affine is
        singular along a requested world axis.

    Notes
    -----
    Per-world-axis spacing comes from a QR decomposition of the full 3x3
    voxel-to-world linear block, following nilearn's `reorder_img`
    (`nilearn.image.resampling`): `Q, R = qr(linear)`, then each world row's
    spacing is `abs(diag(R))[argmax(abs(Q[row]))]` -- the scale of whichever
    orthonormal basis vector that row is most aligned to. This replaces a simpler,
    per-row "look at the raw affine row and pick the largest-magnitude voxel
    column" heuristic, which had a real bug: a spatial dim already squeezed to a
    scalar coordinate was silently dropped from consideration (its column removed
    from the linear block, per `get_voxel_to_world_affine`'s docstring, rather than
    the whole 3-column matrix always being used regardless of array shape), so a
    single-slice panel's predicted resample shape depended on whether the caller
    happened to squeeze it before or after handing it here. `ensure_voxeldata`
    below removes that dependency by always restoring the full, un-folded affine
    first, then QR decomposes it as a whole -- so the result is the same either
    way.
    """
    data = ensure_voxeldata(data)
    # QR only needs the affine, not the data's own per-voxel-dim coordinate
    # spacing -- but `resample_volume` (the eventual consumer of this geometry)
    # builds a regular-grid SimpleITK image from *all 3* native voxel dims
    # regardless of which one QR picks as dominant for a given world axis, so an
    # irregularly spaced dim anywhere still breaks it downstream with an opaque
    # error. Check all 3 up front for a clear message instead.
    for voxel_dim, voxel_spacing in get_voxel_to_world_index_spacing(data).items():
        if voxel_spacing is None:
            raise ValueError(
                f"Cannot resample onto an axis-aligned world grid: voxel "
                f"dimension {voxel_dim!r} has no well-defined spacing "
                "(irregularly spaced coordinates)."
            )
    affine = require_scalar_pose_affine(data, "Computing axis-aligned grid geometry")
    linear = affine[:3, :3]
    row_to_axis, axis_spacing = qr_axis_spacing(linear)

    spacing: list[float] = []
    origin: list[float] = []
    shape: list[int] = []
    for row, dim in enumerate(world_dims):
        values = np.asarray(data.coords[dim].values, dtype=np.float64)
        lower = np.float64(np.min(values)).item()
        upper = np.float64(np.max(values)).item()
        dim_spacing = float(axis_spacing[row_to_axis[row]])
        if dim_spacing == 0.0:
            raise ValueError(
                f"Cannot resample {dim!r} onto an axis-aligned world grid: the "
                "voxel-to-world affine is singular along this axis."
            )
        origin.append(lower)
        spacing.append(dim_spacing)
        # A relative tolerance absorbs floating-point noise in `upper`/`lower`
        # (e.g. from composing several affines) that would otherwise push an
        # exact multiple of `dim_spacing` just over an integer boundary and
        # `ceil` an extra, out-of-bounds slice onto the grid.
        n_steps = (upper - lower) / dim_spacing
        shape.append(np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1)
    return shape, spacing, origin


def resample_to_axis_aligned_world_grid(
    data: xr.DataArray,
    *,
    reference: xr.DataArray | None = None,
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float | None = None,
) -> xr.DataArray:
    """Resample voxel-to-world data onto an axis-aligned world grid for display.

    Parameters
    ----------
    data : xarray.DataArray
        Three-dimensional or three-dimensional-plus-time DataArray. VoxelData
        arrays are resampled using their voxel-to-world index.
    reference : xarray.DataArray, optional
        Axis-aligned world-grid DataArray to reuse as the resampling target.
        If not provided, a new plotting grid is synthesized from `data`'s world
        bounds and per-axis world spacing.
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling. Use `"nearest"` for label/mask
        data, where blending distinct integer labels together is never meaningful.
    fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after resampling.
        If not provided, defaults to `float(data.min())` (see
        [`resample_volume`][confusius.registration.resample_volume]).

    Returns
    -------
    xarray.DataArray
        Axis-aligned world-grid VoxelData array when `data` has
        voxel-to-world geometry; otherwise the original input.
    """
    if not has_voxel_to_world_index(data) or has_axis_aligned_voxel_to_world_index(
        data
    ):
        return data

    from confusius.registration import resample_like, resample_volume

    world_dims = WORLD_DIMS
    if reference is not None:
        if not has_voxel_to_world_index(reference):
            from confusius.xarray import create_voxeldata

            # `reference` is a plain, caller-supplied DataArray here (not one of
            # ConfUSIus's own VoxelData arrays), so its spatial dims may be named
            # z/y/x rather than the native voxel names create_voxeldata now
            # requires; remap them, leaving any other dim name (time, pose, extras)
            # unchanged.
            spatial_to_voxel = dict(zip(WORLD_DIMS, VOXEL_DIMS, strict=True))
            reference_dims = tuple(
                spatial_to_voxel.get(str(dim), str(dim)) for dim in reference.dims
            )
            spacing = []
            origin = []
            for dim in ("z", "y", "x"):
                if dim in reference.coords:
                    values = np.asarray(reference.coords[dim].values, dtype=np.float64)
                    origin.append(float(values[0]))
                    spacing.append(
                        float(np.median(np.diff(values))) if values.size > 1 else 1.0
                    )
                else:
                    origin.append(0.0)
                    spacing.append(1.0)
            reference = create_voxeldata(
                reference.data,
                dims=reference_dims,
                time=reference.coords.get("time"),
                pose=reference.coords.get("pose"),
                spacing=tuple(spacing),
                origin=tuple(origin),
                attrs=reference.attrs,
                name=str(reference.name) if reference.name is not None else None,
            )
        result = resample_like(
            data,
            reference,
            np.eye(len(world_dims) + 1, dtype=np.float64),
            interpolation=interpolation,
            fill_value=fill_value,
        )
    else:
        # `data` is oblique here (axis-aligned data already returned above), so
        # `data.coords[dim]` for a world dim is (k, j, i)-shaped, not 1D -- a
        # representative per-axis spacing therefore can't come from
        # `np.diff(values)` (which would default to differencing along the last
        # voxel axis, regardless of which world axis `dim` actually corresponds
        # to). require_scalar_pose_affine is still called for its validation side
        # effect: rejecting pose-stacked data before attempting to resample it onto
        # one axis-aligned grid.
        require_scalar_pose_affine(
            data, "Resampling to an axis-aligned world grid for display"
        )

        # A voxel dim's own spacing is only a valid proxy for a world axis's
        # resolution when the affine is close to diagonal. For a rotated/permuted
        # affine (see design/multipose-voxel-to-world-index.md's "monomial
        # affines" deferred item -- e.g. a probe swept along what is nominally the
        # `k` voxel axis but physically lands mostly along world `x`), the world
        # axis that needs fine spacing is fed by whichever voxel dim actually
        # dominates that row of the affine, not the positionally-matching one.
        # Pick that dominant voxel dim's spacing for each output world axis.
        shape, spacing, origin = compute_oblique_axis_aligned_grid_geometry(
            data, world_dims
        )

        result = resample_volume(
            data,
            np.eye(len(world_dims) + 1, dtype=np.float64),
            output_sizes=dict(zip(VOXEL_DIMS, shape, strict=True)),
            output_spacing=dict(zip(VOXEL_DIMS, spacing, strict=True)),
            output_origin=dict(zip(WORLD_DIMS, origin, strict=True)),
            output_direction=np.eye(len(world_dims), dtype=np.float64),
            interpolation=interpolation,
            fill_value=fill_value,
        )

    return result
