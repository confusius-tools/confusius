"""Image visualization utilities for fUSI data."""

import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import POSE_DIM, VOXEL_DIMS, WORLD_DIMS
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.geometry import (
    get_voxel_to_world_index_spacing,
    has_axis_aligned_voxel_to_world_index,
    has_voxel_to_world_index,
    require_scalar_pose_affine,
)
from confusius._utils.mask import select_masked_features
from confusius._utils.plotting import (
    blend_red_cyan,
    compute_oblique_axis_aligned_grid_geometry,
    qr_axis_spacing,
    resample_to_axis_aligned_world_grid,
    scale_min_max,
)
from confusius._utils.stack import find_stack_level
from confusius.plotting._hover import (
    _HoverManager,
    _normalize_roi_labels,
)
from confusius.plotting._utils import (
    _auto_fg_color,
    _get_distinct_colors,
    _resolve_font_sizes,
    _style_colorbar,
    coerce_complex_to_magnitude,
    materialize_axis_aligned_world_grid_for_display,
    sort_coords_for_plot,
)
from confusius.signal import clean
from confusius.validation import (
    ensure_voxeldata,
    validate_matching_coordinates,
    validate_time_series,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure, SubFigure

_BASE_SIZE = 4.0
"""Base subplot size for VolumePlotter when creating new figures.

Actual figure size is computed as `(subplot_size * ncols + 1 inch for colorbar,
subplot_size * nrows)` and then constrained to a maximum size.
"""


def _compute_grid_dims(
    n_slices: int, nrows: int | None, ncols: int | None
) -> tuple[int, int]:
    """Compute grid dimensions for a grid of `n_slices` panels."""
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n_slices)))
        nrows = int(np.ceil(n_slices / ncols))
    elif ncols is None:
        assert nrows is not None
        ncols = int(np.ceil(n_slices / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n_slices / ncols))
    return nrows, ncols


def _resolve_default_slice_mode(
    data: xr.DataArray,
    slice_mode: str | None,
) -> str:
    """Resolve the default slice axis for `plot_volume`/`plot_composite`.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to inspect.
    slice_mode : str, optional
        User-provided slice mode. If provided, returned unchanged.

    Returns
    -------
    str
        User-provided `slice_mode`, the only constant world dimension for planar
        data, or `"z"` otherwise.
    """
    if slice_mode is not None:
        return slice_mode

    constant_world_dims = []
    for dim in WORLD_DIMS:
        values = np.asarray(data.coords[dim].values, dtype=float)
        if values.size and np.isclose(np.nanmin(values), np.nanmax(values)):
            constant_world_dims.append(dim)

    return constant_world_dims[0] if len(constant_world_dims) == 1 else "z"


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert 1D coordinate centers to cell edge positions for `pcolormesh`.

    Handles non-uniform spacing by using midpoints between adjacent centers as interior
    edges, and extrapolating half a step at each end.
    """
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    interior = (centers[:-1] + centers[1:]) / 2
    left = centers[0] - (centers[1] - centers[0]) / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], interior, [right]])


def _validate_slice_mode(data: xr.DataArray, slice_mode: str) -> None:
    """Validate slice selection semantics for plotting.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array being sliced for plotting.
    slice_mode : str
        Requested slice dimension.

    Raises
    ------
    ValueError
        If `data` is sliced along an unsupported dimension.
    """
    valid_slice_modes = (
        tuple(str(dim) for dim in data.dims if str(dim) not in VOXEL_DIMS) + WORLD_DIMS
    )
    if slice_mode not in valid_slice_modes:
        raise ValueError(
            f"Unsupported slice_mode={slice_mode!r} for plotting. "
            f"Supported modes: {valid_slice_modes!r}."
        )


@dataclass(frozen=True)
class SliceAxisGrid:
    """One world axis's regular discretization, shared across volumes on a plotter.

    `slice_world_dim`'s own axis needs a shared spacing/origin/count across every
    volume drawn on the same plotter for `match_coordinates` to line up panels by
    physical position -- the two in-plane axes don't (see
    `compute_shared_slice_axis_grid_geometry`), so this only ever describes the
    one sliced axis, never a full grid.

    Attributes
    ----------
    world_dim : str
        World coordinate name (`"z"`/`"y"`/`"x"`) this spec describes.
    spacing : float
        World distance between consecutive slices.
    origin : float
        World position of the first slice.
    count : int
        Number of slices.
    """

    world_dim: str
    spacing: float
    origin: float
    count: int


def compute_shared_slice_axis_grid_geometry(
    data: xr.DataArray,
    slice_world_dim: str,
    *,
    slice_axis_grid: SliceAxisGrid | None = None,
) -> tuple[
    Mapping[Hashable, int],
    Mapping[Hashable, float],
    Mapping[Hashable, float],
    SliceAxisGrid,
]:
    """Compute a `resample_volume` target grid axis-aligned to the world frame.

    Forces all 3 output axes onto the global (identity) direction, so display
    cells are always rectangular -- callers pass `numpy.eye(3)` directly as
    `resample_volume`'s `output_direction`, since this never derives anything
    else. `slice_world_dim` is additionally forced to a regular spacing shared
    across calls (`slice_axis_grid`), required so `match_coordinates` can line up
    panels of the same physical slice across independently-resampled volumes.
    The other two output axes keep each volume's own native per-axis spacing
    (derived via `compute_oblique_axis_aligned_grid_geometry`'s QR-based
    heuristic, see `qr_axis_spacing`), so a lower-resolution and a
    higher-resolution volume overlaid on the same plotter each keep their own
    native in-plane resolution rather than being resampled to match one another.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, oblique or axis-aligned.
    slice_world_dim : str
        World coordinate name (`"z"`/`"y"`/`"x"`) to align.
    slice_axis_grid : SliceAxisGrid, optional
        Shared spacing/origin/count for `slice_world_dim`, established by an
        earlier call on the same plotter. If not provided, derived from `data`'s
        own extent along `slice_world_dim` and returned for the caller to reuse
        on subsequent volumes.

    Returns
    -------
    output_sizes : dict[str, int]
        Number of voxels along each output axis, keyed by `k`/`j`/`i`.
    output_spacing : dict[str, float]
        World distance between output positions, keyed by `k`/`j`/`i`.
    output_origin : dict[str, float]
        World location of output position `(0, 0, 0)`, keyed by `z`/`y`/`x`.
    slice_axis_grid : SliceAxisGrid
        `slice_axis_grid` unchanged if provided, otherwise the one just
        established from `data` -- pass this to the next call on the same
        plotter so every volume's slice axis lines up.

    Raises
    ------
    ValueError
        If `data` does not carry voxel-to-world geometry, or if its geometry is
        pose-dependent.
    """
    affine = require_scalar_pose_affine(
        data, "Slice-axis-aligned resampling for display"
    )
    world_dims = WORLD_DIMS
    slice_row = world_dims.index(slice_world_dim)
    linear = affine[:3, :3]

    # Same per-world-axis spacing heuristic as
    # compute_oblique_axis_aligned_grid_geometry's Notes (nilearn's reorder_img).
    row_to_axis, qr_spacing = qr_axis_spacing(linear)

    if slice_axis_grid is not None:
        slice_spacing = slice_axis_grid.spacing
        slice_origin = slice_axis_grid.origin
        slice_count = slice_axis_grid.count
    else:
        slice_values = np.asarray(
            data.coords[slice_world_dim].values, dtype=np.float64
        ).ravel()
        slice_spacing = float(qr_spacing[row_to_axis[slice_row]])
        slice_origin = float(slice_values.min())
        n_steps = (float(slice_values.max()) - slice_origin) / slice_spacing
        slice_count = np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1
        slice_axis_grid = SliceAxisGrid(
            world_dim=slice_world_dim,
            spacing=slice_spacing,
            origin=slice_origin,
            count=slice_count,
        )

    # `output_direction` is always the identity (`resample_volume` call site), so
    # output voxel dim `k`/`j`/`i` maps positionally to world `z`/`y`/`x` --
    # `materialize_axis_aligned_world_grid_for_display` relies on exactly this
    # correspondence. Build each output array in that same world-axis order,
    # regardless of which row `slice_world_dim` happens to be.
    sizes: list[int] = [0, 0, 0]
    spacings: list[float] = [0.0, 0.0, 0.0]
    origins: list[float] = [0.0, 0.0, 0.0]
    sizes[slice_row] = slice_count
    spacings[slice_row] = slice_spacing
    origins[slice_row] = slice_origin
    for row in range(3):
        if row == slice_row:
            continue
        values = np.asarray(data.coords[world_dims[row]].values, dtype=np.float64)
        dim_spacing = float(qr_spacing[row_to_axis[row]])
        row_min = float(values.min())
        row_max = float(values.max())
        n_steps = (row_max - row_min) / dim_spacing
        sizes[row] = np.int64(np.ceil(n_steps - 1e-6 * max(1.0, n_steps))).item() + 1
        spacings[row] = dim_spacing
        origins[row] = row_min

    return (
        dict(zip(VOXEL_DIMS, sizes, strict=True)),
        dict(zip(VOXEL_DIMS, spacings, strict=True)),
        dict(zip(WORLD_DIMS, origins, strict=True)),
        slice_axis_grid,
    )


def _resample_to_shared_slice_axis_grid(
    data: xr.DataArray,
    slice_world_dim: str,
    *,
    slice_axis_grid: SliceAxisGrid | None = None,
    interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    fill_value: float | None = None,
) -> tuple[xr.DataArray, SliceAxisGrid | None]:
    """Resample `data` so all 3 spatial axes are world-axis-aligned, for display.

    See `compute_shared_slice_axis_grid_geometry` for the geometry this builds
    on `resample_volume` with. The two in-plane axes keep `data`'s own native
    per-axis spacing (see `SliceAxisGrid`); `slice_world_dim` is additionally
    forced onto a shared, regular discretization so panels from different
    volumes/masks line up by physical position.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, oblique or axis-aligned.
    slice_world_dim : str
        World coordinate name (`"z"`/`"y"`/`"x"`) to align.
    slice_axis_grid : SliceAxisGrid, optional
        Shared spacing/origin/count for `slice_world_dim`, established by an
        earlier call on the same plotter. If not provided, derived from `data`'s
        own extent and returned for the caller to reuse on subsequent volumes.
    interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used during resampling. Use `"nearest"` for
        label/mask data, where blending distinct integer labels together is
        never meaningful.
    fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after
        resampling. If not provided, defaults to `float(data.min())` (see
        [`resample_volume`][confusius.registration.resample_volume]).

    Returns
    -------
    result : xarray.DataArray
        `data` resampled so all 3 spatial axes are axis-aligned and (if
        `slice_axis_grid` was provided) `slice_world_dim` matches its shared
        spacing/origin/count. `data` unchanged if it's already fully
        axis-aligned (nothing to fix, and matplotlib only needs matching world
        coordinates to overlay correctly, not a matching pixel grid).
    slice_axis_grid : SliceAxisGrid
        `slice_axis_grid` unchanged if provided, or the one just established
        from `data`'s own geometry otherwise -- even when `data` was already
        axis-aligned and skipped resampling, so a later oblique volume/mask on
        the same plotter still has a grid to align its own slice axis to.
    """
    if has_axis_aligned_voxel_to_world_index(data):
        if slice_axis_grid is not None:
            return data, slice_axis_grid

        _, _, _, established_grid = compute_shared_slice_axis_grid_geometry(
            data, slice_world_dim
        )
        return data, established_grid

    from confusius.registration import resample_volume

    sizes, spacing, origin, established_grid = compute_shared_slice_axis_grid_geometry(
        data,
        slice_world_dim,
        slice_axis_grid=slice_axis_grid,
    )
    result = resample_volume(
        data,
        np.eye(4, dtype=np.float64),
        output_sizes=sizes,
        output_spacing=spacing,
        output_origin=origin,
        output_direction=np.eye(3, dtype=np.float64),
        interpolation=interpolation,
        fill_value=fill_value,
    )
    return result, established_grid


def _resample_to_planar_world_grid(
    data: xr.DataArray,
    *,
    slice_mode: str,
    interpolation: Literal["linear", "nearest", "bspline"],
    fill_value: float | None,
) -> xr.DataArray:
    """Resample `data` onto an axis-aligned world grid, requiring a flat 2D result.

    Unlike `_resample_to_shared_slice_axis_grid`, no axis is forced onto a
    shared cross-volume discretization (`SliceAxisGrid`) -- used wherever nothing
    needs to line up across volumes: a non-pose extra `slice_mode` (its whole
    array, in `_prepare_slice_inputs`) or a `pose`-faceted panel (per-panel, in
    `_resample_pose_slices_to_world_grid`, after `.isel` has collapsed a
    pose-dependent affine to scalar). Since nothing forces one axis flat the way
    a spatial `slice_mode` does, a cheap collapse-to-a-2D-plane check
    (`compute_oblique_axis_aligned_grid_geometry`, bounds/spacing arithmetic, no
    interpolation) runs first for oblique input, raising `ValueError` up front if
    `data`'s geometry is oblique to the world axes and would not lie flat on any
    world plane -- display has no meaningful 2D result for a genuinely
    3D-extended (non-planar) input either, so this avoids paying for an
    interpolation that can't produce a usable result. Does not materialize the
    result onto plain `z`/`y`/`x` dims -- callers materialize themselves.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array, oblique or axis-aligned.
    slice_mode : str
        The caller's `slice_mode`, named in the collapse-check error message.
    interpolation : {"linear", "nearest", "bspline"}
        Interpolation method used during resampling.
    fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after
        resampling. If not provided, defaults to `float(data.min())` (see
        [`resample_volume`][confusius.registration.resample_volume]).

    Returns
    -------
    xarray.DataArray
        `data` resampled onto an axis-aligned world grid, or unchanged if it
        carries no voxel-to-world geometry.

    Raises
    ------
    ValueError
        If `data`'s spatial geometry is oblique to the world axes and would not
        collapse to a 2D plane.
    """
    if not has_axis_aligned_voxel_to_world_index(data):
        shape, _, _ = compute_oblique_axis_aligned_grid_geometry(data, WORLD_DIMS)
        if sum(size > 1 for size in shape) != 2:
            raise ValueError(
                f"Displaying slice_mode={slice_mode!r}'s data in world space would "
                "not collapse to a 2D plane (predicted shape "
                f"{dict(zip(WORLD_DIMS, shape, strict=True))}). This happens when "
                "the data's spatial geometry is oblique to the world axes and does "
                "not lie flat on any world plane."
            )
    return resample_to_axis_aligned_world_grid(
        data,
        reference=None,
        interpolation=interpolation,
        fill_value=fill_value,
    )


def _slice_edges_and_centers(
    slice_da: xr.DataArray,
    dim_row: str,
    dim_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 1D edge/center coordinates for the plotted row/column dimensions.

    `slice_da` is always already materialized onto plain rectilinear dims by
    this point (`_prepare_slice_inputs`/`_resample_pose_slices_to_world_grid` force
    every panel with voxel-to-world geometry onto the axis-aligned world grid
    before display), so a plain per-axis coordinate lookup is always correct.
    """
    if dim_col in slice_da.coords:
        x_centers = slice_da.coords[dim_col].values.astype(float)
        x_edges = _centers_to_edges(x_centers)
    else:
        x_centers = np.arange(slice_da.sizes[dim_col], dtype=float)
        x_edges = np.arange(slice_da.sizes[dim_col] + 1, dtype=float)

    if dim_row in slice_da.coords:
        y_centers = slice_da.coords[dim_row].values.astype(float)
        y_edges = _centers_to_edges(y_centers)
    else:
        y_centers = np.arange(slice_da.sizes[dim_row], dtype=float)
        y_edges = np.arange(slice_da.sizes[dim_row] + 1, dtype=float)

    return x_edges, y_edges, x_centers, y_centers


def _resolve_norm(
    slices: list,
    norm: "Normalize | None",
    data_attrs_norm: "Normalize | None",
    vmin: float | None,
    vmax: float | None,
) -> "Normalize":
    """Determine the colormap normalization.

    Precedence:
    - If `norm` is passed explicitly, it wins and vmin/vmax are ignored.
    - Otherwise, vmin/vmax (if given) override whatever is in `data_attrs`.
    - Otherwise, fall back to the norm stored in `data_attrs["norm"]`, or
      compute percentile-based limits from the data.
    """
    user_set_norm = norm is not None
    data_has_norm = data_attrs_norm is not None
    user_set_vmin = vmin is not None
    user_set_vmax = vmax is not None

    resolved_norm = norm if user_set_norm else data_attrs_norm

    if not user_set_norm:
        if data_has_norm:
            assert resolved_norm is not None
            default_vmin = resolved_norm.vmin
            default_vmax = resolved_norm.vmax
        else:
            all_vals = np.concatenate([s.values.ravel().astype(float) for s in slices])
            all_vals = all_vals[np.isfinite(all_vals)]
            default_vmin = (
                float(np.percentile(all_vals, 2)) if len(all_vals) > 0 else 0.0
            )
            default_vmax = (
                float(np.percentile(all_vals, 98)) if len(all_vals) > 0 else 1.0
            )

        vmin = vmin if user_set_vmin else default_vmin
        vmax = vmax if user_set_vmax else default_vmax

        if (not data_has_norm) or user_set_vmin or user_set_vmax:
            from matplotlib.colors import Normalize

            resolved_norm = Normalize(vmin=vmin, vmax=vmax)

    assert resolved_norm is not None

    return resolved_norm


_STAT_MAP_DIVERGING_CMAP = "coolwarm"
"""Default colormap for diverging statistical maps (see `plot_stat_map`)."""

_STAT_MAP_SEQUENTIAL_CMAP = "viridis"
"""Default colormap for non-negative statistical maps (see `plot_stat_map`)."""

_STAT_MAP_SEQUENTIAL_CMAP_NEGATIVE = "viridis_r"
"""Default colormap for non-positive statistical maps (see `plot_stat_map`).

Reversed relative to `_STAT_MAP_SEQUENTIAL_CMAP` so that, in both the
non-negative and non-positive cases, values near zero map to the same end of
the colormap (dark purple) and the most extreme magnitude maps to the other
end (yellow).
"""


def _resolve_stat_map_style(
    data: xr.DataArray,
    vmin: float | None,
    vmax: float | None,
    cmap: "str | Colormap | None",
    auto_range: bool,
) -> tuple[float, float, "str | Colormap"]:
    """Resolve `(vmin, vmax, cmap)` for a statistical map.

    `vmin`/`vmax` fall back to the actual min/max of `data` when not provided.

    When `auto_range` is `True` (default), the sign of `data` determines the
    layout:

    - Both positive and negative values: diverging, symmetric `[-m, m]` range
      where `m = max(|vmin|, |vmax|)` over the bounds actually provided, falling
      back to the largest magnitude in `data` when neither is given, with `cmap`
      defaulting to `_STAT_MAP_DIVERGING_CMAP`.
    - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
      defaulting to `_STAT_MAP_SEQUENTIAL_CMAP`.
    - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
      defaulting to `_STAT_MAP_SEQUENTIAL_CMAP_NEGATIVE`.

    When `auto_range` is `False`, the resolved `vmin`/`vmax` are used directly with
    no zero-anchoring, and `cmap` defaults to `_STAT_MAP_DIVERGING_CMAP` regardless
    of `data`'s sign. In both cases, an explicitly provided `cmap` is always used
    as-is.
    """
    values = data.values.ravel().astype(float)
    values = values[np.isfinite(values)]
    data_min = float(values.min()) if len(values) > 0 else 0.0
    data_max = float(values.max()) if len(values) > 0 else 1.0

    resolved_vmin = vmin if vmin is not None else data_min
    resolved_vmax = vmax if vmax is not None else data_max

    if not auto_range:
        return (
            resolved_vmin,
            resolved_vmax,
            cmap if cmap is not None else _STAT_MAP_DIVERGING_CMAP,
        )

    if data_min < 0 < data_max:
        # A bound given on its own caps the symmetric range by itself: falling back
        # to the data's own min/max for the missing one would drop it silently.
        explicit = [abs(bound) for bound in (vmin, vmax) if bound is not None]
        abs_max = max(explicit) if explicit else max(abs(data_min), abs(data_max))
        return -abs_max, abs_max, cmap if cmap is not None else _STAT_MAP_DIVERGING_CMAP

    if data_max > 0:
        return (
            0.0,
            resolved_vmax,
            cmap if cmap is not None else _STAT_MAP_SEQUENTIAL_CMAP,
        )

    return (
        resolved_vmin,
        0.0,
        cmap if cmap is not None else _STAT_MAP_SEQUENTIAL_CMAP_NEGATIVE,
    )


def _threshold_slices(
    slices: list[xr.DataArray],
    threshold: float | None,
    threshold_mode: Literal["lower", "upper"],
) -> list[xr.DataArray | np.ndarray]:
    """Apply thresholding to a list of slices, returning masked arrays."""
    if threshold is None:
        return [s.values for s in slices]

    thresholded = []
    for s in slices:
        if threshold_mode == "lower":
            mask = np.abs(s) >= threshold
        else:
            mask = np.abs(s) <= threshold
        thresholded.append(s.where(mask))
    return thresholded


def _resolve_cmap(
    cmap: "str | Colormap | None",
    data_attrs_cmap: "str | Colormap | None",
    norm: "Normalize",
    threshold: float | None,
    threshold_mode: Literal["lower", "upper"],
) -> "Colormap":
    """Build colormap with gray band indicating thresholded regions.

    For `threshold_mode='lower'`: gray between `[-threshold, threshold]`.
    For `threshold_mode='upper'`: gray outside `[-threshold, threshold]`.

    Raises
    ------
    ValueError
        If `norm.vmin` or `norm.vmax` is not finite.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    if (
        norm.vmin is None
        or norm.vmax is None
        or not math.isfinite(norm.vmin)
        or not math.isfinite(norm.vmax)
    ):
        raise ValueError(
            "norm.vmin and norm.vmax must be finite, got "
            f"vmin={norm.vmin!r}, vmax={norm.vmax!r}."
        )

    cmap = (
        cmap
        if cmap is not None
        else data_attrs_cmap
        if data_attrs_cmap is not None
        else "gray"
    )
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    cmap_colors = [(i / (cmap.N - 1), cmap(i / (cmap.N - 1))) for i in range(cmap.N)]

    threshold = 0.0 if threshold is None else abs(threshold)
    gray_low = norm(-threshold)
    gray_high = norm(threshold)
    if threshold_mode == "lower":
        colors_before = [c for c in cmap_colors if c[0] <= gray_low]
        colors_after = [c for c in cmap_colors if c[0] >= gray_high]

        if gray_low < gray_high:
            # We clip the gray low/high points to [0, 1] because Matplotlib expects
            # colormap values in that range.
            gray_band = [
                (max(0, gray_low), "gray"),
                (min(1.0, gray_high), "gray"),
            ]
        else:
            gray_band = []
        new_colors = colors_before + gray_band + colors_after
    else:
        colors_middle = [c for c in cmap_colors if gray_low <= c[0] <= gray_high]

        if gray_low > 0.0:
            gray_band_low = [(0.0, "gray"), (gray_low, "gray")]
        else:
            gray_band_low = []

        if gray_high < 1.0:
            gray_band_high = [(gray_high, "gray"), (1.0, "gray")]
        else:
            gray_band_high = []
        new_colors = gray_band_low + colors_middle + gray_band_high

    # Matplotlib 3.11+ requires (value, color) pairs passed to `from_list` to be
    # strictly monotonically increasing in value. Our boundary points can collide
    # with neighboring cmap entries (e.g. when `gray_low == 0` the first
    # `colors_before` entry shares its value with the start of `gray_band`).
    # Collapse duplicates by value, keeping the later entry so gray-band
    # boundaries take precedence over the underlying cmap at the same value.
    deduped: dict[float, str | tuple[float, ...] | list[float]] = {}
    for value, color in new_colors:
        deduped[value] = color
    new_colors = list(deduped.items())

    # Preserve the source colormap's resolution. The default N=256 of
    # `LinearSegmentedColormap.from_list` collapses larger discrete cmaps such as
    # the atlas `ListedColormap` (N == number of regions, often >256), aliasing
    # high indices to wrong (or out-of-range) colours. Propagate under/over/bad
    # colours so the atlas's transparent under-colour for label 0 (background)
    # survives the rebuild. Cast to tuple because the getters return numpy arrays
    # but the kwargs expect a color-like. `from_list` accepts these kwargs since
    # matplotlib 3.11, no need for a separate `with_extremes` round-trip.
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_thresholded",
        new_colors,
        N=cmap.N,
        under=tuple(cmap.get_under()),
        over=tuple(cmap.get_over()),
        bad=tuple(cmap.get_bad()),
    )


def _build_axis_label(da: xr.DataArray, dim: str) -> str:
    """Return axis label for `dim`, including units when available.

    Parameters
    ----------
    da : xarray.DataArray
        The panel being labeled.
    dim : str
        The array dimension displayed on this axis.
    """
    # `isel`ing away a dim the custom VoxelToWorldIndex is attached to can drop the
    # index outright (an xarray custom-index limitation, most visible for
    # pose-stacked oblique geometry) even though the geometry is fully recoverable
    # -- restore it first so plottability reflects the panel's true geometry, not
    # an isel artifact. Not always possible: an already-materialized world-space
    # panel (z/y/x dims, k/j/i intentionally dropped by
    # `_materialize_axis_aligned_world_grid_for_display`) genuinely has no voxel
    # dim left to restore, so fall back to `da` as-is in that case.
    try:
        da = ensure_voxeldata(da)
    except (TypeError, ValueError):
        pass
    if dim != POSE_DIM and has_voxel_to_world_index(da):
        return f"{dim} in-plane (mm)"

    label = dim
    if dim in da.coords:
        units = da.coords[dim].attrs.get("units")
        if units:
            label = f"{dim} ({units})"
    return label


def _format_coord(coord: Hashable) -> str:
    """Format a slice coordinate for display: `.3g` for numeric, `str` otherwise."""
    if isinstance(coord, numbers.Real):
        return f"{coord:.3g}"
    return str(coord)


def _coords_match(
    stored_coord: Hashable, target_coord: Hashable, tolerance: float
) -> bool:
    """Compare two slice coordinates, using a tolerance for numeric values.

    Parameters
    ----------
    stored_coord : collections.abc.Hashable
        Coordinate value already recorded for an existing axis.
    target_coord : collections.abc.Hashable
        Coordinate value being matched against `stored_coord`.
    tolerance : float
        Maximum allowed absolute difference for numeric coordinates. Pass `0.0`
        to require an exact match (e.g. for a non-spatial `slice_mode`, where
        nearness has no physical meaning even for a numeric coordinate like a
        `pose` id).

    Returns
    -------
    bool
        Whether `stored_coord` and `target_coord` are considered the same slice.
    """
    if isinstance(stored_coord, numbers.Real) and isinstance(
        target_coord, numbers.Real
    ):
        return abs(float(stored_coord) - float(target_coord)) <= tolerance
    return stored_coord == target_coord


def _default_slice_coords(data: xr.DataArray, slice_mode: str) -> list[Hashable]:
    """Return every coordinate value along `slice_mode`, for the default (all-panels) case.

    Parameters
    ----------
    data : xarray.DataArray
        Data already passed through `_prepare_slice_inputs`.
    slice_mode : str
        The dimension/coordinate to enumerate values for.

    Returns
    -------
    list of collections.abc.Hashable
        One entry per distinct value along `slice_mode`. Positional indices
        (`range(data.sizes[slice_mode])`) when `slice_mode` has no coordinate at
        all.
    """
    if slice_mode not in data.coords:
        return list(range(data.sizes[slice_mode]))
    return list(data.coords[slice_mode].values)


def _extract_slices(
    data: xr.DataArray,
    slice_mode: str,
    slice_coords: Sequence[Hashable],
    *,
    slice_spacing: float | None = None,
) -> tuple[list[xr.DataArray], list[Hashable]]:
    """Extract 2D slices from `data` along `slice_mode`.

    A spatial `slice_mode` (`"z"`/`"y"`/`"x"`) is matched by nearest-neighbour
    lookup within `slice_spacing / 2` of the requested coordinate -- physical
    proximity is meaningful there. Any other `slice_mode` (e.g. `"region"`, a
    numeric `"pose"` id) always requires an exact match: nearness has no
    physical meaning for a categorical/discrete facet, so a request for pose 2
    must never silently return pose 1's data mislabeled as pose 2.

    Parameters
    ----------
    data : xarray.DataArray
        Data already passed through `_prepare_slice_inputs`.
    slice_mode : str
        The dimension/coordinate to extract slices along.
    slice_coords : collections.abc.Sequence of collections.abc.Hashable
        Coordinate values to extract.
    slice_spacing : float, optional
        `data`'s own world spacing along `slice_mode`, used as the
        nearest-neighbour tolerance for a spatial `slice_mode`. Ignored (no
        tolerance, no nearest-matching) for any other `slice_mode`.

    Returns
    -------
    slices : list of xarray.DataArray
        One panel per requested coordinate that had a match; entries with no
        match within tolerance (spatial) or no exact match (non-spatial) are
        skipped, with a warning.
    actual_coords : list of collections.abc.Hashable
        The actual (snapped) coordinate value for each returned panel.

    Warns
    -----
    UserWarning
        For each requested coordinate with no match.
    """
    slices: list[xr.DataArray] = []
    actual_coords: list[Hashable] = []
    for coord in slice_coords:
        if slice_mode in data.coords:
            use_nearest = slice_mode in WORLD_DIMS and slice_spacing is not None
            try:
                slice_da = data.sel(
                    {slice_mode: coord},
                    method="nearest" if use_nearest else None,
                    tolerance=slice_spacing / 2 if use_nearest else None,
                )
            except KeyError:
                warnings.warn(
                    f"No slice found for slice_mode={slice_mode!r} coordinate "
                    f"{coord!r}"
                    + (
                        f" within {slice_spacing / 2:g} of any available position"
                        if use_nearest
                        else ""
                    )
                    + ". This slice will not be plotted.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                continue
            actual_coord = slice_da.coords[slice_mode].item()
        else:
            if not isinstance(coord, numbers.Real):
                raise ValueError(
                    f"slice_mode '{slice_mode}' has no coordinates, so slice_coords "
                    f"must be numeric positional indices, got {coord!r}."
                )
            idx = round(coord)
            slice_da = data.isel({slice_mode: idx})
            actual_coord = float(coord)
        slices.append(slice_da)
        actual_coords.append(actual_coord)
    return slices, actual_coords


class VolumePlotter:
    """Manager for volume slice plots with coordinate-based overlay support.

    This class maintains the state of a figure with multiple axes, each representing a
    slice through a volume at a specific coordinate. It enables overlaying multiple
    volumes on the same axes by matching coordinates.

    Parameters
    ----------
    slice_mode : str
        World dimension (`"z"`, `"y"`, `"x"`) or extra non-voxel dimension to
        slice. Native voxel dimensions (`"k"`, `"j"`, `"i"`) are not valid slice
        modes.
    figure : matplotlib.figure.Figure, optional
        The figure containing the axes. If not provided, a new figure will be created
        on the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume].
    axes : numpy.ndarray[matplotlib.axes.Axes] or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or an array of them. If not
        provided, axes will be created on the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume].
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    yincrease : bool, default: False
        Whether the y-axis increases upward. When `False`, y coordinates decrease
        upward.
    xincrease : bool, default: True
        Whether the x-axis increases to the right.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used whenever oblique voxel-to-world data is
        resampled onto the axis-aligned world grid for display -- for the whole
        array upfront when `slice_mode` is spatial (`"z"`/`"y"`/`"x"`) or any
        other non-`pose` dim (e.g. `"region"`); per-panel, after slicing, when
        `slice_mode="pose"` (the only dimension that can vary the underlying
        voxel-to-world geometry). Applies to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume]; mask data passed
        to [`add_contours`][confusius.plotting.VolumePlotter.add_contours] always
        resamples with `"nearest"` regardless of this setting, since blending
        distinct integer labels together is never meaningful.
    resample_fill_value : float, optional
        Value assigned to voxels outside the source data's field of view after
        resampling oblique data. If not provided, defaults to the source data's own
        minimum value (see
        [`resample_volume`][confusius.registration.resample_volume]).
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.

    Attributes
    ----------
    slice_mode : str
        World dimension (`"z"`, `"y"`, `"x"`) or extra non-voxel dimension being
        sliced.
    figure : matplotlib.figure.Figure or None
        The figure. `None` until the first call to
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume] when no figure
        is provided at construction time.
    axes : numpy.ndarray or None
        Array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes]. `None` until the
        first call to [`add_volume`][confusius.plotting.VolumePlotter.add_volume]
        when no axes are provided at construction time.
    """

    axes: "npt.NDArray[Any] | None"

    def __init__(
        self,
        slice_mode: str = "z",
        figure: "Figure | None" = None,
        axes: "npt.NDArray[Any] | Axes | None" = None,
        *,
        bg_color: str = "black",
        fg_color: str | None = None,
        yincrease: bool = False,
        xincrease: bool = True,
        resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
        resample_fill_value: float | None = None,
        transpose: bool = False,
    ):
        self.slice_mode = slice_mode
        self._transpose = transpose
        if axes is not None and not isinstance(axes, np.ndarray):
            axes = np.asarray([[axes]])
        self.axes = axes
        self._user_provided_axes = axes is not None
        if figure is None and axes is not None:
            from matplotlib.axes import Axes

            first_axis = axes.flat[0]
            if not isinstance(first_axis, Axes):
                raise TypeError("axes must contain matplotlib.axes.Axes instances.")
            self.figure = first_axis.get_figure(root=True)
        else:
            self.figure = figure
        self._bg_color = bg_color
        self._fg_color = fg_color
        self._yincrease = yincrease
        self._xincrease = xincrease
        self._coord_to_axis: dict[Hashable, int] = {}
        # Explicitly tracked axis data limits to avoid matplotlib's auto-margin.
        self._axis_xlims: dict[int, tuple[float, float]] = {}
        self._axis_ylims: dict[int, tuple[float, float]] = {}

        self._resample_interpolation = resample_interpolation
        self._resample_fill_value = resample_fill_value
        self._hover_manager = _HoverManager()
        self._slice_axis_grid: SliceAxisGrid | None = None

    def _resample_pose_slices_to_world_grid(
        self,
        slices: list[xr.DataArray],
        *,
        interpolation: Literal["linear", "nearest", "bspline"] | None = None,
        fill_value: float | None = None,
    ) -> list[xr.DataArray]:
        """Regularize each `pose`-faceted panel for world-space display.

        No-op unless `slice_mode` is `pose` (the only case this function ever runs
        for): every other `slice_mode` -- spatial (`"z"`/`"y"`/`"x"`) or a non-pose
        extra dim (e.g. `"region"`) -- already has the whole array resampled and
        materialized upfront in `_prepare_slice_inputs`, since neither case can
        vary the voxel-to-world affine per panel (only `pose` can, see
        `require_scalar_pose_affine`). A per-pose affine stack can't be resampled
        as a whole (`resample_volume` requires a scalar affine), so `pose` is
        deferred here, per-panel, after `.isel` has collapsed each panel to its own
        single affine. `match_coordinates` matches these panels by the facet's own
        coordinate value (e.g. a pose label), never by grid position, so no shared
        discretization is needed across volumes here -- contrast with
        `_prepare_slice_inputs`'s spatial-`slice_mode` resample, where the slice
        axis's discretization genuinely must be shared (`SliceAxisGrid`) for
        `match_coordinates` to work.

        Each panel is resampled via `_resample_to_planar_world_grid` (see that
        function for the collapse-to-a-2D-plane pre-flight check and the no-op
        short-circuit for already axis-aligned panels), then materialized.

        Parameters
        ----------
        slices : list of xarray.DataArray
            Per-panel 2D (or 2D-plus-time) slices already `isel`'d along `slice_mode`.
        interpolation : {"linear", "nearest", "bspline"}, optional
            Interpolation method for the resample. If not provided, defaults to
            `self._resample_interpolation`.
        fill_value : float, optional
            Value assigned to voxels outside the source panel's field of view for
            the resample. If not provided, defaults to `self._resample_fill_value`.

        Returns
        -------
        list of xarray.DataArray
            The input `slices`, each renamed onto 1D `z`/`y`/`x` dims, unchanged
            for anything else (no voxel-to-world geometry, ...).

        Raises
        ------
        ValueError
            If a panel's spatial geometry is oblique to the world axes and would
            not collapse to a 2D plane.
        """
        if self.slice_mode != POSE_DIM:
            return slices
        interp = (
            self._resample_interpolation if interpolation is None else interpolation
        )
        fill = self._resample_fill_value if fill_value is None else fill_value
        resampled = []
        for slice_da in slices:
            grid = _resample_to_planar_world_grid(
                slice_da,
                slice_mode=self.slice_mode,
                interpolation=interp,
                fill_value=fill,
            )
            grid = materialize_axis_aligned_world_grid_for_display(grid)
            world_dims = [d for d in grid.dims if str(d) in WORLD_DIMS]
            squeeze_dims = [d for d in world_dims if grid.sizes[d] == 1]
            if squeeze_dims:
                grid = grid.squeeze(dim=squeeze_dims)
            assert grid.ndim == 2, (
                "predicted shape should have already rejected a non-collapsing "
                f"panel; got shape {grid.shape} with dims {list(grid.dims)}"
            )
            resampled.append(grid)
        return resampled

    def _extract_display_slices(
        self,
        data: xr.DataArray,
        slice_coords: Sequence[Hashable],
        *,
        slice_spacing: float | None = None,
        interpolation: Literal["linear", "nearest", "bspline"] | None = None,
        fill_value: float | None = None,
    ) -> tuple[list[xr.DataArray], list[Hashable], str, str]:
        """Extract per-panel slices, resample to world space if requested, and resolve
        the row/column display dims (applying `transpose`).

        Parameters
        ----------
        data : xarray.DataArray
            Data already passed through `_prepare_slice_inputs`.
        slice_coords : collections.abc.Sequence of collections.abc.Hashable
            Coordinate values along `slice_mode` at which to extract slices.
        slice_spacing : float, optional
            `data`'s own world spacing along `slice_mode`, from
            `_prepare_slice_inputs`. Used as the nearest-neighbour tolerance for
            a spatial `slice_mode`; ignored otherwise.
        interpolation : {"linear", "nearest", "bspline"}, optional
            Interpolation method for the resample. If not provided, defaults to
            `self._resample_interpolation`.
        fill_value : float, optional
            Value assigned to voxels outside the source panel's field of view. If not
            provided, defaults to `self._resample_fill_value`.

        Returns
        -------
        slices : list of xarray.DataArray
            One 2D (or 2D-plus-unsliced-extra-dim) panel per requested slice
            coordinate that had a match (see `_extract_slices`).
        actual_coords : list of collections.abc.Hashable
            The actual coordinate value matched for each returned panel.
        dim_row : str
            Dim displayed on the row (y) axis.
        dim_col : str
            Dim displayed on the column (x) axis.
        """
        slices, actual_coords = _extract_slices(
            data, self.slice_mode, slice_coords, slice_spacing=slice_spacing
        )
        slices = self._resample_pose_slices_to_world_grid(
            slices, interpolation=interpolation, fill_value=fill_value
        )
        display_dims = (
            [str(d) for d in slices[0].dims]
            if slices
            else [str(d) for d in data.dims if d != self.slice_mode]
        )
        dim_row, dim_col = display_dims[::-1] if self._transpose else display_dims
        # Reorder each panel's own array axes to (dim_row, dim_col): geometry
        # (`_slice_edges_and_centers`) only reads dim_row/dim_col by name, but the
        # pixel array handed to `pcolormesh` must match that row/col axis order.
        slices = [s.transpose(dim_row, dim_col) for s in slices]
        return slices, actual_coords, dim_row, dim_col

    def _ensure_figure(
        self,
        n_slices: int,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
        x_range: float | None = None,
        y_range: float | None = None,
    ) -> None:
        """Create figure and/or axes if not already fully initialized."""
        import matplotlib.pyplot as plt

        if self.figure is not None and self.axes is not None:
            return

        _nrows, _ncols = _compute_grid_dims(n_slices, nrows, ncols)

        if self.figure is None:
            if (
                x_range is not None
                and y_range is not None
                and x_range > 0
                and y_range > 0
            ):
                aspect = x_range / y_range
                subplot_width = _BASE_SIZE * max(1.0, aspect)
                subplot_height = _BASE_SIZE * max(1.0, 1.0 / aspect)
            else:
                subplot_width = subplot_height = _BASE_SIZE

            fig_width = max(8.0, min(20.0, _ncols * subplot_width + 1.0))
            fig_height = min(16.0, _nrows * subplot_height)

            self.figure, axes_array = plt.subplots(
                _nrows,
                _ncols,
                figsize=(fig_width, fig_height),
                dpi=dpi,
                squeeze=False,
                layout="constrained",
            )
        else:
            axes_array = self.figure.subplots(_nrows, _ncols, squeeze=False)

        self.axes = np.array(axes_array)
        self.figure.patch.set_facecolor(self._bg_color)

    def _attach_or_update_hover_manager(self, roi_labels: dict[int, str]) -> None:
        """Ensure hover manager is attached to figure and update its ROI labels.

        Parameters
        ----------
        roi_labels : dict[int, str]
            Mapping from integer label to display name during mouse hover.
        """

        if self.figure is not None:
            if not self._hover_manager.is_attached():
                self._hover_manager.attach_figure(self.figure)

            self._hover_manager.roi_labels.update(roi_labels)

    def _slice_match_tolerance(self, slice_spacing: float | None) -> float:
        """Cross-volume matching tolerance for `_find_matching_axes`.

        Parameters
        ----------
        slice_spacing : float, optional
            `data`'s own world spacing along `self.slice_mode`, from
            `_prepare_slice_inputs`.

        Returns
        -------
        float
            `slice_spacing / 2` when `self.slice_mode` is spatial and
            `slice_spacing` is known, so two volumes at different native
            resolutions can still overlay; `0.0` (exact match only) otherwise --
            nearness has no physical meaning for a non-spatial `self.slice_mode`.
        """
        if self.slice_mode in WORLD_DIMS and slice_spacing is not None:
            return slice_spacing / 2
        return 0.0

    def _find_matching_axes(
        self, actual_coords: list[Hashable], tolerance: float
    ) -> list[tuple[int, int]]:
        """Find axis indices matching the target coordinates.

        Uses the coordinate-to-axis mapping stored when the figure was first created,
        avoiding any dependency on axis titles.

        Parameters
        ----------
        actual_coords : list[collections.abc.Hashable]
            The actual coordinate values of the slices being plotted.
        tolerance : float
            Tolerance for matching numeric coordinates. For a spatial
            `slice_mode`, this should be the new volume's own slice spacing
            halved, so two volumes at different native resolutions can still
            overlay; `0.0` forces an exact match, appropriate for a non-spatial
            `slice_mode` (e.g. a `pose` id), where nearness has no physical
            meaning. Non-numeric coordinates (e.g. region labels) always match by
            equality regardless of `tolerance`.

        Returns
        -------
        list[tuple[int, int]]
            List of `(axis_flat_idx, slice_idx)` tuples for matched coordinates.
        """
        matched = []
        for slice_idx, target_coord in enumerate(actual_coords):
            best_axis_idx: int | None = None
            best_distance = math.inf
            for stored_coord, axis_idx in self._coord_to_axis.items():
                if not _coords_match(stored_coord, target_coord, tolerance):
                    continue
                # Among several stored coordinates within tolerance (e.g. a wide
                # spacing-derived tolerance spanning multiple existing axes), pick
                # the closest one -- not merely the first one encountered in dict
                # (insertion/ascending) order, which can otherwise steal a nearby
                # axis that another, more distant slice actually belongs to.
                distance = (
                    abs(float(stored_coord) - float(target_coord))
                    if isinstance(stored_coord, numbers.Real)
                    and isinstance(target_coord, numbers.Real)
                    else 0.0
                )
                if distance < best_distance:
                    best_distance = distance
                    best_axis_idx = axis_idx
            if best_axis_idx is not None:
                matched.append((best_axis_idx, slice_idx))
        return matched

    @property
    def _text_color(self) -> str:
        """Foreground color: explicit fg_color or WCAG-derived contrast color."""
        if self._fg_color is not None:
            return self._fg_color
        return _auto_fg_color(self._bg_color)

    def _style_ax(self, ax: "Axes") -> None:
        """Apply background and spine/tick styling to an axes."""
        color = self._text_color
        ax.set_facecolor(self._bg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        ax.tick_params(colors=color, which="both")

    def _set_ax_lims(
        self,
        ax: "Axes",
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """Set axis limits respecting the x/y increase direction."""
        ax.set_ylim(ylim if self._yincrease else (ylim[1], ylim[0]))
        ax.set_xlim(xlim if self._xincrease else (xlim[1], xlim[0]))

    def _update_stored_lim(
        self,
        store: dict[int, tuple[float, float]],
        axis_idx: int,
        new_lim: tuple[float, float],
    ) -> tuple[float, float]:
        """Expand the stored limit for `axis_idx` to encompass `new_lim`."""
        if axis_idx in store:
            prev = store[axis_idx]
            new_lim = (min(prev[0], new_lim[0]), max(prev[1], new_lim[1]))
        store[axis_idx] = new_lim
        return new_lim

    def _warn_unmatched(self, unmatched_slices: list[tuple[int, Hashable]]) -> None:
        """Warn about slice coordinates that could not be matched to axes."""
        unmatched_str = ", ".join(
            f"{self.slice_mode}={_format_coord(coord)}" for _, coord in unmatched_slices
        )
        available_coords = [_format_coord(c) for c in self._coord_to_axis]
        warnings.warn(
            f"Could not find matching axes for slices: {unmatched_str}. "
            f"These slices will not be plotted. "
            f"Available coordinates: {available_coords}",
            stacklevel=find_stack_level(),
        )

    def _build_slice_title(self, data: xr.DataArray, coord: Hashable) -> str:
        """Build a slice title such as `z = 0.001 mm` or `region = LGN`."""
        units = (
            data.coords[self.slice_mode].attrs.get("units")
            if self.slice_mode in data.coords
            else None
        )
        title = f"{self.slice_mode} = {_format_coord(coord)}"
        if units:
            title += f" {units}"
        return title

    def _init_sequential_layout(
        self, actual_coords: list[Hashable]
    ) -> list[tuple[int, int]]:
        """Initialise the coordinate-to-axis map and return sequential plot indices."""
        if not self._coord_to_axis:
            self._coord_to_axis = {
                coord: idx for idx, coord in enumerate(actual_coords)
            }
        return [(idx, idx) for idx in range(len(actual_coords))]

    def _prepare_slice_inputs(
        self,
        data: xr.DataArray,
        *,
        interpolation: Literal["linear", "nearest", "bspline"] | None = None,
    ) -> tuple[xr.DataArray, float | None]:
        """Coerce complex, squeeze, validate `slice_mode`/3D, and sort display coords.

        Shared by `add_volume`/`add_composite`/`add_contours`. `interpolation` lets a
        mask force `"nearest"` regardless of `self._resample_interpolation`.

        Resamples/materializes the whole array up front for every `slice_mode`
        except `pose`; only `pose` can vary the voxel-to-world affine per panel
        (`require_scalar_pose_affine`), so it's the only case that must wait for
        `_resample_pose_slices_to_world_grid`'s per-panel resample after `.isel` has
        collapsed each panel to its own scalar affine.

        Parameters
        ----------
        data : xarray.DataArray
            VoxelData array to prepare for slicing/display.
        interpolation : {"linear", "nearest", "bspline"}, optional
            Interpolation method used for any resample this triggers. If not provided,
            defaults to `self._resample_interpolation`.

        Returns
        -------
        data : xarray.DataArray
            `data` with complex values converted to magnitude; resampled and
            materialized onto plain `z`/`y`/`x` dims for every `slice_mode` except
            `pose`; size-1 dims other than `self.slice_mode` squeezed away; and the two
            display dims sorted into ascending coordinate order.
        slice_spacing : float, optional
            `data`'s own world spacing along `self.slice_mode`, captured from its
            voxel-to-world geometry before materialize strips it. `None` for a
            non-spatial `self.slice_mode`, where spacing has no meaning.

        Raises
        ------
        ValueError
            If `self.slice_mode` is not a valid dimension/world coordinate of `data`, if
            the result is not 3D after squeezing, or if `data`'s spatial geometry cannot
            be resampled for display (e.g. pose-dependent geometry for a spatial
            `self.slice_mode`, or geometry that would not collapse to a 2D plane for a
            non-pose extra `self.slice_mode`).
        """
        data = ensure_voxeldata(data)
        data = coerce_complex_to_magnitude(data, caller="VolumePlotter")
        # Data is computed here to avoid repeated computations of the same Dask graph
        # downstream (per-panel .isel, etc.).
        data = data.compute()

        _validate_slice_mode(data, self.slice_mode)

        # pose always means distinct probe positions, so a non-pose slice_mode can't
        # proceed with more than one left un-reduced.
        if self.slice_mode != POSE_DIM:
            if POSE_DIM in data.dims and data.sizes[POSE_DIM] == 1:
                data = data.squeeze(dim=POSE_DIM)
            require_scalar_pose_affine(data, f"slice_mode={self.slice_mode!r}")

        resolved_interpolation = (
            self._resample_interpolation if interpolation is None else interpolation
        )

        slice_spacing: float | None = None
        if self.slice_mode in WORLD_DIMS:
            data, slice_axis_grid = _resample_to_shared_slice_axis_grid(
                data,
                self.slice_mode,
                slice_axis_grid=self._slice_axis_grid,
                interpolation=resolved_interpolation,
                fill_value=self._resample_fill_value,
            )

            # Capture the slice axis's spec so a later volume/mask on this same plotter
            # lines up on the same physical slices. The two in-plane axes are never
            # shared (each volume keeps its own native resolution/orientation, see
            # SliceAxisGrid).
            if self._slice_axis_grid is None:
                self._slice_axis_grid = slice_axis_grid

            # `data`'s own actual spacing, not `slice_axis_grid.spacing` -- an
            # already axis-aligned volume keeps its own native grid regardless of
            # any established SliceAxisGrid (see `_resample_to_shared_slice_axis_grid`),
            # so the two can differ. Captured here, before materialize strips the
            # index.
            voxel_dim = VOXEL_DIMS[WORLD_DIMS.index(self.slice_mode)]
            slice_spacing = get_voxel_to_world_index_spacing(data)[voxel_dim]
        elif self.slice_mode != POSE_DIM:
            # Non-pose extra dim (e.g. "region"): slicing along it never touches
            # the voxel-to-world affine (only `pose` can vary geometry), so unlike
            # `pose` there's no need to wait for a per-panel `.isel` before
            # resampling.
            data = _resample_to_planar_world_grid(
                data,
                slice_mode=self.slice_mode,
                interpolation=resolved_interpolation,
                fill_value=self._resample_fill_value,
            )

        # For `pose`, materializing here (before per-pose `.isel`) would be wrong:
        # A pose-dependent affine can be individually axis-aligned per pose while still
        # varying across poses, and materialize's world-coordinate lookup collapses
        # every non-spatial dim to its first index, silently mislabeling every other
        # pose with pose 0's world coordinates. `_resample_pose_slices_to_world_grid`
        # materializes correctly instead, per panel, after `.isel` has collapsed `pose`
        # to a scalar affine.
        if self.slice_mode != POSE_DIM:
            data = materialize_axis_aligned_world_grid_for_display(data)

        squeeze_dims = [
            d
            for d in data.dims
            if d != self.slice_mode
            and data.sizes[d] == 1
            and (self.slice_mode not in WORLD_DIMS or d not in WORLD_DIMS)
        ]
        if squeeze_dims:
            data = data.squeeze(dim=squeeze_dims)

        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D, but got shape {data.shape} with dims "
                f"{list(data.dims)}."
            )
        # Only the two display dims need sorting for pcolormesh/contour geometry;
        # sorting slice_mode itself would silently reorder panels (e.g. non-monotonic
        # z, or a "region" dim built from an arbitrary list of acronyms).
        display_dims = [d for d in data.dims if d != self.slice_mode]
        return sort_coords_for_plot(data, display_dims), slice_spacing

    def _resolve_axes_layout(
        self,
        data: xr.DataArray,
        n_slices: int,
        actual_coords: list[Hashable],
        dim_row: str,
        dim_col: str,
        *,
        slice_spacing: float | None,
        match_coordinates: bool,
        nrows: int | None,
        ncols: int | None,
        dpi: int | None,
    ) -> list[tuple[int, int]]:
        """Resolve the per-slice axis assignment, creating the figure if needed.

        Parameters
        ----------
        data : xarray.DataArray
            Data already passed through `_prepare_slice_inputs`.
        n_slices : int
            Number of panels being placed.
        actual_coords : list[collections.abc.Hashable]
            The actual (snapped) coordinate value for each panel.
        dim_row : str
            Dim displayed on the row (y) axis.
        dim_col : str
            Dim displayed on the column (x) axis.
        slice_spacing : float, optional
            `data`'s own world spacing along `self.slice_mode`, used (halved) as
            the cross-volume matching tolerance for a spatial `self.slice_mode`.
            Ignored (exact matching) for any other `self.slice_mode`.
        match_coordinates : bool
            Whether to match panels onto the existing coordinate-to-axis mapping
            instead of laying them out sequentially.
        nrows : int, optional
            Number of rows when creating a new figure.
        ncols : int, optional
            Number of columns when creating a new figure.
        dpi : int, optional
            Figure resolution in dots per inch, for a new figure.

        Returns
        -------
        list[tuple[int, int]]
            List of `(axis_flat_idx, slice_idx)` tuples for the panels to draw.

        Raises
        ------
        ValueError
            If `match_coordinates` is `True` but no axes exist yet, or if
            user-provided axes don't match the number of slices.
        """
        if match_coordinates:
            if self.axes is None:
                raise ValueError(
                    "Cannot match coordinates: no existing axes. Either create a "
                    "VolumePlotter with axes or use match_coordinates=False."
                )
            tolerance = self._slice_match_tolerance(slice_spacing)
            matched_indices = self._find_matching_axes(actual_coords, tolerance)
            matched_slice_indices = {idx for _, idx in matched_indices}
            unmatched_slices = [
                (idx, actual_coords[idx])
                for idx in range(n_slices)
                if idx not in matched_slice_indices
            ]
            if unmatched_slices:
                self._warn_unmatched(unmatched_slices)
            return matched_indices

        if self.axes is None:
            x_range = None
            y_range = None
            if dim_col in data.coords and dim_row in data.coords:
                x_coords = data.coords[dim_col].values.astype(float)
                y_coords = data.coords[dim_row].values.astype(float)
                x_range = float(np.max(x_coords) - np.min(x_coords))
                y_range = float(np.max(y_coords) - np.min(y_coords))
            self._ensure_figure(
                n_slices,
                nrows=nrows,
                ncols=ncols,
                dpi=dpi,
                x_range=x_range,
                y_range=y_range,
            )

        if self._user_provided_axes:
            assert self.axes is not None
            if n_slices != self.axes.size:
                raise ValueError(
                    f"Number of slices ({n_slices}) must match number of axes "
                    f"({self.axes.size}). Got {n_slices} slice_coords but axes "
                    f"has shape {self.axes.shape}."
                )

        return self._init_sequential_layout(actual_coords)

    def _style_slice_axis(
        self,
        ax: "Axes",
        axis_idx: int,
        data: xr.DataArray,
        coord: Hashable,
        dim_row: str,
        dim_col: str,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        *,
        show_titles: bool,
        show_axis_labels: bool,
        show_axis_ticks: bool,
        show_axes: bool,
        title_fontsize: float | None,
        label_fontsize: float | None,
        tick_fontsize: float | None,
    ) -> None:
        """Apply post-draw styling (aspect, spines, title, labels, lims) to a slice axis."""
        ax.set_aspect("equal")
        self._style_ax(ax)

        text_color = self._text_color
        ax.set_title(
            self._build_slice_title(data, coord) if show_titles else "",
            color=text_color,
            fontsize=title_fontsize,
        )

        if show_axes:
            if show_axis_labels:
                ax.set_xlabel(
                    _build_axis_label(data, dim_col),
                    color=text_color,
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    _build_axis_label(data, dim_row),
                    color=text_color,
                    fontsize=label_fontsize,
                )
            if show_axis_ticks:
                ax.tick_params(labelsize=tick_fontsize)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        else:
            ax.axis("off")

        # Expand stored limits to encompass overlaid volumes with different extents.
        current_xlim = self._update_stored_lim(
            self._axis_xlims,
            axis_idx,
            (float(x_edges.min()), float(x_edges.max())),
        )
        current_ylim = self._update_stored_lim(
            self._axis_ylims,
            axis_idx,
            (float(y_edges.min()), float(y_edges.max())),
        )
        self._set_ax_lims(ax, current_xlim, current_ylim)

    def _prepare_alpha_slices(
        self,
        alpha: "float | xr.DataArray | None",
        data: xr.DataArray,
        *,
        slice_coords: Sequence[Hashable],
        dim_row: Hashable,
        dim_col: Hashable,
    ) -> "tuple[float | None, list[np.ndarray] | None]":
        """Validate and slice a per-voxel `alpha`, or pass a scalar through.

        `data` is assumed already processed by `_prepare_slice_inputs`. Returns
        `(alpha, alpha_slices)`: exactly one is non-`None` -- `alpha_slices` for a
        per-voxel DataArray input, the (possibly `None`) scalar `alpha` otherwise.
        """
        if alpha is not None and not isinstance(alpha, (int, float, xr.DataArray)):
            raise TypeError(
                "`alpha` must be a scalar or a DataArray, not "
                f"{type(alpha).__name__}; a bare array carries no coordinates, so "
                "it cannot be validated or aligned against `data`."
            )

        if not isinstance(alpha, xr.DataArray):
            return alpha, None

        # Preprocess alpha exactly like data (squeeze unitary dims, sort coords
        # ascending). Otherwise, an alpha array created from the data could end up
        # with a shape or coordinate mismatch after slicing. alpha's own spacing is
        # unused: `validate_matching_coordinates` below guarantees its coordinates
        # are bit-identical to `data`'s, so extracting it at `data`'s own actual
        # (already-snapped) coordinates always exact-matches, no tolerance needed.
        alpha, _ = self._prepare_slice_inputs(alpha)
        if set(alpha.dims) != set(data.dims):
            raise ValueError(
                f"`alpha` dims {sorted(str(d) for d in alpha.dims)} do not match "
                f"`data` dims {sorted(str(d) for d in data.dims)}."
            )
        for dim in data.dims:
            if alpha.sizes[dim] != data.sizes[dim]:
                raise ValueError(
                    f"`alpha` size along '{dim}' ({alpha.sizes[dim]}) does not "
                    f"match `data` size ({data.sizes[dim]})."
                )
        validate_matching_coordinates(data, alpha, left_name="data", right_name="alpha")
        alpha_da_slices, _ = _extract_slices(alpha, self.slice_mode, slice_coords)
        alpha_da_slices = self._resample_pose_slices_to_world_grid(alpha_da_slices)
        alpha_da_slices = [s.transpose(dim_row, dim_col) for s in alpha_da_slices]
        return None, [s.values for s in alpha_da_slices]

    def add_volume(
        self,
        data: xr.DataArray,
        *,
        slice_coords: Sequence[Hashable] | None = None,
        match_coordinates: bool = True,
        cmap: "str | Colormap | None" = None,
        norm: "Normalize | None" = None,
        vmin: float | None = None,
        vmax: float | None = None,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        alpha: "float | xr.DataArray | None" = None,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        cbar_kwargs: "dict[str, Any] | None" = None,
        roi_labels: dict[int, str] | None = None,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        fontsize: float | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot or overlay a volume on the axes.

        Parameters
        ----------
        data : xarray.DataArray
            3D volume data. Unitary dimensions (except `slice_mode`) are squeezed
            before processing. Complex-valued inputs are converted to magnitude
            (`abs(data)`) with a warning.
        slice_coords : list[collections.abc.Hashable], optional
            Specific coordinates to plot. Numeric coordinates are matched by
            nearest-neighbour lookup; non-numeric coordinates (e.g. region labels)
            require an exact match. If not provided, uses all coordinates from data.
        match_coordinates : bool, default: True
            Whether to match slice coordinates to the stored coordinate mapping (for
            overlays) instead of plotting sequentially on all axes (which requires
            an exact axis count match).
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap. When not provided, falls back to `data.attrs["cmap"]` if
            present, otherwise `"gray"`.
        norm : matplotlib.colors.Normalize, optional
            Normalization instance (e.g. `BoundaryNorm` for integer label maps). When
            not provided, falls back to `data.attrs["norm"]` if present. When a norm
            is active, `vmin` and `vmax` are ignored.
        vmin : float, optional
            Lower bound of the colormap. Defaults to the 2nd percentile. Ignored
            when `norm` is provided explicitly (that is, not just inherited from data
            attributes).
        vmax : float, optional
            Upper bound of the colormap. Defaults to the 98th percentile. Ignored
            when `norm` is provided explicitly (that is, not just inherited from data
            attributes).
        threshold : float, optional
            Threshold value for masking.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Whether to mask values below or above threshold.
        alpha : float or xarray.DataArray, optional
            Opacity of the image: a single scalar value, or a DataArray sharing `data`'s
            dims, shape, and coordinates (for independent per-slice, per-voxel opacity —
            e.g. fading out low-confidence voxels). A per-voxel opacity must be a
            DataArray, not a bare array, so it can be validated and aligned against
            `data`. If not provided, the colormap's own alpha channel is respected.
        show_colorbar : bool, default: True
            Whether to add a colorbar.
        cbar_label : str, optional
            Label for the colorbar.
        cbar_kwargs : dict, optional
            Additional keyword arguments forwarded to
            [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
            (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the
            colorbar when it spans a multi-panel grid, since the defaults are sized
            for a single axes.
        roi_labels : dict[int, str], optional
            Mapping from integer label to display name. When provided (or when
            `data.attrs["roi_labels"]` is populated), hovering the cursor over a
            voxel shows `<layer.name>=<id> (<name>)` in the matplotlib status bar.
        show_titles : bool, default: True
            Whether to display subplot titles.
        show_axis_labels : bool, default: True
            Whether to display axis labels.
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When `False`,
            overrides `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels and the colorbar label use `0.9 * fontsize`; tick
            labels use `0.85 * fontsize`. If not provided, uses the active Matplotlib
            defaults.
        nrows : int, optional
            Number of rows in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        ncols : int, optional
            Number of columns in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when using an existing figure.

        Returns
        -------
        VolumePlotter
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If no matching coordinates are found or axis count doesn't match.
        ValueError
            If `alpha` is a DataArray and its dims, shape, or coordinates do not
            match `data`.
        TypeError
            If `alpha` is neither a scalar nor a DataArray.
        """
        resolved_roi_labels = _normalize_roi_labels(
            roi_labels if roi_labels is not None else data.attrs.get("roi_labels")
        )

        data, slice_spacing = self._prepare_slice_inputs(data)

        if slice_coords is None:
            slice_coords = _default_slice_coords(data, self.slice_mode)

        unthresholded_slices, actual_coords, dim_row, dim_col = (
            self._extract_display_slices(
                data, slice_coords, slice_spacing=slice_spacing
            )
        )
        n_slices = len(unthresholded_slices)
        if n_slices == 0:
            # Every requested slice_coords entry was skipped (already warned by
            # `_extract_slices`) -- nothing to draw.
            return self

        alpha, alpha_slices = self._prepare_alpha_slices(
            alpha, data, slice_coords=slice_coords, dim_row=dim_row, dim_col=dim_col
        )

        norm = _resolve_norm(
            slices=unthresholded_slices,
            norm=norm,
            data_attrs_norm=data.attrs.get("norm"),
            vmin=vmin,
            vmax=vmax,
        )

        thresholded_slices = _threshold_slices(
            unthresholded_slices, threshold=threshold, threshold_mode=threshold_mode
        )

        cmap = _resolve_cmap(
            cmap=cmap,
            data_attrs_cmap=data.attrs.get("cmap"),
            norm=norm,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )

        plot_indices = self._resolve_axes_layout(
            data,
            n_slices,
            actual_coords,
            dim_row,
            dim_col,
            slice_spacing=slice_spacing,
            match_coordinates=match_coordinates,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
        )

        assert (self.axes is not None) and (self.figure is not None)

        text_color = self._text_color
        title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)
        plotted_quadmesh = None

        axes_flat = self.axes.ravel()
        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            arr = thresholded_slices[slice_idx]
            slice_da = unthresholded_slices[slice_idx]
            x_edges, y_edges, hover_x, hover_y = _slice_edges_and_centers(
                slice_da, dim_row, dim_col
            )

            panel_alpha = alpha_slices[slice_idx] if alpha_slices is not None else alpha
            plotted_quadmesh = ax.pcolormesh(
                x_edges,
                y_edges,
                np.ma.masked_invalid(arr),
                cmap=cmap,
                norm=norm,
                alpha=panel_alpha,
            )
            self._attach_or_update_hover_manager(resolved_roi_labels)
            self._hover_manager.register_data_to_axis(
                ax,
                hover_x,
                hover_y,
                slice_da.values,
                role="labels" if resolved_roi_labels else "volume",
                name=str(data.name) if data.name is not None else "value",
                units=data.attrs.get("units"),
            )

            self._style_slice_axis(
                ax,
                axis_idx,
                slice_da,
                actual_coords[slice_idx],
                dim_row,
                dim_col,
                x_edges,
                y_edges,
                show_titles=show_titles,
                show_axis_labels=show_axis_labels,
                show_axis_ticks=show_axis_ticks,
                show_axes=show_axes,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
            )

        if not match_coordinates:
            for ax in axes_flat[n_slices:]:
                ax.set_visible(False)

        if show_colorbar and plotted_quadmesh is not None:
            non_cbar_axes = [
                ax for ax in self.figure.axes if not hasattr(ax, "_colorbar")
            ]
            cbar = self.figure.colorbar(
                plotted_quadmesh, ax=non_cbar_axes, **(cbar_kwargs or {})
            )
            if cbar_label is None:
                long_name = data.attrs.get("long_name")
                units = data.attrs.get("units")
                if long_name and units:
                    cbar_label = f"{long_name} ({units})"
                elif long_name:
                    cbar_label = long_name
                elif units:
                    cbar_label = f"({units})"
            _style_colorbar(
                cbar,
                text_color,
                tick_fontsize,
                label=cbar_label,
                label_fontsize=label_fontsize,
            )

        return self

    def add_stat_map(
        self,
        stat_map: xr.DataArray,
        *,
        slice_coords: Sequence[Hashable] | None = None,
        match_coordinates: bool = True,
        cmap: "str | Colormap | None" = None,
        norm: "Normalize | None" = None,
        vmin: float | None = None,
        vmax: float | None = None,
        auto_range: bool = True,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        alpha: "float | xr.DataArray | None" = None,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        cbar_kwargs: "dict[str, Any] | None" = None,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        fontsize: float | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Overlay a statistical map on the axes.

        Thin wrapper around [`add_volume`][confusius.plotting.VolumePlotter.add_volume]
        that additionally picks the colormap and range automatically based on the sign
        of `stat_map`, as done by
        [`plot_stat_map`][confusius.plotting.plot_stat_map]. Use this method instead of
        `plot_stat_map` to overlay a statistical map onto an existing plot (e.g. one
        built with [`add_composite`][confusius.plotting.VolumePlotter.add_composite]).

        Parameters
        ----------
        stat_map : xarray.DataArray
            Statistical map to plot. 3D volume data. Unitary dimensions (except
            `slice_mode`) are squeezed before processing.
        slice_coords : list[collections.abc.Hashable], optional
            Specific coordinates to plot. Numeric coordinates are matched by
            nearest-neighbour lookup; non-numeric coordinates (e.g. region labels)
            require an exact match. If not provided, uses all coordinates from
            `stat_map`.
        match_coordinates : bool, default: True
            Whether to match slice coordinates to the stored coordinate mapping (for
            overlays) instead of plotting sequentially on all axes (which requires
            an exact axis count match).
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for `stat_map`. If not provided, the default depends on
            `auto_range` and the sign of `stat_map` (see below); an explicit `cmap`
            is always used as-is regardless of `auto_range`.
        norm : matplotlib.colors.Normalize, optional
            Normalization instance (e.g. `TwoSlopeNorm`, `BoundaryNorm`, `LogNorm`)
            for cases `vmin`/`vmax`/`auto_range` can't express. When provided,
            `vmin`, `vmax`, and `auto_range`'s range computation are bypassed
            entirely; `cmap` still follows the usual rules above.
        vmin : float, optional
            Lower bound of the colormap. If not provided, defaults to the minimum
            value of `stat_map`, computed over the full array rather than just the
            displayed slices. Ignored when `norm` is provided, when
            `auto_range=True` and `stat_map` has only non-negative values, or when
            `auto_range=True`, `stat_map` spans both signs and `vmax` is given on
            its own (see `auto_range`).
        vmax : float, optional
            Upper bound of the colormap. If not provided, defaults to the maximum
            value of `stat_map`, computed over the full array rather than just the
            displayed slices. Ignored when `norm` is provided, when
            `auto_range=True` and `stat_map` has only non-positive values, or when
            `auto_range=True`, `stat_map` spans both signs and `vmin` is given on
            its own (see `auto_range`).
        auto_range : bool, default: True
            Whether to pick the colormap range and default colormap automatically
            based on the sign of `stat_map`:

            - Both positive and negative values: diverging, symmetric `[-m, m]`
              range where `m = max(|vmin|, |vmax|)` over the bounds actually
              provided, falling back to the largest magnitude in `stat_map` when
              neither is given, with `cmap` defaulting to `"coolwarm"` — the right
              choice for diverging statistics where the sign is meaningful (e.g.
              t-statistics, correlation coefficients, PCA/ICA component maps).
            - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
              defaulting to `"viridis"` — the right choice for non-diverging
              statistics where only magnitude matters (e.g. R², F-statistics).
            - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
              defaulting to `"viridis_r"` (reversed, so that values near zero map
              to the same end of the colormap in both the non-negative and
              non-positive cases).

            Set to `False` to use the resolved `vmin`/`vmax` directly with no
            zero-anchoring (`cmap` then defaults to `"coolwarm"` regardless of
            sign).
        threshold : float, optional
            Threshold applied to `|stat_map|`. See `threshold_mode` for the masking
            direction. If not provided, no thresholding is applied.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Controls how `threshold` is applied:

            - `"lower"`: set pixels where `|stat_map| < threshold` to NaN.
            - `"upper"`: set pixels where `|stat_map| > threshold` to NaN.

        alpha : float or xarray.DataArray, optional
            Opacity of the `stat_map` overlay: a single scalar value, or a 3D
            DataArray sharing `stat_map`'s dims, shape, and coordinates (for
            independent per-slice, per-voxel opacity, e.g. to fade out
            low-magnitude voxels instead of masking them out with `threshold`). A
            per-voxel opacity must be a DataArray, not a bare array, so it can be
            validated and aligned against `stat_map`. If not provided, the
            colormap's own alpha channel is respected.
        show_colorbar : bool, default: True
            Whether to add a colorbar for `stat_map`.
        cbar_label : str, optional
            Label for the colorbar.
        cbar_kwargs : dict, optional
            Additional keyword arguments forwarded to
            [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
            (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the
            colorbar when it spans a multi-panel grid, since the defaults are sized
            for a single axes.
        show_titles : bool, default: True
            Whether to display subplot titles.
        show_axis_labels : bool, default: True
            Whether to display axis labels.
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When
            `False`, overrides `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels and the colorbar label use `0.9 * fontsize`; tick
            labels use `0.85 * fontsize`. If not provided, uses the active
            Matplotlib defaults.
        nrows : int, optional
            Number of rows in the subplot grid when creating a new figure. If not
            provided, computed automatically.
        ncols : int, optional
            Number of columns in the subplot grid when creating a new figure. If
            not provided, computed automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when using an existing
            figure.

        Returns
        -------
        VolumePlotter
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If no matching coordinates are found or axis count doesn't match.
        ValueError
            If `alpha` is a DataArray and its dims, shape, or coordinates do not
            match `stat_map`.
        TypeError
            If `alpha` is neither a scalar nor a DataArray.

        Examples
        --------
        >>> import xarray as xr
        >>> from confusius.plotting import plot_volume
        >>> anatomical = xr.open_zarr("output.zarr")["power_doppler"]
        >>> t_map = xr.open_zarr("output.zarr")["t_stat"]
        >>> plotter = plot_volume(anatomical, show_colorbar=False)
        >>> plotter = plotter.add_stat_map(t_map, threshold=3.0)
        """
        stat_map = stat_map.compute()
        resolved_vmin, resolved_vmax, resolved_cmap = _resolve_stat_map_style(
            stat_map, vmin, vmax, cmap, auto_range
        )
        return self.add_volume(
            stat_map,
            slice_coords=slice_coords,
            match_coordinates=match_coordinates,
            cmap=resolved_cmap,
            norm=norm,
            vmin=resolved_vmin,
            vmax=resolved_vmax,
            threshold=threshold,
            threshold_mode=threshold_mode,
            alpha=alpha,
            show_colorbar=show_colorbar,
            cbar_label=cbar_label,
            cbar_kwargs=cbar_kwargs,
            show_titles=show_titles,
            show_axis_labels=show_axis_labels,
            show_axis_ticks=show_axis_ticks,
            show_axes=show_axes,
            fontsize=fontsize,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
        )

    def add_composite(
        self,
        data1: xr.DataArray,
        data2: xr.DataArray,
        *,
        resample: bool = True,
        resample_kwargs: "dict[str, Any] | None" = None,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        normalize_strategy: Literal["per_volume", "per_slice", "shared"] = "per_volume",
        slice_coords: Sequence[Hashable] | None = None,
        match_coordinates: bool = False,
        alpha: "float | npt.NDArray[np.floating] | None" = None,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        fontsize: float | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot a red/cyan composite of two volumes on the axes.

        Each slice is rendered as an RGB image where `data1` drives the red channel
        and `data2` drives the green and blue channels (cyan), making overlap
        visible as desaturated grey. This is the same visual encoding used by the
        live registration progress preview.

        Parameters
        ----------
        data1 : xarray.DataArray
            First volume, plotted in red. 3D volume data. Unitary dimensions (except
            `slice_mode`) are squeezed before processing. Complex-valued inputs are
            converted to magnitude (`abs(data)`) with a warning.
        data2 : xarray.DataArray
            Second volume, plotted in cyan. Must have the same dimensionality as
            `data1` after squeezing; when `resample=True` it is resampled onto
            `data1`'s grid before plotting, so its native shape and coordinates may
            differ.
        resample : bool, default: True
            Whether to resample `data2` onto `data1`'s grid using an identity
            transform before blending. When `False`, the two arrays must
            already share the same dimensions and shape, and their coordinates
            must match within `rtol`/`atol`; once validated, `data2`'s
            coordinates are replaced with `data1`'s so the two volumes share
            an exact coordinate frame downstream.
        resample_kwargs : dict, optional
            Extra keyword arguments forwarded to
            [`resample_like`][confusius.registration.resample_like] when
            `resample=True`. Ignored when `resample=False`.
        rtol : float, default: 1e-5
            Relative tolerance used to validate that `data1` and `data2` share
            coordinates when `resample=False`. Widen to accept acquisitions on
            slightly offset grids known to be equivalent. Ignored when
            `resample=True`.
        atol : float, default: 1e-8
            Absolute tolerance used to validate that `data1` and `data2` share
            coordinates when `resample=False`. Ignored when `resample=True`.
        normalize_strategy : {"per_volume", "per_slice", "shared"}, default: "per_volume"
            Intensity normalisation strategy.

            - `"per_volume"`: rescale each input to `[0, 1]` independently over its
              full volume. Preserves slice-to-slice contrast within each array
              but loses the absolute-intensity relationship between `data1` and
              `data2`.
            - `"per_slice"`: rescale each 2D slice independently. Maximises
              contrast on dim slices at the cost of cross-slice comparability.
            - `"shared"`: rescale both volumes together using a single shared
              `[min(data1.min(), data2.min()), max(data1.max(), data2.max())]`
              range. Preserves the absolute-intensity relationship between the
              two inputs, useful when comparing data acquired at the same
              dynamic range.
        slice_coords : Sequence[collections.abc.Hashable], optional
            Coordinate values along `slice_mode` at which to extract slices. Numeric
            coordinates are matched by nearest-neighbour lookup; non-numeric
            coordinates (e.g. region labels) require an exact match. If not provided,
            all coordinate values from `data1` are used.
        match_coordinates : bool, default: False
            If True, match slice coordinates to the stored coordinate mapping of an
            existing figure (for use as an overlay). If False, plot sequentially on
            a fresh grid of axes — the natural mode for a standalone composite plot.
        alpha : float or numpy.ndarray, optional
            Opacity of the composite image, either a single value or a per-voxel
            array matching the shape of the displayed slices. If not provided, the
            image is fully opaque.
        show_titles : bool, default: True
            Whether to display subplot titles showing the slice coordinate.
        show_axis_labels : bool, default: True
            Whether to display axis labels (with units when available).
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show axis decorations. When `False`, overrides
            `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels use `0.9 * fontsize`; tick labels use
            `0.85 * fontsize`. If not provided, uses the active Matplotlib
            defaults.
        nrows : int, optional
            Number of rows in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        ncols : int, optional
            Number of columns in the subplot grid when creating a new figure.
            If not provided, computed automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when using an existing
            figure.

        Returns
        -------
        VolumePlotter
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If either input has a `time` dimension, is not 2D or 3D, lacks
            `slice_mode` as a dimension, or (when `resample=False`) the two
            arrays do not share dims, shape, and coordinates within
            `rtol`/`atol`.

        Notes
        -----
        The composite is rendered with
        [`pcolormesh`][matplotlib.axes.Axes.pcolormesh] using its RGB-`C`
        codepath, so panels line up with overlays drawn by
        [`add_volume`][confusius.plotting.VolumePlotter.add_volume] /
        [`add_contours`][confusius.plotting.VolumePlotter.add_contours].
        Hover tooltips, colormaps, colorbars, and intensity thresholds are
        not supported on composite axes — use `add_volume` for those.
        """
        if normalize_strategy not in ("per_volume", "per_slice", "shared"):
            raise ValueError(
                f"Invalid normalization strategy {normalize_strategy!r}. "
                f"Expected 'per_volume', 'per_slice', or 'shared'."
            )

        if resample:
            from confusius.registration.resampling import resample_like

            data2_name = data2.name or "data2"
            _kw: dict[str, Any] = dict(resample_kwargs or {})
            # The identity transform is always spatial-only (3D): resample_like reads
            # only data1's k/j/i grid, regardless of any time dimension it carries.
            data2 = resample_like(data2, data1, np.eye(len(VOXEL_DIMS) + 1), **_kw)
            data2.name = data2_name

        # data1 is the reference: data2 is either resampled onto data1's exact
        # grid above (resample=True) or validated coordinate-identical to it below
        # (resample=False), so data1's spacing is authoritative for both -- data2's
        # own returned spacing is unused.
        data1, slice_spacing = self._prepare_slice_inputs(data1)
        data2, _ = self._prepare_slice_inputs(data2)

        if not resample:
            if data1.dims != data2.dims:
                raise ValueError(
                    f"With resample=False, data1 and data2 must share dimensions; "
                    f"got {data1.dims} vs {data2.dims}."
                )
            if data1.shape != data2.shape:
                raise ValueError(
                    f"With resample=False, data1 and data2 must share shape; "
                    f"got {data1.shape} vs {data2.shape}."
                )
            validate_matching_coordinates(
                data1,
                data2,
                left_name="data1",
                right_name="data2",
                rtol=rtol,
                atol=atol,
            )
            # Replace data2's coords with data1's so harmless floating-point
            # drift does not propagate into downstream slicing.
            data2 = data2.assign_coords(
                {d: data1.coords[d] for d in data1.dims if d in data1.coords}
            )

        if slice_coords is None:
            slice_coords = _default_slice_coords(data1, self.slice_mode)

        input_slices1, _ = _extract_slices(
            data1, self.slice_mode, slice_coords, slice_spacing=slice_spacing
        )
        input_slices2, _ = _extract_slices(
            data2, self.slice_mode, slice_coords, slice_spacing=slice_spacing
        )
        input_slices1 = self._resample_pose_slices_to_world_grid(input_slices1)
        input_slices2 = self._resample_pose_slices_to_world_grid(input_slices2)

        if normalize_strategy == "per_volume":
            data1 = data1.copy(data=scale_min_max(data1.values.astype(float)))
            data2 = data2.copy(data=scale_min_max(data2.values.astype(float)))
        elif normalize_strategy == "shared":
            arr1 = data1.values.astype(float)
            arr2 = data2.values.astype(float)
            finite = np.concatenate([arr1[np.isfinite(arr1)], arr2[np.isfinite(arr2)]])
            if finite.size == 0:
                raise ValueError(
                    "Cannot normalize data1/data2 with 'shared' strategy: no finite "
                    "values found in either array."
                )
            lo = float(finite.min())
            hi = float(finite.max())
            if hi == lo:
                arr1 = np.zeros_like(arr1)
                arr2 = np.zeros_like(arr2)
            else:
                arr1 = np.clip((arr1 - lo) / (hi - lo), 0.0, 1.0)
                arr2 = np.clip((arr2 - lo) / (hi - lo), 0.0, 1.0)
            data1 = data1.copy(data=arr1)
            data2 = data2.copy(data=arr2)

        slices1, actual_coords, dim_row, dim_col = self._extract_display_slices(
            data1, slice_coords, slice_spacing=slice_spacing
        )
        slices2 = self._resample_pose_slices_to_world_grid(
            _extract_slices(
                data2, self.slice_mode, slice_coords, slice_spacing=slice_spacing
            )[0]
        )
        slices2 = [s.transpose(dim_row, dim_col) for s in slices2]
        input_slices1 = [s.transpose(dim_row, dim_col) for s in input_slices1]
        input_slices2 = [s.transpose(dim_row, dim_col) for s in input_slices2]
        n_slices = len(slices1)

        plot_indices = self._resolve_axes_layout(
            data1,
            n_slices,
            actual_coords,
            dim_row,
            dim_col,
            slice_spacing=slice_spacing,
            match_coordinates=match_coordinates,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
        )

        assert (self.axes is not None) and (self.figure is not None)

        title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)
        axes_flat = self.axes.ravel()

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            slice1 = slices1[slice_idx]
            slice2 = slices2[slice_idx]

            arr1 = slice1.values.astype(float)
            arr2 = slice2.values.astype(float)
            if normalize_strategy == "per_slice":
                arr1 = scale_min_max(arr1)
                arr2 = scale_min_max(arr2)
            rgb = blend_red_cyan(arr1, arr2)

            x_edges, y_edges, hover_x, hover_y = _slice_edges_and_centers(
                slice1, dim_row, dim_col
            )

            ax.pcolormesh(x_edges, y_edges, rgb, alpha=alpha)
            self._attach_or_update_hover_manager({})
            for i, (data, input_slices) in enumerate(
                zip((data1, data2), (input_slices1, input_slices2))
            ):
                self._hover_manager.register_data_to_axis(
                    ax,
                    hover_x,
                    hover_y,
                    input_slices[slice_idx].values,
                    role="volume",
                    name=str(data.name) if data.name is not None else f"data{i + 1}",
                    units=data.attrs.get("units"),
                )

            self._style_slice_axis(
                ax,
                axis_idx,
                slice1,
                actual_coords[slice_idx],
                dim_row,
                dim_col,
                x_edges,
                y_edges,
                show_titles=show_titles,
                show_axis_labels=show_axis_labels,
                show_axis_ticks=show_axis_ticks,
                show_axes=show_axes,
                title_fontsize=title_fontsize,
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
            )

        if not match_coordinates:
            for ax in axes_flat[n_slices:]:
                ax.set_visible(False)

        return self

    def add_contours(
        self,
        mask: xr.DataArray,
        *,
        colors: dict[int | str, str] | str | None = None,
        linewidths: float = 1.5,
        linestyles: str = "solid",
        match_coordinates: bool = True,
        slice_coords: list[Hashable] | None = None,
        fontsize: float | None = None,
        roi_labels: dict[int, str] | None = None,
        **kwargs,
    ) -> "VolumePlotter":
        """Add mask contours to existing axes.

        Parameters
        ----------
        mask : xarray.DataArray
            Integer label map in one of two formats:

            - **Flat label map**: Spatial dims only, e.g. `(k, j, i)`. Background voxels
              labeled `0`; each unique non-zero integer identifies a distinct,
              non-overlapping region. The `region` coordinate of the output holds the
              integer label values.
            - **Stacked mask format**: Has a leading `mask` dimension followed by
              spatial dims, e.g. `(mask, k, j, i)`. Each layer has values in `{0,
              region_id}` and regions may overlap. The `region` coordinate of the
              output holds the `mask` coordinate values (e.g., region label).

            Drawn on the same display geometry as the plotter's own data -- world
            space, resampled onto the axis-aligned grid as needed (see
            [`VolumePlotter`][confusius.plotting.VolumePlotter]).
        colors : dict[int | str, str] or str, optional
            Color specification for contour lines.

            - `dict`: maps each label (integer index) or region acronym (string)
              to a color string.
            - `str`: applies one color to all regions.
            - `None`: colors are derived from `attrs["cmap"]` and
              `attrs["norm"]` when present, otherwise from the
              `tab10`/`tab20` colormap.
        linewidths : float, default: 1.5
            Width of contour lines in points.
        linestyles : str, default: "solid"
            Line style for contour lines (e.g. `"solid"`, `"dashed"`).
        match_coordinates : bool, default: True
            If `True`, overlay contours on axes whose slice coordinate matches the
            mask. If `False`, plot sequentially on all axes.
        slice_coords : list[collections.abc.Hashable], optional
            Coordinate values along the plotter's `slice_mode` at which to draw
            contours. Numeric coordinates are matched by nearest-neighbour lookup;
            non-numeric coordinates (e.g. region labels) require an exact match. If
            not provided, all coordinate values along `slice_mode` are used.
        fontsize : float, optional
            Base font size for text elements when a standalone contour figure is created
            (`match_coordinates=False`). Subplot titles use `fontsize` directly;
            axis labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`.
            If not provided, uses the active Matplotlib defaults.
        roi_labels : dict[int, str], optional
            Mapping from integer label to display name. When provided (or when
            `mask.attrs["roi_labels"]` is populated), hovering the cursor over a
            voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar. The
            cursor samples the underlying label map directly, so hovering inside
            a closed contour is sufficient — there is no need to be on the line.
        **kwargs
            Additional keyword arguments passed to
            [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot].

        Returns
        -------
        VolumePlotter
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If the plotter's `slice_mode` is not a dimension of `mask`.
        ValueError
            If `mask` is not 3D or 4D with a leading `mask` dimension.
        """
        import matplotlib.colors as mcolors
        from skimage.measure import find_contours

        resolved_roi_labels = _normalize_roi_labels(
            roi_labels if roi_labels is not None else mask.attrs.get("roi_labels")
        )

        # Stacked mask format: (mask, z, y, x) — one layer per region.
        if "mask" in mask.dims:
            # Compute once here rather than once per layer below (`np.unique`)
            # plus again inside each layer's recursive `add_contours` call (whose
            # own `_prepare_slice_inputs` would otherwise recompute a dask-backed
            # mask's already-computed-once-per-layer data yet again).
            mask = mask.compute()
            cmap_attr = mask.attrs.get("cmap")
            norm_attr = mask.attrs.get("norm")
            # cmap/norm are dropped on Zarr save; reconstruct from rgb_lookup when
            # present so structure colors survive a serialization round-trip.
            if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in mask.attrs:
                cmap_attr, norm_attr = build_atlas_cmap_and_norm(
                    mask.attrs["rgb_lookup"]
                )
            acronyms = mask.coords["mask"].values

            for i in range(mask.sizes["mask"]):
                layer = mask.isel(mask=i)
                acronym = str(acronyms[i])

                unique_nonzero = [v for v in np.unique(layer.values) if v != 0]
                if not unique_nonzero:
                    continue
                label = int(unique_nonzero[0])

                if isinstance(colors, str):
                    layer_color: str = colors
                elif isinstance(colors, dict):
                    # We accept both acronym-keyed and id-keyed dicts for flexibility.
                    layer_color = colors.get(acronym, colors.get(label, "white"))
                elif cmap_attr is not None and norm_attr is not None:
                    layer_color = mcolors.to_hex(cmap_attr(norm_attr(label)))
                else:
                    layer_color = _get_distinct_colors(mask.sizes["mask"])[i]

                per_layer_colors: dict[int | str, Any] = {label: layer_color}
                self.add_contours(
                    layer,
                    colors=per_layer_colors,
                    linewidths=linewidths,
                    linestyles=linestyles,
                    match_coordinates=match_coordinates,
                    slice_coords=slice_coords,
                    fontsize=fontsize,
                    roi_labels=resolved_roi_labels or None,
                    **kwargs,
                )
            return self

        # `_prepare_slice_inputs` handles a mask exactly like `add_volume`'s data --
        # it can carry the same extra facet dim as `self.slice_mode` (e.g.
        # "region"), and must land on the same native-vs-world dims as the data it
        # overlays. `interpolation="nearest"` regardless of
        # `self._resample_interpolation`: mask/label data is a set of distinct
        # integer regions, and blending them together (linear/bspline) would
        # fabricate boundary values that match no real label.
        mask, slice_spacing = self._prepare_slice_inputs(mask, interpolation="nearest")

        unique_labels = sorted(
            [label for label in np.unique(mask.values) if label != 0]
        )
        if not unique_labels:
            return self

        if colors is None:
            cmap_attr = mask.attrs.get("cmap")
            norm_attr = mask.attrs.get("norm")
            # cmap/norm are dropped on Zarr save; reconstruct from rgb_lookup when
            # present so structure colors survive a serialization round-trip.
            if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in mask.attrs:
                cmap_attr, norm_attr = build_atlas_cmap_and_norm(
                    mask.attrs["rgb_lookup"]
                )
            if cmap_attr is not None and norm_attr is not None:
                color_map = {
                    label: mcolors.to_hex(cmap_attr(norm_attr(label)))
                    for label in unique_labels
                }
            else:
                distinct_colors = _get_distinct_colors(len(unique_labels))
                color_map = {
                    label: color for label, color in zip(unique_labels, distinct_colors)
                }
        elif isinstance(colors, str):
            color_map = {label: colors for label in unique_labels}
        else:
            color_map = colors

        if slice_coords is None:
            slice_coords = _default_slice_coords(mask, self.slice_mode)

        # Always "nearest" for the per-panel world resample too: mask/label data is a
        # set of distinct integer regions, never meaningfully blended.
        slices, actual_coords, dim_row, dim_col = self._extract_display_slices(
            mask, slice_coords, slice_spacing=slice_spacing, interpolation="nearest"
        )
        n_slices = len(slices)

        if match_coordinates:
            tolerance = self._slice_match_tolerance(slice_spacing)
            matched_indices = self._find_matching_axes(actual_coords, tolerance)

            matched_slice_indices = {idx for _, idx in matched_indices}
            unmatched_slices = [
                (idx, actual_coords[idx])
                for idx in range(n_slices)
                if idx not in matched_slice_indices
            ]
            if unmatched_slices:
                self._warn_unmatched(unmatched_slices)
            plot_indices = matched_indices
        else:
            x_range = None
            y_range = None
            if dim_col in mask.coords and dim_row in mask.coords:
                x_vals_all = mask.coords[dim_col].values.astype(float)
                y_vals_all = mask.coords[dim_row].values.astype(float)
                x_range = float(np.max(x_vals_all) - np.min(x_vals_all))
                y_range = float(np.max(y_vals_all) - np.min(y_vals_all))
            self._ensure_figure(n_slices, x_range=x_range, y_range=y_range)

            if self._user_provided_axes:
                assert self.axes is not None
                if n_slices != self.axes.size:
                    raise ValueError(
                        f"Number of slices ({n_slices}) must match number of axes "
                        f"({self.axes.size}). Got {n_slices} slice_coords but axes has "
                        f"shape {self.axes.shape}."
                    )

            plot_indices = self._init_sequential_layout(actual_coords)

        if self.axes is None:
            raise RuntimeError("No axes available")

        axes_flat = self.axes.ravel()
        title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)

        for axis_idx, slice_idx in plot_indices:
            ax = axes_flat[axis_idx]
            slice_da = slices[slice_idx]
            slice_data = slice_da.values

            # Same geometry `add_volume` draws the underlying pcolormesh with, so
            # contour vertices land exactly on the pixels they outline instead of a
            # separately reconstructed coordinate lookup.
            x_edges, y_edges, x_centers, y_centers = _slice_edges_and_centers(
                slice_da, dim_row, dim_col
            )

            for label in unique_labels:
                binary_mask = (slice_data == label).astype(np.uint8)
                if not binary_mask.any():
                    continue

                padded = np.pad(binary_mask, 1, mode="constant")
                contours = find_contours(padded, level=0.5)
                contours = [c - 1 for c in contours]

                color = color_map.get(label, "white")

                for contour in contours:
                    if len(contour) < 2:
                        continue

                    # Map pixel indices to display coordinates. Contours are at
                    # pixel boundaries, so we interpolate between coordinate
                    # centers to get edge positions.
                    # contour[:, 0] is row (y) index, contour[:, 1] is col (x) index
                    x_idx = contour[:, 1]
                    y_idx = contour[:, 0]
                    x_world = np.interp(x_idx, np.arange(len(x_centers)), x_centers)
                    y_world = np.interp(y_idx, np.arange(len(y_centers)), y_centers)
                    ax.plot(
                        x_world,
                        y_world,
                        color=color,
                        linewidth=linewidths,
                        linestyle=linestyles,
                        **kwargs,
                    )

            if resolved_roi_labels and self.figure is not None:
                self._attach_or_update_hover_manager(resolved_roi_labels)
                self._hover_manager.register_data_to_axis(
                    ax,
                    x_coords=np.asarray(x_centers, dtype=float),
                    y_coords=np.asarray(y_centers, dtype=float),
                    data_2d=np.asarray(slice_data),
                    role="labels",
                    name=str(mask.name) if mask.name is not None else "label",
                )

            if not match_coordinates:
                ax.set_aspect("equal")

                # Compute limits from the same edges pcolormesh would use, not from
                # auto-scaled matplotlib limits, which may include padding.
                xlim = (float(x_edges.min()), float(x_edges.max()))
                ylim = (float(y_edges.min()), float(y_edges.max()))
                self._set_ax_lims(ax, xlim, ylim)
                self._style_ax(ax)
                ax.set_xlabel(
                    _build_axis_label(slice_da, dim_col),
                    color=self._text_color,
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    _build_axis_label(slice_da, dim_row),
                    color=self._text_color,
                    fontsize=label_fontsize,
                )
                ax.set_title(
                    self._build_slice_title(slice_da, actual_coords[slice_idx]),
                    color=self._text_color,
                    fontsize=title_fontsize,
                )
                ax.tick_params(labelsize=tick_fontsize)

        return self

    def savefig(self, fname: str, **kwargs) -> None:
        """Save the figure to a file.

        Parameters
        ----------
        fname : str
            Path to save the figure. Extension determines format (e.g., `.png`, `.pdf`).
        **kwargs
            Additional arguments passed to
            [`matplotlib.figure.Figure.savefig`][matplotlib.figure.Figure.savefig].

        Raises
        ------
        RuntimeError
            If called before any data has been plotted.
        """
        if self.figure is None:
            raise RuntimeError("No figure to save.")
        self.figure.savefig(fname, **kwargs)

    def show(self) -> None:
        """Display the figure.

        Raises
        ------
        RuntimeError
            If called before any data has been plotted.
        """
        if self.figure is None:
            raise RuntimeError("No figure to show.")
        self.figure.show()

    def close(self) -> None:
        """Close the figure and release resources."""
        import matplotlib.pyplot as plt

        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
            self._coord_to_axis.clear()
            self._axis_xlims.clear()
            self._axis_ylims.clear()
            self._hover_manager.clear()


def plot_contours(
    mask: xr.DataArray,
    *,
    colors: dict[int | str, str] | str | None = None,
    linewidths: float = 1.5,
    linestyles: str = "solid",
    slice_mode: str = "z",
    slice_coords: list[Hashable] | None = None,
    transpose: bool = False,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    roi_labels: dict[int, str] | None = None,
    **kwargs,
) -> VolumePlotter:
    """Plot mask contours as a grid of 2D slice panels.

    Displays contour lines for each labeled region in `mask` across a grid of subplots.
    Each panel shows the contours for one slice along `slice_mode`, drawn in world
    coordinates when available.

    Parameters
    ----------
    mask : xarray.DataArray
        Integer label map in one of two formats:

        - **Flat label map**: Spatial dims only, e.g. `(k, j, i)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `masks` dimension followed by
          spatial dims, e.g. `(masks, k, j, i)`. Each layer has values in `{0,
          region_id}` and regions may overlap. The `regions` coordinate of the
          output holds the `masks` coordinate values (e.g., region label).

    colors : dict[int | str, str] or str, optional
        Color specification for contour lines. A `dict` maps each label (integer index
        or region acronym string) to a color; a `str` applies one color to all regions.
        If not provided, colors are derived from `attrs["cmap"]` and `attrs["norm"]`
        when present, otherwise from the `tab10`/`tab20` colormap.
    linewidths : float, default: 1.5
        Width of contour lines in points.
    linestyles : str, default: "solid"
        Line style for contour lines (e.g. `"solid"`, `"dashed"`).
    slice_mode : str, default: "z"
        World dimension (`"z"`, `"y"`, `"x"`) or extra non-voxel dimension to
        slice. Native voxel dimensions (`"k"`, `"j"`, `"i"`) are not valid slice
        modes. After slicing, each panel must be 2D.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values along `slice_mode` are used.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    fontsize : float, optional
        Base font size for text elements. Subplot titles use `fontsize` directly; axis
        labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If not provided,
        uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. A single
        `Axes` is wrapped automatically. If not provided, new axes are created inside
        `figure`.
    roi_labels : dict[int, str], optional
        Mapping from integer label to display name. When provided (or when
        `mask.attrs["roi_labels"]` is populated), hovering the cursor over a
        voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar.
    **kwargs
        Additional keyword arguments passed to
        [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot].

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If `slice_mode` is not a dimension of `mask`.
    ValueError
        If `mask` is not 3D.

    Notes
    -----
    Contours are computed with `skimage.measure.find_contours` on a binary mask for each
    label, then mapped to world coordinates via linear interpolation between
    coordinate centers. Each panel has `aspect="equal"` so that 1 unit in x matches 1
    unit in y.

    The returned [`VolumePlotter`][confusius.plotting.VolumePlotter] stores the
    coordinate-to-axis mapping, so you can overlay a volume afterwards with
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume].

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_contours
    >>> mask = xr.open_zarr("output.zarr")["roi_mask"]
    >>> plotter = plot_contours(mask, slice_mode="z")

    >>> # Custom colors per label.
    >>> plotter = plot_contours(mask, slice_mode="z", colors={1: "red", 2: "cyan"})

    >>> # Overlay contours on an existing volume plot.
    >>> from confusius.plotting import plot_volume
    >>> volume = xr.open_zarr("output.zarr")["power_doppler"]
    >>> plotter = plot_volume(volume, slice_mode="z")
    >>> plotter.add_contours(mask, colors="yellow")
    """
    plotter = VolumePlotter(
        slice_mode=slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
        transpose=transpose,
    )

    return plotter.add_contours(
        mask,
        colors=colors,
        linewidths=linewidths,
        linestyles=linestyles,
        match_coordinates=False,
        slice_coords=slice_coords,
        fontsize=fontsize,
        roi_labels=roi_labels,
        **kwargs,
    )


def plot_volume(
    data: xr.DataArray,
    *,
    slice_coords: list[Hashable] | None = None,
    slice_mode: str | None = None,
    transpose: bool = False,
    cmap: "str | Colormap | None" = None,
    norm: "Normalize | None" = None,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    threshold_mode: Literal["lower", "upper"] = "lower",
    alpha: "float | xr.DataArray | None" = None,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    cbar_kwargs: "dict[str, Any] | None" = None,
    roi_labels: dict[int, str] | None = None,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
    resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    resample_fill_value: float | None = None,
) -> VolumePlotter:
    """Plot 2D slices of a volume using matplotlib.

    Displays a series of 2D slices extracted along `slice_mode` as a grid of subplots.
    Each slice is rendered using world coordinates for axis ticks when available.
    If `slice_mode` is not provided and `data` is planar, the singleton world
    dimension is used; otherwise the default is `"z"`.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array. Unitary non-world dimensions are squeezed before
        processing; singleton world display axes are preserved. After that, data
        must be 3D. Complex-valued data is converted to magnitude before display.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values along `slice_mode` are used.
    slice_mode : str, optional
        World dimension (`"z"`, `"y"`, `"x"`) or extra non-voxel dimension to
        slice. Native voxel dimensions (`"k"`, `"j"`, `"i"`) are not valid slice
        modes. If not provided, planar data is sliced along its singleton world
        dimension and full 3D data is sliced along `"z"`. After slicing, each panel
        must be 2D.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap. When not provided, falls back to `data.attrs["cmap"]` if
        present, otherwise `"gray"`.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance (e.g. `BoundaryNorm` for integer label maps such
        as atlas annotations). When not provided, falls back to
        `data.attrs["norm"]` if present. When a norm is active, `vmin` and
        `vmax` are ignored.
    vmin : float, optional
        Lower bound of the colormap. Defaults to the 2nd percentile. Ignored
        when a norm is active.
    vmax : float, optional
        Upper bound of the colormap. Defaults to the 98th percentile. Ignored
        when a norm is active.
    threshold : float, optional
        Threshold applied to `|data|`. See `threshold_mode` for the masking
        direction. If not provided, no thresholding is applied.
    threshold_mode : {"lower", "upper"}, default: "lower"
        Controls how `threshold` is applied:

        - `"lower"`: set pixels where `|data| < threshold` to NaN.
        - `"upper"`: set pixels where `|data| > threshold` to NaN.

    alpha : float or xarray.DataArray, optional
        Opacity of the image: a single scalar value, or a 3D DataArray sharing
        `data`'s dims, shape, and coordinates (for independent per-slice,
        per-voxel opacity). A per-voxel opacity must be a DataArray, not a bare
        array, so it can be validated and aligned against `data`. If not provided,
        the colormap's own alpha channel is respected.
    show_colorbar : bool, default: True
        Whether to add a shared colorbar to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    cbar_kwargs : dict, optional
        Additional keyword arguments forwarded to
        [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
        (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the colorbar
        when it spans a multi-panel grid, since the defaults are sized for a single
        axes.
    roi_labels : dict[int, str], optional
        Mapping from integer label to display name. When provided (or when
        `data.attrs["roi_labels"]` is populated), hovering the cursor over a
        voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations (spines, ticks, labels). When `False`,
        overrides `show_axis_labels` and `show_axis_ticks`.
    fontsize : float, optional
        Base font size for all text elements. Subplot titles use `fontsize` directly;
        axis labels and the colorbar label use `0.9 * fontsize`; tick labels use `0.85 *
        fontsize`. If not provided, uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. Must
        contain exactly as many elements as there are slices. A single `Axes` is
        wrapped automatically and limits the plot to one slice. If not provided, new
        axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    roi_labels : dict[int, str], optional
        Mapping from integer label to display name. When provided (or when
        `data.attrs["roi_labels"]` is populated), hovering the cursor over a
        voxel shows `<data_name>=<id> (<roi_name>)` in the matplotlib status bar.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `data` onto an axis-aligned world grid for display.
    resample_fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after resampling
        oblique data. If not provided, defaults to `data`'s own minimum value.

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If `slice_mode` is not a dimension of `data`.
    ValueError
        If `data` is not 3D after squeezing unitary dimensions.
    ValueError
        If `axes` is provided but does not contain enough elements for all slices.

    Notes
    -----
    Rendering is done with [`pcolormesh`][matplotlib.pyplot.pcolormesh], which accepts
    coordinate arrays directly and therefore handles non-uniform coordinate spacing
    correctly. Because each panel is drawn in world coordinate space, multiple calls
    with different `axes` elements will overlay correctly as long as the displayed
    dimensions are the same.

    The two dimensions that remain after slicing define the panel axes: the
    first remaining dimension maps to the vertical axis and the second to the
    horizontal axis. Coordinates are used directly as axis tick values; each
    axis has `aspect="equal"` so that 1 unit in x matches 1 unit in y.

    NaN and Inf values (including those introduced by `threshold`) are rendered
    transparently via a masked array.

    When the figure is created internally, `layout="constrained"` is used so
    that subplot titles, axis labels, tick labels, and the colorbar are spaced
    automatically without overlapping. When an external `figure` or `axes`
    is provided, layout management is left to the caller.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_volume
    >>> data = xr.open_zarr("output.zarr")["power_doppler"]
    >>> plotter = plot_volume(data, slice_mode="z")

    >>> # Select specific z slices.
    >>> plotter = plot_volume(data, slice_coords=[0.0, 1.5, 3.0], slice_mode="z")

    >>> # Threshold noise and label the colorbar.
    >>> plotter = plot_volume(
    ...     data,
    ...     slice_mode="z",
    ...     threshold=0.5,
    ...     threshold_mode="lower",
    ...     cbar_label="Power (dB)",
    ... )
    """
    resolved_slice_mode = _resolve_default_slice_mode(data, slice_mode)
    plotter = VolumePlotter(
        slice_mode=resolved_slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
        resample_interpolation=resample_interpolation,
        resample_fill_value=resample_fill_value,
        transpose=transpose,
    )

    return plotter.add_volume(
        data=data,
        slice_coords=slice_coords,
        match_coordinates=False,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
        threshold_mode=threshold_mode,
        alpha=alpha,
        show_colorbar=show_colorbar,
        cbar_label=cbar_label,
        cbar_kwargs=cbar_kwargs,
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        fontsize=fontsize,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
        roi_labels=roi_labels,
    )


def plot_composite(
    data1: xr.DataArray,
    data2: xr.DataArray,
    *,
    resample: bool = True,
    resample_kwargs: "dict[str, Any] | None" = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    normalize_strategy: Literal["per_volume", "per_slice", "shared"] = "per_volume",
    slice_coords: Sequence[Hashable] | None = None,
    slice_mode: str | None = None,
    transpose: bool = False,
    alpha: "float | npt.NDArray[np.floating] | None" = None,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
    resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    resample_fill_value: float | None = None,
) -> VolumePlotter:
    """Plot a red/cyan composite of two volumes as a grid of 2D slice panels.

    Each slice is rendered as an RGB image where `data1` drives the red channel
    and `data2` drives the green and blue channels (cyan), making overlap
    visible as desaturated grey. This is the same visual encoding used by the
    live registration progress preview.
    If `slice_mode` is not provided and `data1` is planar, the singleton world
    dimension is used; otherwise the default is `"z"`.

    Parameters
    ----------
    data1 : xarray.DataArray
        First volume, plotted in red. 3D volume data. Unitary dimensions (except
        `slice_mode`) are squeezed before processing. Complex-valued inputs are
        converted to magnitude (`abs(data)`) with a warning.
    data2 : xarray.DataArray
        Second volume, plotted in cyan. Must have the same dimensionality as `data1`
        after squeezing; when `resample=True` it is resampled onto `data1`'s grid before
        plotting, so its native shape and coordinates may differ.
    resample : bool, default: True
        Whether to resample `data2` onto `data1`'s grid using an identity transform
        before blending. When `False`, the two arrays must already share dimensions and
        shape, and their coordinates must match within `rtol`/`atol`; once validated,
        `data2`'s coordinates are replaced with `data1`'s so the two volumes share an
        exact coordinate frame downstream.
    resample_kwargs : dict, optional
        Extra keyword arguments forwarded to
        [`resample_like`][confusius.registration.resample_like] when `resample=True`.
        Ignored when `resample=False`.
    rtol : float, default: 1e-5
        Relative tolerance used to validate that `data1` and `data2` share coordinates
        when `resample=False`. Widen to accept acquisitions on slightly offset grids
        known to be equivalent. Ignored when `resample=True`.
    atol : float, default: 1e-8
        Absolute tolerance used to validate that `data1` and `data2` share coordinates
        when `resample=False`. Ignored when `resample=True`.
    normalize_strategy : {"per_volume", "per_slice", "shared"}, default: "per_volume"
        Intensity normalisation strategy.

        - `"per_volume"`: rescale each input to `[0, 1]` independently over its full volume.
          Preserves slice-to-slice contrast within each array but loses the
          absolute-intensity relationship between `data1` and `data2`.
        - `"per_slice"`: rescale each 2D slice independently. Maximises contrast on dim
          slices at the cost of cross-slice comparability.
        - `"shared"`: rescale both volumes together using a single shared
          `[min(data1.min(), data2.min()), max(data1.max(), data2.max())]` range.
          Preserves the absolute-intensity relationship between the two inputs.
    slice_coords : Sequence[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        from `data1` are used.
    slice_mode : str, optional
        World dimension (`"z"`, `"y"`, `"x"`) or extra non-voxel dimension to
        slice. Native voxel dimensions (`"k"`, `"j"`, `"i"`) are not valid slice
        modes. If not provided, planar `data1` is sliced along its singleton world
        dimension and full 3D data is sliced along `"z"`. After slicing, each
        panel must be 2D.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    alpha : float or numpy.ndarray, optional
        Opacity of the composite image, either a single value or a per-voxel array
        matching the shape of the displayed slices. If not provided, the image is
        fully opaque.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations. When `False`, overrides `show_axis_labels`
        and `show_axis_ticks`.
    fontsize : float, optional
        Base font size for all text elements. Subplot titles use `fontsize` directly;
        axis labels use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If not
        provided, uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color string
        (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula (white
        on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. Must
        contain exactly as many elements as there are slices. A single `Axes` is wrapped
        automatically. If not provided, new axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `data1`/`data2` onto an axis-aligned world grid for display.
        Distinct from `resample_kwargs`, which controls resampling `data2` onto
        `data1`'s grid for compositing.
    resample_fill_value : float, optional
        Value assigned to voxels outside `data1`/`data2`'s field of view after
        display resampling. If not provided, defaults to each array's own minimum
        value.

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If either input has a `time` dimension, is not 2D or 3D, lacks `slice_mode` as a
        dimension, or (when `resample=False`) the two arrays do not share dims, shape,
        and coordinates within `rtol`/`atol`.

    Notes
    -----
    Rendering uses [`pcolormesh`][matplotlib.axes.Axes.pcolormesh] with an RGB `C`
    array, so panels share their cell geometry with
    [`plot_volume`][confusius.plotting.plot_volume] /
    [`plot_contours`][confusius.plotting.plot_contours] and overlay correctly.
    Colormaps, colorbars, intensity thresholds, and hover tooltips are not supported on
    composite axes — use `plot_volume` for those.

    The returned [`VolumePlotter`][confusius.plotting.VolumePlotter] stores the
    coordinate-to-axis mapping, so you can overlay further volumes or contours with
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume] or
    [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours].

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_composite
    >>> fixed = xr.open_zarr("fixed.zarr")["power_doppler"]
    >>> moving = xr.open_zarr("moving.zarr")["power_doppler"]
    >>> plotter = plot_composite(fixed, moving, slice_mode="z")

    >>> # Skip resampling when the two volumes are already aligned.
    >>> plotter = plot_composite(fixed, registered_moving, resample=False)

    >>> # Maximise contrast on dim slices.
    >>> plotter = plot_composite(fixed, moving, normalize_strategy="per_slice")
    """
    resolved_slice_mode = _resolve_default_slice_mode(data1, slice_mode)
    plotter = VolumePlotter(
        slice_mode=resolved_slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
        resample_interpolation=resample_interpolation,
        resample_fill_value=resample_fill_value,
        transpose=transpose,
    )

    return plotter.add_composite(
        data1,
        data2,
        resample=resample,
        resample_kwargs=resample_kwargs,
        rtol=rtol,
        atol=atol,
        normalize_strategy=normalize_strategy,
        slice_coords=slice_coords,
        match_coordinates=False,
        alpha=alpha,
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        fontsize=fontsize,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
    )


def plot_stat_map(
    stat_map: xr.DataArray,
    *,
    bg_volume: xr.DataArray | None = None,
    slice_coords: list[Hashable] | None = None,
    slice_mode: str = "z",
    transpose: bool = False,
    bg_kwargs: "dict[str, Any] | None" = None,
    cmap: "str | Colormap | None" = None,
    norm: "Normalize | None" = None,
    vmin: float | None = None,
    vmax: float | None = None,
    auto_range: bool = True,
    alpha: "float | xr.DataArray | None" = None,
    threshold: float | None = None,
    threshold_mode: Literal["lower", "upper"] = "lower",
    show_colorbar: bool = True,
    cbar_label: str | None = None,
    cbar_kwargs: "dict[str, Any] | None" = None,
    show_titles: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
    show_axes: bool = True,
    fontsize: float | None = None,
    yincrease: bool = False,
    xincrease: bool = True,
    bg_color: str = "black",
    fg_color: str | None = None,
    figure: "Figure | None" = None,
    axes: "npt.NDArray[Any] | Axes | None" = None,
    nrows: int | None = None,
    ncols: int | None = None,
    dpi: int | None = None,
    resample_interpolation: Literal["linear", "nearest", "bspline"] = "linear",
    resample_fill_value: float | None = None,
) -> VolumePlotter:
    """Plot a statistical map, optionally over a background volume.

    Performs the recurring pattern of [`plot_volume`][confusius.plotting.plot_volume] to
    show a background anatomical volume +
    [`VolumePlotter.add_stat_map`][confusius.plotting.VolumePlotter.add_stat_map] to
    overlay a statistical map, with the colormap and range picked automatically based
    on whether the statistic is diverging (has both positive and negative values) or
    one-signed.

    Parameters
    ----------
    stat_map : xarray.DataArray
        Statistical map to plot. 3D volume data. Unitary dimensions (except
        `slice_mode`) are squeezed before processing.
    bg_volume : xarray.DataArray, optional
        Background anatomical volume, plotted underneath `stat_map`. When `alpha` is
        not provided, `stat_map` fully covers `bg_volume` wherever it has a value;
        `bg_volume` only shows through where `stat_map` is masked out by `threshold`.
        Lower `alpha` to blend the two layers instead. Must share `slice_mode` and,
        after squeezing, the same display dimensions as `stat_map`. If not provided,
        `stat_map` is plotted on its own.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values from `bg_volume` (or `stat_map` when `bg_volume` is not
        provided) along `slice_mode` are used.
    slice_mode : str, default: "z"
        World dimension (`"z"`, `"y"`, `"x"`) or extra non-voxel dimension to
        slice. Native voxel dimensions (`"k"`, `"j"`, `"i"`) are not valid slice
        modes. After slicing, each panel must be 2D.
    transpose : bool, default: False
        Whether to swap the row/column display dims of each slice panel.
    bg_kwargs : dict, optional
        Additional keyword arguments forwarded to
        [`plot_volume`][confusius.plotting.plot_volume] for the background layer
        (e.g. `cmap`, `vmin`, `vmax`, `norm`, `alpha`, `roi_labels`). Ignored when
        `bg_volume` is not provided. Layout and text styling (`slice_coords`,
        `slice_mode`, `show_titles`, `fontsize`, etc.) are controlled by this
        function's own parameters instead, so that both layers share consistent
        styling.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for `stat_map`. If not provided, the default depends on
        `auto_range` and the sign of `stat_map` (see below); an explicit `cmap` is
        always used as-is regardless of `auto_range`.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance (e.g. `TwoSlopeNorm`, `BoundaryNorm`, `LogNorm`) for
        cases `vmin`/`vmax`/`auto_range` can't express. When provided, `vmin`,
        `vmax`, and `auto_range`'s range computation are bypassed entirely; `cmap`
        still follows the usual rules above.
    vmin : float, optional
        Lower bound of the colormap. If not provided, defaults to the minimum value
        of `stat_map`, computed over the full array rather than just the displayed
        slices. Ignored when `norm` is provided, when `auto_range=True` and
        `stat_map` has only non-negative values, or when `auto_range=True`,
        `stat_map` spans both signs and `vmax` is given on its own (see
        `auto_range`).
    vmax : float, optional
        Upper bound of the colormap. If not provided, defaults to the maximum value
        of `stat_map`, computed over the full array rather than just the displayed
        slices. Ignored when `norm` is provided, when `auto_range=True` and
        `stat_map` has only non-positive values, or when `auto_range=True`,
        `stat_map` spans both signs and `vmin` is given on its own (see
        `auto_range`).
    auto_range : bool, default: True
        Whether to pick the colormap range and default colormap automatically based
        on the sign of `stat_map`:

        - Both positive and negative values: diverging, symmetric `[-m, m]` range
          where `m = max(|vmin|, |vmax|)` over the bounds actually provided,
          falling back to the largest magnitude in `stat_map` when neither is
          given, with `cmap` defaulting to `"coolwarm"` — the right choice for
          diverging statistics where the sign is meaningful (e.g. t-statistics,
          correlation coefficients, PCA/ICA component maps).
        - Only non-negative values: sequential `[0, vmax]` range, with `cmap`
          defaulting to `"viridis"` — the right choice for non-diverging
          statistics where only magnitude matters (e.g. R², F-statistics).
        - Only non-positive values: sequential `[vmin, 0]` range, with `cmap`
          defaulting to `"viridis_r"` (reversed, so that values near zero map to
          the same end of the colormap in both the non-negative and
          non-positive cases).

        Set to `False` to use the resolved `vmin`/`vmax` directly with no
        zero-anchoring (`cmap` then defaults to `"coolwarm"` regardless of sign).
    alpha : float or xarray.DataArray, optional
        Opacity of the `stat_map` overlay: a single scalar value, or a 3D DataArray
        sharing `stat_map`'s dims, shape, and coordinates (for independent per-slice,
        per-voxel opacity, e.g. to fade out low-magnitude voxels instead of masking them
        out with `threshold`). A per-voxel opacity must be a DataArray, not a bare
        array, so it can be validated and aligned against `stat_map`; note it is
        validated against `stat_map`, not `bg_volume`. If not provided, the colormap's
        own alpha channel is respected.
    threshold : float, optional
        Threshold applied to `|stat_map|`. See `threshold_mode` for the masking
        direction. If not provided, no thresholding is applied.
    threshold_mode : {"lower", "upper"}, default: "lower"
        Controls how `threshold` is applied:

        - `"lower"`: set pixels where `|stat_map| < threshold` to NaN.
        - `"upper"`: set pixels where `|stat_map| > threshold` to NaN.

    show_colorbar : bool, default: True
        Whether to add a shared colorbar for `stat_map` to the figure.
    cbar_label : str, optional
        Label for the colorbar.
    cbar_kwargs : dict, optional
        Additional keyword arguments forwarded to
        [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
        (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the colorbar
        when it spans a multi-panel grid, since the defaults are sized for a single
        axes.
    show_titles : bool, default: True
        Whether to display subplot titles showing the slice coordinate.
    show_axis_labels : bool, default: True
        Whether to display axis labels (with units when available).
    show_axis_ticks : bool, default: True
        Whether to display axis tick labels.
    show_axes : bool, default: True
        Whether to show all axis decorations (spines, ticks, labels). When `False`,
        overrides `show_axis_labels` and `show_axis_ticks`.
    fontsize : float, optional
        Base font size for all text elements. Subplot titles use `fontsize` directly;
        axis labels and the colorbar label use `0.9 * fontsize`; tick labels use `0.85 *
        fontsize`. If not provided, uses the active Matplotlib defaults.
    yincrease : bool, default: False
        Whether the y-axis increases upward (`True`) or downward (`False`).
    xincrease : bool, default: True
        Whether the x-axis increases to the right (`True`) or left (`False`).
    bg_color : str, default: "black"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    figure : matplotlib.figure.Figure, optional
        Existing figure to draw into. If not provided, a new figure is created.
    axes : numpy.ndarray or matplotlib.axes.Axes, optional
        Existing axes to draw into: either a single
        [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them. Must
        contain exactly as many elements as there are slices. A single `Axes` is
        wrapped automatically and limits the plot to one slice. If not provided, new
        axes are created inside `figure`.
    nrows : int, optional
        Number of rows in the subplot grid. If not provided, computed automatically.
    ncols : int, optional
        Number of columns in the subplot grid. If not provided, computed automatically.
    dpi : int, optional
        Figure resolution in dots per inch. Ignored when `figure` is provided.
    resample_interpolation : {"linear", "nearest", "bspline"}, default: "linear"
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `stat_map`/`bg_volume` onto an axis-aligned world grid for
        display. Applied to both, since they share one `VolumePlotter`.
    resample_fill_value : float, optional
        Value assigned to voxels outside `stat_map`/`bg_volume`'s field of view
        after display resampling. If not provided, defaults to each array's own
        minimum value.

    Returns
    -------
    VolumePlotter
        Object managing the figure, axes, and coordinate mapping for overlays.

    Raises
    ------
    ValueError
        If `slice_mode` is not a dimension of `stat_map` or `bg_volume`.
    ValueError
        If `stat_map` or `bg_volume` is not 3D after squeezing unitary dimensions.

    Notes
    -----
    When `bg_volume` is provided, this is equivalent to calling
    `plot_volume(bg_volume, show_colorbar=False, ...)` followed by
    `plotter.add_stat_map(stat_map, ...)`. Use
    [`VolumePlotter.add_stat_map`][confusius.plotting.VolumePlotter.add_stat_map]
    directly to overlay a statistical map onto an existing plot, or
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume] for full
    manual control over the colormap and range.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_stat_map
    >>> anatomical = xr.open_zarr("output.zarr")["power_doppler"]
    >>> t_map = xr.open_zarr("output.zarr")["t_stat"]
    >>> plotter = plot_stat_map(t_map, bg_volume=anatomical, slice_mode="z")

    >>> # Suppress subthreshold voxels and cap the colormap range explicitly.
    >>> plotter = plot_stat_map(
    ...     t_map,
    ...     bg_volume=anatomical,
    ...     threshold=3.0,
    ...     vmax=6.0,
    ...     cbar_label="t-statistic",
    ... )

    >>> # Blend the overlay with the background instead of fully covering it.
    >>> plotter = plot_stat_map(t_map, bg_volume=anatomical, alpha=0.6)

    >>> # Non-diverging statistic (e.g. R²): sequential range and colormap picked
    >>> # automatically since r2_map has only non-negative values.
    >>> r2_map = xr.open_zarr("output.zarr")["r2"]
    >>> plotter = plot_stat_map(r2_map, bg_volume=anatomical)

    >>> # No background: plot the statistical map on its own.
    >>> plotter = plot_stat_map(t_map, slice_mode="z")
    """
    if bg_volume is not None:
        plotter = plot_volume(
            bg_volume,
            slice_coords=slice_coords,
            slice_mode=slice_mode,
            show_colorbar=False,
            show_titles=show_titles,
            show_axis_labels=show_axis_labels,
            show_axis_ticks=show_axis_ticks,
            show_axes=show_axes,
            fontsize=fontsize,
            yincrease=yincrease,
            xincrease=xincrease,
            bg_color=bg_color,
            fg_color=fg_color,
            figure=figure,
            axes=axes,
            nrows=nrows,
            ncols=ncols,
            dpi=dpi,
            resample_interpolation=resample_interpolation,
            resample_fill_value=resample_fill_value,
            transpose=transpose,
            **(bg_kwargs or {}),
        )
    else:
        plotter = VolumePlotter(
            slice_mode=slice_mode,
            figure=figure,
            axes=axes,
            bg_color=bg_color,
            fg_color=fg_color,
            yincrease=yincrease,
            xincrease=xincrease,
            resample_interpolation=resample_interpolation,
            resample_fill_value=resample_fill_value,
            transpose=transpose,
        )

    return plotter.add_stat_map(
        stat_map,
        slice_coords=slice_coords,
        match_coordinates=bg_volume is not None,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        auto_range=auto_range,
        threshold=threshold,
        threshold_mode=threshold_mode,
        alpha=alpha,
        show_colorbar=show_colorbar,
        cbar_label=cbar_label,
        cbar_kwargs=cbar_kwargs,
        show_titles=show_titles,
        show_axis_labels=show_axis_labels,
        show_axis_ticks=show_axis_ticks,
        show_axes=show_axes,
        fontsize=fontsize,
        nrows=nrows,
        ncols=ncols,
        dpi=dpi,
    )


def _prepare_carpet_data(
    data: xr.DataArray,
    mask: xr.DataArray | None = None,
    detrend_order: int | None = None,
    standardize: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    decimation_threshold: int | None = 800,
) -> dict:
    """Prepare carpet plot data, separating expensive computation from drawing.

    Intended to run in a background thread; the result is passed to `plot_carpet` via
    its `_precomputed` keyword.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a `"time"` dimension and coordinate: a VoxelData
        array, or an already-extracted signals array.
    mask : xarray.DataArray, optional
        Boolean mask to select elements. Defaults to all non-zero elements.
    detrend_order : int, optional
        Polynomial order for detrending. See `plot_carpet`.
    standardize : bool, default: True
        Whether to z-score each voxel signals.
    vmin : float, optional
        Lower colormap bound. Computed from data when `None`.
    vmax : float, optional
        Upper colormap bound. Computed from data when `None`.
    decimation_threshold : int or None, default: 800
        Downsample time axis when the number of frames exceeds this value.

    Returns
    -------
    dict
        Keys: `signals` (DataArray with shape `(time, space)`), `vmin` (float),
        `vmax` (float), `xlabel` (str), `time_coord` (DataArray | None).
    """
    if np.iscomplexobj(data):
        data = xr.ufuncs.abs(data)

    validate_time_series(data, "plot_carpet", require_unchunked_time=False)

    n_timepoints = data.sizes["time"]

    non_zero = (data != 0).any(dim="time")
    if mask is None:
        mask = non_zero
    else:
        mask = mask & non_zero

    signals = select_masked_features(data, mask)

    # Carpet plots don't need spatial coordinates, and multi-index coordinates will make
    # plotting fail.
    space_coords = [c for c in signals.coords if "space" in signals.coords[c].dims]
    signals = signals.drop_vars(space_coords).assign_coords(
        space=np.arange(signals.sizes["space"])
    )

    signals = clean(
        signals,
        detrend_order=detrend_order,
        standardize_method="zscore" if standardize else None,
    )

    if vmin is None or vmax is None:
        std_val = float(signals.std(axis=0).mean().values)
        default_vmin = float(signals.mean().values - (2 * std_val))
        default_vmax = float(signals.mean().values + (2 * std_val))
        vmin = vmin or default_vmin
        vmax = vmax or default_vmax

    if decimation_threshold is not None and n_timepoints > decimation_threshold:
        n_decimations = int(
            np.ceil(np.log2(np.ceil(n_timepoints / decimation_threshold)))
        )
        decimation_factor = 2**n_decimations
        signals = signals[::decimation_factor, :]

    return {
        "signals": signals,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "xlabel": _build_axis_label(data, "time").capitalize(),
        "time_coord": data.coords.get("time"),
    }


def _draw_carpet(
    prep: dict,
    cmap: "str | Colormap" = "gray",
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    fontsize: float | None = None,
    bg_color: str = "white",
    fg_color: str | None = None,
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Draw a carpet plot from pre-computed data.

    Low-level drawing counterpart of `_prepare_carpet_data`. Intended to run on the main
    thread after the expensive data preparation has been done in a background thread.

    Parameters
    ----------
    prep : dict
        Pre-computed dict returned by `_prepare_carpet_data`.
    cmap : str, default: `"gray"`
        Matplotlib colormap name.
    figsize : tuple[float, float], default: (10, 5)
        Figure size in inches, used only when *ax* is `None`.
    title : str, optional
        Plot title.
    fontsize : float, optional
        Base font size for text elements. Title uses `fontsize` directly; axis labels
        and colorbar label use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If
        not provided, uses the active Matplotlib defaults.
    bg_color : str, default: "white"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when `None`.

    Returns
    -------
    figure : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Figure containing the carpet plot.
    axes : matplotlib.axes.Axes
        Axes with the carpet plot.
    """
    import matplotlib.pyplot as plt

    signals = prep["signals"]
    vmin = prep["vmin"]
    vmax = prep["vmax"]
    xlabel = prep["xlabel"]

    text_color = fg_color if fg_color is not None else _auto_fg_color(bg_color)
    title_fontsize, label_fontsize, tick_fontsize = _resolve_font_sizes(fontsize)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
        figure.patch.set_facecolor(bg_color)
    else:
        figure = ax.figure

    ax.set_facecolor(bg_color)

    plotted_quadmesh = signals.T.plot(
        cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, yincrease=False
    )

    if plotted_quadmesh.colorbar is not None:
        cbar = plotted_quadmesh.colorbar
        _style_colorbar(
            cbar,
            text_color,
            tick_fontsize,
            bg_color=bg_color,
            label_fontsize=label_fontsize,
        )

    ax.grid(False)
    ax.set_yticks([])
    ax.set_ylabel("Voxels", color=text_color, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, color=text_color, fontsize=label_fontsize)
    ax.tick_params(colors=text_color, labelsize=tick_fontsize)

    if title:
        ax.set_title(title, color=text_color, fontsize=title_fontsize)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_edgecolor(text_color)
    ax.spines["left"].set_edgecolor(text_color)

    return figure, ax


def plot_carpet(
    data: xr.DataArray,
    mask: xr.DataArray | None = None,
    detrend_order: int | None = None,
    standardize: bool = True,
    cmap: "str | Colormap" = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    decimation_threshold: int | None = 800,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    fontsize: float | None = None,
    bg_color: str = "white",
    fg_color: str | None = None,
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Plot voxel intensities across time as a raster image.

    A carpet plot (also known as "grayplot" or "Power plot") displays voxel
    intensities as a 2D raster image with time on the x-axis and voxels on
    the y-axis. Each row represents one voxel's signals, typically
    standardized to z-scores.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a `time` dimension: a VoxelData array, or
        an already-extracted signals array (e.g.
        [`extract_with_labels`][confusius.extract.extract_with_labels] output).
    mask : xarray.DataArray, optional
        Boolean mask with the same non-`time` dimensions and coordinates as `data`.
        True values indicate elements to include. If not provided, all non-zero
        elements from the data are included.
    detrend_order : int, optional
        Polynomial order for detrending:

        - `0`: Remove mean (constant detrending).
        - `1`: Remove linear trend using least squares regression.
        - `2+`: Remove polynomial trend of specified order.

        If not provided, no detrending is applied.
    standardize : bool, default: True
        Whether to standardize each voxel's signals to z-scores.
    cmap : str, default: "gray"
        Matplotlib colormap name.
    vmin : float, optional
        Minimum value for colormap. If not provided, uses `mean - 2*std`.
    vmax : float, optional
        Maximum value for colormap. If not provided, uses `mean + 2*std`.
    decimation_threshold : int or None, default: 800
        If the number of timepoints exceeds this value, data is downsampled
        along the time axis to improve plotting performance. Set to `None` to
        disable downsampling.
    figsize : tuple[float, float], default: (10, 5)
        Figure size in inches `(width, height)`.
    title : str, optional
        Plot title.
    fontsize : float, optional
        Base font size for text elements. Title uses `fontsize` directly; axis labels
        and colorbar label use `0.9 * fontsize`; tick labels use `0.85 * fontsize`. If
        not provided, uses the active Matplotlib defaults.
    bg_color : str, default: "white"
        Background color for the figure and axes. Any matplotlib-compatible color
        string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
    fg_color : str, optional
        Color for text, labels, ticks, and spines. If not provided, derived
        automatically from `bg_color` using the WCAG relative luminance formula
        (white on dark backgrounds, black on light ones).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If not provided, creates new figure and axes.

    Returns
    -------
    figure : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Figure object containing the carpet plot.
    axes : matplotlib.axes.Axes
        Axes object with the carpet plot.

    Notes
    -----
    Complex-valued data is converted to magnitude before processing.

    This function was inspired by Nilearn's `nilearn.plotting.plot_carpet`.

    References
    ----------
    [^1]:
        Power, Jonathan D. “A Simple but Useful Way to Assess fMRI Scan Qualities.”
        NeuroImage, vol. 154, July 2017, pp. 150–58. DOI.org (Crossref),
        <https://doi.org/10.1016/j.neuroimage.2016.08.009>.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_carpet
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> fig, ax = plot_carpet(data)

    >>> # With linear detrending
    >>> fig, ax = plot_carpet(data, detrend_order=1)

    >>> # With mask
    >>> import numpy as np
    >>> mask = xr.DataArray(
    ...     np.abs(data.isel(time=0)) > threshold,
    ...     dims=["z", "y", "x"],
    ... )
    >>> fig, ax = plot_carpet(data, mask=mask)
    """
    prep = _prepare_carpet_data(
        data, mask, detrend_order, standardize, vmin, vmax, decimation_threshold
    )
    return _draw_carpet(
        prep,
        cmap=cmap,
        figsize=figsize,
        title=title,
        fontsize=fontsize,
        bg_color=bg_color,
        fg_color=fg_color,
        ax=ax,
    )
