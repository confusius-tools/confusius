"""Image visualization utilities for fUSI data."""

import math
import numbers
import warnings
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
import xarray as xr

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.geometry import get_voxel_affine_physical_coord_names
from confusius._utils.plotting import blend_red_cyan, scale_min_max
from confusius._utils.stack import find_stack_level
from confusius.extract import extract_with_mask
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
    sort_coords_for_plot,
)
from confusius.signal import clean
from confusius.validation import validate_matching_coordinates, validate_time_series

if TYPE_CHECKING:
    import numpy.typing as npt
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


def _has_voxel_affine_geometry(data: xr.DataArray) -> bool:
    """Return whether `data` carries voxel-affine geometry metadata."""
    return "voxel_to_physical" in data.attrs and all(
        str(dim) in {"k", "j", "i"} for dim in data.dims
    )


def _validate_voxel_affine_slice_mode(data: xr.DataArray, slice_mode: str) -> None:
    """Validate slice selection semantics for voxel-affine data.

    Parameters
    ----------
    data : xarray.DataArray
        Data being sliced for plotting.
    slice_mode : str
        Requested slice dimension.

    Raises
    ------
    ValueError
        If voxel-affine data is sliced along an unsupported dimension.
    """
    if not _has_voxel_affine_geometry(data):
        return

    valid_slice_modes = tuple(dim for dim in ("k", "j", "i") if dim in data.dims)
    valid_slice_modes += tuple(
        dim
        for dim in get_voxel_affine_physical_coord_names(data)
        if dim not in valid_slice_modes
    )
    if slice_mode not in valid_slice_modes:
        raise ValueError(
            "Voxel-affine plotting supports only native voxel-plane slicing or "
            f"physical z/y/x slicing, got slice_mode={slice_mode!r}. Supported "
            f"modes: {valid_slice_modes!r}."
        )
    if slice_mode in {"z", "y", "x"} and len(data.dims) != 3:
        raise ValueError(
            "Physical z/y/x slicing for voxel-affine plotting requires 3D data."
        )


def _voxel_affine_dim_order(slice_da: xr.DataArray) -> tuple[str, ...]:
    """Return voxel-space dimension order implied by the stored affine."""
    ndim = np.asarray(slice_da.attrs["voxel_to_physical"]).shape[0] - 1
    dims = tuple(
        dim for dim in ("k", "j", "i") if dim in slice_da.dims or dim in slice_da.coords
    )
    if len(dims) != ndim:
        raise ValueError(
            "Voxel-affine plotting could not infer voxel dimension order from "
            f"dims/coords {dims!r} and affine shape {slice_da.attrs['voxel_to_physical'].shape}."
        )
    return dims


def _project_voxel_affine_plane(
    slice_da: xr.DataArray,
    dim_row: str,
    dim_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a voxel-affine 2D slice into an in-plane orthonormal basis.

    Parameters
    ----------
    slice_da : xarray.DataArray
        Two-dimensional slice whose dimensions are voxel-space dims.
    dim_row : str
        Voxel-space dimension displayed on rows.
    dim_col : str
        Voxel-space dimension displayed on columns.

    Returns
    -------
    x_edges : (H+1, W+1) numpy.ndarray
        In-plane x coordinates of cell corners.
    y_edges : (H+1, W+1) numpy.ndarray
        In-plane y coordinates of cell corners.
    x_centers : (H, W) numpy.ndarray
        In-plane x coordinates of cell centers.
    y_centers : (H, W) numpy.ndarray
        In-plane y coordinates of cell centers.
    """
    affine = np.asarray(slice_da.attrs["voxel_to_physical"], dtype=np.float64)
    dim_order = _voxel_affine_dim_order(slice_da)
    linear = affine[:-1, :-1]

    row_vals = (
        slice_da.coords[dim_row].values.astype(float)
        if dim_row in slice_da.coords
        else np.arange(slice_da.sizes[dim_row], dtype=float)
    )
    col_vals = (
        slice_da.coords[dim_col].values.astype(float)
        if dim_col in slice_da.coords
        else np.arange(slice_da.sizes[dim_col], dtype=float)
    )
    row_edges = _centers_to_edges(row_vals)
    col_edges = _centers_to_edges(col_vals)

    col_vec = linear[:, dim_order.index(dim_col)]
    row_vec = linear[:, dim_order.index(dim_row)]
    e1 = col_vec / np.linalg.norm(col_vec)
    row_perp = row_vec - np.dot(row_vec, e1) * e1
    row_perp_norm = np.linalg.norm(row_perp)
    if np.isclose(row_perp_norm, 0.0):
        raise ValueError(
            f"Voxel-affine plotting requires non-collinear plane axes, got {dim_row!r} "
            f"and {dim_col!r}."
        )
    e2 = row_perp / row_perp_norm

    def _project(
        row_axis: np.ndarray, col_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        col_grid, row_grid = np.meshgrid(col_axis, row_axis, indexing="xy")
        voxel_components: list[np.ndarray] = []
        for dim in dim_order:
            if dim == dim_row:
                voxel_components.append(row_grid)
            elif dim == dim_col:
                voxel_components.append(col_grid)
            else:
                voxel_components.append(
                    np.full_like(
                        row_grid, float(slice_da.coords[dim].item()), dtype=float
                    )
                )
        homogeneous = np.stack(
            [*voxel_components, np.ones_like(row_grid, dtype=float)], axis=0
        ).reshape(len(dim_order) + 1, -1)
        physical = (affine @ homogeneous).reshape((affine.shape[0], *row_grid.shape))[
            :-1
        ]
        origin = physical[:, 0, 0][:, np.newaxis, np.newaxis]
        rel = physical - origin
        x = np.tensordot(e1, rel, axes=(0, 0))
        y = np.tensordot(e2, rel, axes=(0, 0))
        return x, y

    x_edges, y_edges = _project(row_edges, col_edges)
    x_centers, y_centers = _project(row_vals, col_vals)
    return x_edges, y_edges, x_centers, y_centers


def _resample_voxel_affine_to_physical_grid(
    data: xr.DataArray,
    slice_mode: str,
    reference: xr.DataArray | None = None,
) -> xr.DataArray:
    """Resample voxel-affine data onto an axis-aligned physical grid for plotting.

    Parameters
    ----------
    data : xarray.DataArray
        Three-dimensional voxel-affine DataArray.
    slice_mode : str
        Requested plotting slice dimension.
    reference : xarray.DataArray, optional
        Axis-aligned physical-grid DataArray to reuse as the resampling target.
        If not provided, a new plotting grid is synthesized from `data`'s physical
        bounds and finest spatial spacing.

    Returns
    -------
    xarray.DataArray
        Axis-aligned physical-grid DataArray when `slice_mode` is a physical
        `z`/`y`/`x` axis, otherwise the original input.
    """
    if not _has_voxel_affine_geometry(data) or slice_mode not in {"z", "y", "x"}:
        return data

    from confusius.registration import resample_like, resample_volume

    physical_dims = get_voxel_affine_physical_coord_names(data)
    if slice_mode not in physical_dims:
        return data

    if reference is not None:
        result = resample_like(
            data,
            reference,
            np.eye(len(physical_dims) + 1, dtype=np.float64),
        )
    else:
        spacing_values = [
            float(spacing)
            for dim, spacing in data.fusi.spacing.items()
            if dim != "time" and spacing is not None
        ]
        spacing = min(spacing_values)

        origin: list[float] = []
        shape: list[int] = []
        for dim in physical_dims:
            values = np.asarray(data.coords[dim].values, dtype=np.float64)
            lower = float(np.min(values))
            upper = float(np.max(values))
            origin.append(lower)
            shape.append(int(np.ceil((upper - lower) / spacing)) + 1)

        result = resample_volume(
            data,
            np.eye(len(physical_dims) + 1, dtype=np.float64),
            shape=shape,
            spacing=[spacing] * len(physical_dims),
            origin=origin,
            dims=physical_dims,
            direction=np.eye(len(physical_dims), dtype=np.float64),
        )
    result.attrs.pop("voxel_to_physical", None)
    for dim in physical_dims:
        result.coords[dim].attrs = data.coords[dim].attrs.copy()
    return result


def _slice_edges_and_centers(
    slice_da: xr.DataArray, dim_row: str, dim_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return plotting geometry for a 2D slice.

    For standard axis-aligned data, returns 1D edge/center coordinates for the plotted
    row/column dimensions. For voxel-affine data, returns 2D corner and center meshes
    obtained by projecting the physical slice into an orthonormal in-plane basis.
    """
    if _has_voxel_affine_geometry(slice_da):
        return _project_voxel_affine_plane(slice_da, dim_row, dim_col)

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
      where `m = max(|vmin|, |vmax|)` (using the resolved bounds above), with
      `cmap` defaulting to `_STAT_MAP_DIVERGING_CMAP`.
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
        abs_max = max(abs(resolved_vmin), abs(resolved_vmax))
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
    deduped: dict[float, "str | tuple[float, ...] | list[float]"] = {}
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
    """Return axis label for `dim`, including units when available."""
    if _has_voxel_affine_geometry(da):
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
    """Compare two slice coordinates, using a tolerance for numeric values."""
    if isinstance(stored_coord, numbers.Real) and isinstance(
        target_coord, numbers.Real
    ):
        return abs(float(stored_coord) - float(target_coord)) < tolerance
    return stored_coord == target_coord


def _extract_slices(
    data: xr.DataArray, slice_mode: str, slice_coords: Sequence[Hashable]
) -> tuple[list[xr.DataArray], list[Hashable]]:
    """Extract 2D slices from `data` along `slice_mode`.

    Numeric coordinates are matched by nearest-neighbour lookup; non-numeric
    coordinates (e.g. region labels) require an exact match.

    Returns the slices and their actual snapped coordinate values.
    """
    slices: list[xr.DataArray] = []
    actual_coords: list[Hashable] = []
    for coord in slice_coords:
        if slice_mode in data.coords:
            is_numeric = np.issubdtype(data.coords[slice_mode].dtype, np.number)
            slice_da = data.sel(
                {slice_mode: coord}, method="nearest" if is_numeric else None
            )
            actual_coord = slice_da.coords[slice_mode].values.item()
        else:
            if not isinstance(coord, numbers.Real):
                raise ValueError(
                    f"slice_mode '{slice_mode}' has no coordinates, so slice_coords "
                    f"must be numeric positional indices, got {coord!r}."
                )
            idx = int(round(coord))
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
        The dimension along which slices are taken (e.g., `"z"`).
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

    Attributes
    ----------
    slice_mode : str
        The dimension along which slices are taken.
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
    ):
        self.slice_mode = slice_mode
        if axes is not None and not isinstance(axes, np.ndarray):
            axes = np.asarray([[axes]])
        self.axes = axes
        self._user_provided_axes = axes is not None
        if figure is None and axes is not None:
            self.figure = axes.flat[0].figure
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

        self._hover_manager = _HoverManager()
        self._physical_grid_reference: xr.DataArray | None = None

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
            if x_range is not None and y_range is not None and y_range > 0:
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

    def _find_matching_axes(
        self, actual_coords: list[Hashable], tolerance: float = 1e-6
    ) -> list[tuple[int, int]]:
        """Find axis indices matching the target coordinates.

        Uses the coordinate-to-axis mapping stored when the figure was first created,
        avoiding any dependency on axis titles.

        Parameters
        ----------
        actual_coords : list[collections.abc.Hashable]
            The actual coordinate values of the slices being plotted.
        tolerance : float, default: 1e-6
            Tolerance for matching numeric coordinates, accounting for floating-point
            precision. Non-numeric coordinates (e.g. region labels) are matched by
            equality.

        Returns
        -------
        list[tuple[int, int]]
            List of `(axis_flat_idx, slice_idx)` tuples for matched coordinates.
        """
        matched = []
        for slice_idx, target_coord in enumerate(actual_coords):
            for stored_coord, axis_idx in self._coord_to_axis.items():
                if _coords_match(stored_coord, target_coord, tolerance):
                    matched.append((axis_idx, slice_idx))
                    break
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

    def _prepare_slice_inputs(self, data: xr.DataArray, *, caller: str) -> xr.DataArray:
        """Coerce complex, squeeze, validate `slice_mode`/3D, and sort display coords."""
        data = coerce_complex_to_magnitude(data, caller=caller)
        _validate_voxel_affine_slice_mode(data, self.slice_mode)
        data = _resample_voxel_affine_to_physical_grid(
            data,
            self.slice_mode,
            reference=self._physical_grid_reference,
        )
        squeeze_dims = [
            d for d in data.dims if d != self.slice_mode and data.sizes[d] == 1
        ]
        if squeeze_dims:
            data = data.squeeze(dim=squeeze_dims)
        if self.slice_mode not in data.dims:
            raise ValueError(
                f"slice_mode '{self.slice_mode}' is not a dimension of data. "
                f"Available dimensions: {list(data.dims)}."
            )
        if data.ndim != 3:
            raise ValueError(
                f"Data must be 3D, but got shape {data.shape} with dims "
                f"{list(data.dims)}."
            )
        # Only the two display dims need sorting for pcolormesh/contour geometry;
        # sorting slice_mode itself would silently reorder panels (e.g. non-monotonic
        # z, or a "region" dim built from an arbitrary list of acronyms).
        display_dims = [d for d in data.dims if d != self.slice_mode]
        return sort_coords_for_plot(data, display_dims)

    def _resolve_axes_layout(
        self,
        data: xr.DataArray,
        n_slices: int,
        actual_coords: list[Hashable],
        dim_row: str,
        dim_col: str,
        *,
        match_coordinates: bool,
        nrows: int | None,
        ncols: int | None,
        dpi: int | None,
    ) -> list[tuple[int, int]]:
        """Resolve the per-slice axis assignment, creating the figure if needed."""
        if match_coordinates:
            if self.axes is None:
                raise ValueError(
                    "Cannot match coordinates: no existing axes. Either create a "
                    "VolumePlotter with axes or use match_coordinates=False."
                )
            matched_indices = self._find_matching_axes(actual_coords)
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
            If True, match slice coordinates to the stored coordinate mapping (for
            overlays). If False, plot sequentially on all axes (requires exact axis
            count match).
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
        data = self._prepare_slice_inputs(data, caller="VolumePlotter.add_volume")
        if self.slice_mode in {"z", "y", "x"} and self._physical_grid_reference is None:
            self._physical_grid_reference = data

        display_dims = [str(d) for d in data.dims if d != self.slice_mode]
        dim_row, dim_col = display_dims[0], display_dims[1]

        if slice_coords is None:
            if self.slice_mode in data.coords:
                slice_coords = list(data.coords[self.slice_mode].values)
            else:
                slice_coords = list(range(data.sizes[self.slice_mode]))

        unthresholded_slices, actual_coords = _extract_slices(
            data, self.slice_mode, slice_coords
        )
        n_slices = len(unthresholded_slices)

        if alpha is not None and not isinstance(alpha, (int, float, xr.DataArray)):
            raise TypeError(
                "`alpha` must be a scalar or a DataArray, not "
                f"{type(alpha).__name__}; a bare array carries no coordinates, so "
                "it cannot be validated or aligned against `data`."
            )

        alpha_slices: list[np.ndarray] | None = None
        if isinstance(alpha, xr.DataArray):
            # Preprocess alpha exactly like data (squeeze unitary dims, sort coords
            # ascending). Otherwise, an alpha array created from the data could end up
            # with a shape or coordinate mismatch after slicing.
            alpha = self._prepare_slice_inputs(
                alpha, caller="VolumePlotter.add_volume (alpha)"
            )
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
            validate_matching_coordinates(
                data, alpha, left_name="data", right_name="alpha"
            )
            alpha_da_slices, _ = _extract_slices(alpha, self.slice_mode, slice_coords)
            alpha_slices = [s.values for s in alpha_da_slices]
            alpha = None

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
            if hover_x.ndim == 1 and hover_y.ndim == 1:
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
                data,
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

        data1 = self._prepare_slice_inputs(
            data1, caller="VolumePlotter.add_composite (data1)"
        )
        data2 = self._prepare_slice_inputs(
            data2, caller="VolumePlotter.add_composite (data2)"
        )

        if resample:
            from confusius.registration.resampling import resample_like

            data2_name = data2.name or "data2"
            _kw: dict[str, Any] = dict(resample_kwargs or {})
            data2 = resample_like(data2, data1, np.eye(data1.ndim + 1), **_kw)
            data2.name = data2_name
        else:
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
            if self.slice_mode in data1.coords:
                slice_coords = list(data1.coords[self.slice_mode].values)
            else:
                slice_coords = list(range(data1.sizes[self.slice_mode]))

        input_slices1, _ = _extract_slices(data1, self.slice_mode, slice_coords)
        input_slices2, _ = _extract_slices(data2, self.slice_mode, slice_coords)

        if normalize_strategy == "per_volume":
            data1 = data1.copy(data=scale_min_max(data1.values.astype(float)))
            data2 = data2.copy(data=scale_min_max(data2.values.astype(float)))
        elif normalize_strategy == "shared":
            arr1 = data1.values.astype(float)
            arr2 = data2.values.astype(float)
            lo = float(min(arr1.min(), arr2.min()))
            hi = float(max(arr1.max(), arr2.max()))
            if hi == lo:
                arr1 = np.zeros_like(arr1)
                arr2 = np.zeros_like(arr2)
            else:
                arr1 = (arr1 - lo) / (hi - lo)
                arr2 = (arr2 - lo) / (hi - lo)
            data1 = data1.copy(data=arr1)
            data2 = data2.copy(data=arr2)

        slices1, actual_coords = _extract_slices(data1, self.slice_mode, slice_coords)
        slices2, _ = _extract_slices(data2, self.slice_mode, slice_coords)
        n_slices = len(slices1)

        display_dims = [str(d) for d in data1.dims if d != self.slice_mode]
        dim_row, dim_col = display_dims[0], display_dims[1]

        plot_indices = self._resolve_axes_layout(
            data1,
            n_slices,
            actual_coords,
            dim_row,
            dim_col,
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
            if hover_x.ndim == 1 and hover_y.ndim == 1:
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
                        name=str(data.name)
                        if data.name is not None
                        else f"data{i + 1}",
                        units=data.attrs.get("units"),
                    )

            self._style_slice_axis(
                ax,
                axis_idx,
                data1,
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

            - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
              labeled `0`; each unique non-zero integer identifies a distinct,
              non-overlapping region. The `region` coordinate of the output holds the
              integer label values.
            - **Stacked mask format**: Has a leading `mask` dimension followed by
              spatial dims, e.g. `(mask, z, y, x)`. Each layer has values in `{0,
              region_id}` and regions may overlap. The `region` coordinate of the
              output holds the `mask` coordinate values (e.g., region label).

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

        squeeze_dims = [
            d for d in mask.dims if d != self.slice_mode and mask.sizes[d] == 1
        ]
        if squeeze_dims:
            mask = mask.squeeze(dim=squeeze_dims)

        _validate_voxel_affine_slice_mode(mask, self.slice_mode)

        if self.slice_mode not in mask.dims:
            raise ValueError(f"slice_mode '{self.slice_mode}' not in mask dimensions")

        if mask.ndim != 3:
            raise ValueError(f"mask must be 3D, got shape {mask.shape}")

        # Only sort the display dims: sorting slice_mode itself would silently
        # reorder panels (e.g. non-monotonic z, or an arbitrary "region" order).
        display_dims = [str(d) for d in mask.dims if d != self.slice_mode]
        mask = sort_coords_for_plot(mask, display_dims)

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

        dim_row, dim_col = display_dims[0], display_dims[1]

        if slice_coords is None:
            if self.slice_mode in mask.coords:
                slice_coords = list(mask.coords[self.slice_mode].values)
            else:
                slice_coords = list(range(mask.sizes[self.slice_mode]))

        slices, actual_coords = _extract_slices(mask, self.slice_mode, slice_coords)
        n_slices = len(slices)

        if match_coordinates:
            matched_indices = self._find_matching_axes(actual_coords)

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

            if dim_col in slice_da.coords:
                x_coords = slice_da.coords[dim_col].values.astype(float)
            else:
                x_coords = np.arange(slice_da.sizes[dim_col])
            if dim_row in slice_da.coords:
                y_coords = slice_da.coords[dim_row].values.astype(float)
            else:
                y_coords = np.arange(slice_da.sizes[dim_row])

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

                    # Map pixel indices to physical coordinates.
                    # Contours are at pixel boundaries, so we need to interpolate
                    # between coordinate centers to get edge positions.
                    # contour[:, 0] is row (y) index, contour[:, 1] is col (x) index
                    x_idx = contour[:, 1]
                    y_idx = contour[:, 0]
                    x_physical = np.interp(
                        x_idx,
                        np.arange(len(x_coords)),
                        x_coords,
                    )
                    y_physical = np.interp(
                        y_idx,
                        np.arange(len(y_coords)),
                        y_coords,
                    )
                    ax.plot(
                        x_physical,
                        y_physical,
                        color=color,
                        linewidth=linewidths,
                        linestyle=linestyles,
                        **kwargs,
                    )

            if resolved_roi_labels and self.figure is not None:
                self._attach_or_update_hover_manager(resolved_roi_labels)
                self._hover_manager.register_data_to_axis(
                    ax,
                    x_coords=np.asarray(x_coords, dtype=float),
                    y_coords=np.asarray(y_coords, dtype=float),
                    data_2d=np.asarray(slice_data),
                    role="labels",
                    name=str(mask.name) if mask.name is not None else "label",
                )

            if not match_coordinates:
                ax.set_aspect("equal")

                # Compute limits from physical coordinates, not from auto-scaled
                # matplotlib limits, which may include padding.
                x_edges = _centers_to_edges(x_coords)
                y_edges = _centers_to_edges(y_coords)
                xlim = (float(x_edges.min()), float(x_edges.max()))
                ylim = (float(y_edges.min()), float(y_edges.max()))
                self._set_ax_lims(ax, xlim, ylim)
                self._style_ax(ax)
                ax.set_xlabel(
                    _build_axis_label(mask, dim_col),
                    color=self._text_color,
                    fontsize=label_fontsize,
                )
                ax.set_ylabel(
                    _build_axis_label(mask, dim_row),
                    color=self._text_color,
                    fontsize=label_fontsize,
                )
                ax.set_title(
                    self._build_slice_title(mask, actual_coords[slice_idx]),
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
    Each panel shows the contours for one slice along `slice_mode`, drawn in physical
    coordinates when available.

    Parameters
    ----------
    mask : xarray.DataArray
        Integer label map in one of two formats:

        - **Flat label map**: Spatial dims only, e.g. `(z, y, x)`. Background voxels
          labeled `0`; each unique non-zero integer identifies a distinct,
          non-overlapping region. The `regions` coordinate of the output holds the
          integer label values.
        - **Stacked mask format**: Has a leading `masks` dimension followed by
          spatial dims, e.g. `(masks, z, y, x)`. Each layer has values in `{0,
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
        Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`). After
        slicing, each panel must be 2D.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values along `slice_mode` are used.
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
    label, then mapped to physical coordinates via linear interpolation between
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
    slice_mode: str = "z",
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
) -> VolumePlotter:
    """Plot 2D slices of a volume using matplotlib.

    Displays a series of 2D slices extracted along `slice_mode` as a grid of subplots.
    Each slice is rendered using physical coordinates for axis ticks when available.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array. Unitary dimensions are squeezed before processing. After
        squeezing, data must be 3D. Complex-valued data is converted to magnitude
        before display.
    slice_coords : list[collections.abc.Hashable], optional
        Coordinate values along `slice_mode` at which to extract slices. Numeric
        coordinates are matched by nearest-neighbour lookup; non-numeric
        coordinates (e.g. region labels) require an exact match. If not provided,
        all coordinate values along `slice_mode` are used.
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g., `"x"`, `"y"`, `"z"`,
        `"time"`). After slicing, each panel must be 2D.
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
    correctly. Because each panel is drawn in physical coordinate space, multiple calls
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
    plotter = VolumePlotter(
        slice_mode=slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
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
    slice_mode: str = "z",
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
) -> VolumePlotter:
    """Plot a red/cyan composite of two volumes as a grid of 2D slice panels.

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
    slice_mode : str, default: "z"
        Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`). After slicing, each
        panel must be 2D.
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
    plotter = VolumePlotter(
        slice_mode=slice_mode,
        figure=figure,
        axes=axes,
        bg_color=bg_color,
        fg_color=fg_color,
        yincrease=yincrease,
        xincrease=xincrease,
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
) -> VolumePlotter:
    """Plot a statistical map, optionally over a background volume.

    Performs the recurring pattern of [`plot_volume`][confusius.plotting.plot_volume] to
    show a background anatomical volume +
    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume] to overlay
    a statistical map, with the colormap and range picked automatically based on
    whether the statistic is diverging (has both positive and negative values) or
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
        Dimension along which to slice (e.g., `"x"`, `"y"`, `"z"`, `"time"`). After
        slicing, each panel must be 2D.
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
        slices. Ignored when `norm` is provided, or when `auto_range` resolves to a
        range anchored at zero (see below).
    vmax : float, optional
        Upper bound of the colormap. If not provided, defaults to the maximum value
        of `stat_map`, computed over the full array rather than just the displayed
        slices. Ignored when `norm` is provided, or when `auto_range=True` and
        `stat_map` has only non-positive values.
    auto_range : bool, default: True
        Whether to pick the colormap range and default colormap automatically based
        on the sign of `stat_map`:

        - Both positive and negative values: diverging, symmetric `[-m, m]` range
          where `m = max(|vmin|, |vmax|)` (using the resolved bounds above),
          with `cmap` defaulting to `"coolwarm"` — the right choice for diverging
          statistics where the sign is meaningful (e.g. t-statistics, correlation
          coefficients, PCA/ICA component maps).
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
    `plot_volume(bg_volume, ...)` followed by `plotter.add_volume(stat_map,
    alpha=alpha, cmap=resolved_cmap, vmin=resolved_vmin, vmax=resolved_vmax, ...)`.
    Use those functions directly for finer control, e.g. a custom, non-zero-anchored
    asymmetric range.

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
        )

    resolved_vmin, resolved_vmax, resolved_cmap = _resolve_stat_map_style(
        stat_map, vmin, vmax, cmap, auto_range
    )

    return plotter.add_volume(
        stat_map,
        slice_coords=slice_coords,
        match_coordinates=bg_volume is not None,
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
        Input data array with a `"time"` dimension and coordinate.
    mask : xarray.DataArray, optional
        Boolean mask to select voxels. Defaults to all non-zero voxels.
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
        Keys: `signals` (DataArray with shape `(time, voxels)`), `vmin` (float),
        `vmax` (float), `xlabel` (str), `time_coord` (DataArray | None).
    """
    if np.iscomplexobj(data):
        data = xr.ufuncs.abs(data)

    validate_time_series(data, "plot_carpet", check_time_chunks=False)

    n_timepoints = data.sizes["time"]

    non_zero = (data != 0).any(dim="time")
    if mask is None:
        mask = non_zero
    else:
        mask = mask & non_zero

    signals = extract_with_mask(data, mask)

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
        Input data array with a 'time' dimension.
    mask : xarray.DataArray, optional
        Boolean mask with same spatial dimensions and coordinates as `data`. True
        values indicate voxels to include. If not provided, all non-zero voxels
        from the data are included.
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
