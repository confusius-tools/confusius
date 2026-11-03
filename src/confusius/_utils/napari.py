"""Napari layer-construction helpers shared by the CLI, GUI panels, and file readers."""

import warnings
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from napari.utils.colormaps import DirectLabelColormap, ensure_colormap
from napari.utils.notifications import show_warning

from confusius._dims import TIME_DIM
from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.coordinates import get_coordinate_spacings_best_effort
from confusius._utils.stack import find_stack_level


def infer_layer_type(dtype: npt.DTypeLike) -> Literal["image", "labels"]:
    """Infer the napari layer type to use for an array's dtype.

    Parameters
    ----------
    dtype : numpy.typing.DTypeLike
        Dtype of the array to be displayed.

    Returns
    -------
    {"image", "labels"}
        `"labels"` for integer or boolean dtypes (e.g. atlas annotations, ROI masks),
        which should render with per-label colors rather than a continuous colormap.
        `"image"` for all other dtypes.
    """
    # Dtype kinds: "b" boolean, "i" signed integer, "u" unsigned integer
    return "labels" if np.dtype(dtype).kind in "biu" else "image"


def build_direct_label_colormap(data: xr.DataArray) -> DirectLabelColormap | None:
    """Build a per-label `DirectLabelColormap` from a labels DataArray's attrs.

    Parameters
    ----------
    data : xarray.DataArray
        Labels array. Uses `data.attrs["cmap"]` / `data.attrs["norm"]` when present
        (as set by atlas functions), or reconstructs them from
        `data.attrs["rgb_lookup"]` when `cmap`/`norm` are absent (e.g. after a
        Zarr round-trip, since they are not serializable).

    Returns
    -------
    napari.utils.colormaps.DirectLabelColormap or None
        Colormap mapping each non-zero label to its atlas color, with
        unknown/background labels transparent. Labels are taken from the keys of
        `data.attrs["rgb_lookup"]` when present, so labels absent from `data` are
        colored too; otherwise from the unique values of `data`, which loads it
        into memory. `None` if `data.attrs` carries neither a `cmap`/`norm` pair
        nor `rgb_lookup`.
    """
    cmap_attr = data.attrs.get("cmap")
    norm_attr = data.attrs.get("norm")
    rgb_lookup = data.attrs.get("rgb_lookup")
    if cmap_attr is None or norm_attr is None:
        if rgb_lookup is None:
            return None
        cmap_attr, norm_attr = build_atlas_cmap_and_norm(rgb_lookup)

    color_dict: defaultdict[int | None, np.ndarray] = defaultdict(
        lambda: np.zeros(4, dtype=np.float32)  # unknown labels → transparent.
    )
    if rgb_lookup:
        labels = sorted(int(k) for k in rgb_lookup if int(k) != 0)
    else:
        labels = sorted(int(label) for label in np.unique(data.values) if label != 0)
    for label in labels:
        color_dict[int(label)] = np.array(
            cmap_attr(norm_attr(int(label))), dtype=np.float32
        )
    return DirectLabelColormap(color_dict=color_dict, background_value=0)


def get_napari_layer_geometry(
    data: xr.DataArray,
) -> tuple[
    list[float], list[float], list[str], list[str | None], list[str], dict[str, float]
]:
    """Return napari layer geometry metadata for `data`.

    `data` must carry a voxel-to-world index: `build_napari_layer_data` validates
    this before reaching this point. Spacing is read from `.fusi.spacing`, falling back to
    the plain-coordinate best-effort median only for irregularly spaced coordinates
    (`.fusi.spacing` returns `None` there).

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to compute napari layer geometry for.

    Returns
    -------
    scale : list[float]
        Per-dimension scale, in `data.dims` order.
    translate : list[float]
        Per-dimension translation, in `data.dims` order.
    axis_labels : list[str]
        Per-dimension display label, in `data.dims` order: the linked world
        coordinate name (`z`/`y`/`x`) for a native voxel dim that has one, otherwise
        the dimension's own name.
    units : list[str | None]
        Per-dimension units, in `data.dims` order.
    non_uniform : list[str]
        Dimension names whose coordinates were non-uniform (median spacing used).
    spacing : dict[str, float]
        Spacing per dimension name, as used to compute `scale`.
    """
    all_dims = [str(dim) for dim in data.dims]

    coordinate_spacing, non_uniform = get_coordinate_spacings_best_effort(data)
    fusi_spacing = data.fusi.spacing
    spacing = {
        dim: fusi_spacing[dim] if fusi_spacing.get(dim) is not None else fallback
        for dim, fallback in coordinate_spacing.items()
    }
    origin = data.fusi.origin
    world_dim = {"k": "z", "j": "y", "i": "x"}
    scale: list[float] = []
    translate: list[float] = []
    axis_labels: list[str] = []
    units: list[str | None] = []
    for dim in all_dims:
        coord_name = world_dim.get(dim, dim)
        coord = data.coords.get(coord_name)
        # `origin` (`.fusi.origin`) always has an entry for every dim -- spatial
        # dims via the voxel-to-world index, others defaulting to 0.0 -- so it's a
        # safe fallback translate whenever `coord` isn't itself 1-D numeric world
        # geometry for `dim`.
        is_world_coord = (
            coord is not None
            and coord.dims == (dim,)
            and np.issubdtype(coord.dtype, np.number)
        )
        scale.append(spacing[dim])
        translate.append(
            np.float64(coord.values[0]).item() if is_world_coord else origin[coord_name]
        )
        axis_labels.append(coord_name if coord_name in data.coords else dim)
        units.append(
            data.coords[coord_name].attrs.get("units")
            if coord_name in data.coords
            else None
        )
    return scale, translate, axis_labels, units, non_uniform, spacing


def build_napari_layer_data(
    data: xr.DataArray,
    name: str | None = None,
    layer_type: Literal["image", "labels", "auto"] = "auto",
    resample_interpolation: Literal["linear", "nearest", "bspline"] | None = None,
    resample_fill_value: float | None = None,
) -> tuple[Any, dict[str, Any], Literal["image", "labels"]]:
    """Build napari layer data for a VoxelData array.

    This is the single DataArray-to-layer conversion shared by `plot_napari` and the
    napari plugin readers, so data looks the same however it is opened. Viewer-level
    settings (displayed dimension order, colorbar, scale bar) and contrast limits are
    left to the caller.

    Parameters
    ----------
    data : xarray.DataArray
        VoxelData array to display.
    name : str, optional
        Layer name. If not provided, `data.name` is used.
    layer_type : {"image", "labels", "auto"}, default: "auto"
        Type of layer to build. "auto" picks "labels" for integer or boolean data and
        "image" otherwise.
    resample_interpolation : {"linear", "nearest", "bspline"}, optional
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `data` onto an axis-aligned world grid. If not provided,
        defaults to `"nearest"` for labels layers and `"linear"` for image layers.
    resample_fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after resampling
        oblique data. If not provided, defaults to `data`'s own minimum value.

    Returns
    -------
    data : Any
        Layer data payload: the magnitude of complex image data, and integer values
        for labels.
    kwargs : dict[str, Any]
        Keyword arguments for the napari layer constructor. Holds the layer name,
        world-space `scale`/`translate`, world `axis_labels`, `units` when the
        coordinates carry them, a `colormap` (from `data.attrs` for images, falling
        back to `"gray"`, or a per-label colormap for labels when available), and
        `metadata` with the displayed (`"xarray"`) and input (`"source_xarray"`)
        DataArrays. Image layers also get `rgb=False`, and labels carrying
        `roi_labels` attrs get matching `features`.
    layer_type : {"image", "labels"}
        Napari layer type that was built.

    Raises
    ------
    ValueError
        If `layer_type` is not one of the supported values, or if `data` is not a
        VoxelData array.

    Warns
    -----
    UserWarning
        If a dimension has non-uniform spacing, if complex data is converted to its
        magnitude, or if `data.attrs["cmap"]` is not a valid napari colormap.
    """
    from confusius._utils.plotting import (
        coerce_complex_to_magnitude,
        resample_to_axis_aligned_world_grid,
        sort_coords_for_plot,
    )
    from confusius.validation import ensure_voxeldata

    resolved_layer_type = layer_type
    if resolved_layer_type == "auto":
        resolved_layer_type = infer_layer_type(data.dtype)
    if resolved_layer_type not in ("image", "labels"):
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. Expected 'image', 'labels' or 'auto'."
        )
    source_data = ensure_voxeldata(data)

    # Resampling an oblique input onto an axis-aligned world grid keeps the
    # voxel-to-world index (see resample_to_axis_aligned_world_grid); already
    # axis-aligned input is returned unchanged. Either way `data` stays on its native
    # dims here. Layers are always positioned in world space (scale/translate are
    # real physical spacing/origin), so axis_labels shows world names (z/y/x) to
    # honestly reflect what the dims slider actually scrubs through.
    resolved_interpolation = resample_interpolation or (
        "nearest" if resolved_layer_type == "labels" else "linear"
    )
    data = resample_to_axis_aligned_world_grid(
        source_data,
        interpolation=resolved_interpolation,
        fill_value=resample_fill_value,
    )
    data = sort_coords_for_plot(data, [dim for dim in data.dims if dim != TIME_DIM])

    scale, translate, axis_labels, all_units, non_uniform, spacing = (
        get_napari_layer_geometry(data)
    )
    for dim in non_uniform:
        warnings.warn(
            f"'{dim}' has non-uniform spacing; using median {spacing[dim]:.4g} "
            "(positions along this axis may be approximate).",
            stacklevel=find_stack_level(),
        )

    kwargs: dict[str, Any] = {
        "name": data.name if name is None else name,
        "scale": scale,
        "translate": translate,
        "axis_labels": axis_labels,
        "metadata": {"xarray": data, "source_xarray": source_data},
    }
    if any(u is not None for u in all_units):
        kwargs["units"] = all_units

    if resolved_layer_type == "labels":
        # Keep label conversion lazy for Dask-backed data.
        if not np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int32)
        colormap = build_direct_label_colormap(data)
        if colormap is not None:
            kwargs["colormap"] = colormap
        if (roi_labels := data.attrs.get("roi_labels")) is not None:
            kwargs["features"] = build_roi_labels_features(roi_labels)
    else:
        data = coerce_complex_to_magnitude(data, caller="napari")
        colormap = data.attrs.get("cmap", "gray")
        try:
            ensure_colormap(colormap)
        except (KeyError, TypeError):
            warnings.warn(
                f"{colormap!r} is not a valid napari colormap; falling back to 'gray'.",
                stacklevel=find_stack_level(),
            )
            colormap = "gray"
        kwargs["colormap"] = colormap
        # VoxelData arrays are scalar fields; prevent napari from auto-interpreting a
        # trailing axis of length 3/4 as RGB channels.
        kwargs["rgb"] = False

    return data.data, kwargs, resolved_layer_type


def convert_dataarray_to_layer_data(
    da: xr.DataArray, name: str
) -> tuple[Any, dict[str, Any], Literal["image", "labels"]]:
    """Convert a ConfUSIus DataArray to a napari FullLayerData tuple.

    Used by the napari plugin readers and sample data, which hand the tuple to napari
    instead of adding the layer themselves. Wraps `build_napari_layer_data`, shows
    its warnings as napari notifications, and computes image contrast limits up
    front.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to convert.
    name : str
        Layer name to assign in napari.

    Returns
    -------
    data : Any
        Layer data payload.
    kwargs : dict[str, Any]
        Keyword arguments for the napari layer constructor.
    layer_type : {"image", "labels"}
        Napari layer type inferred from `da.dtype`.
    """
    # napari only routes Python warnings to its notifications inside
    # `notification_manager`, which a reader call is not, so warnings would otherwise
    # only reach the terminal.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data, kwargs, layer_type = build_napari_layer_data(da, name=name)
    for caught_warning in caught:
        if issubclass(caught_warning.category, UserWarning):
            show_warning(str(caught_warning.message))
        else:
            warnings.warn_explicit(
                caught_warning.message,
                caught_warning.category,
                caught_warning.filename,
                caught_warning.lineno,
            )
    if layer_type == "image":
        kwargs["contrast_limits"] = calc_data_range(data)
    return data, kwargs, layer_type


def build_roi_labels_features(roi_labels: dict[int, str]) -> pd.DataFrame:
    """Build a napari `features` table reporting ROI names in the status bar.

    Parameters
    ----------
    roi_labels : dict[int, str]
        Mapping from integer ROI id to display name, as stored in
        `data.attrs["roi_labels"]`.

    Returns
    -------
    pandas.DataFrame
        Table with `"index"` and `"name"` columns, suitable for a
        `napari.layers.Labels.features` assignment (or the `features` constructor
        argument). Napari's built-in `Labels.get_status` then appends `name: <roi
        name>` to the status bar (and cursor tooltip) whenever the cursor is over a
        labelled voxel. A row for label `0` is included with a NaN name so
        background hovers do not show napari's default `[No Properties]`
        placeholder.
    """
    ids: list[int] = [0]
    names: list[float | str] = [float("nan")]
    for sid, name in roi_labels.items():
        sid_int = int(sid)
        if sid_int != 0:
            ids.append(sid_int)
            names.append(str(name))
    return pd.DataFrame({"index": ids, "name": names})
