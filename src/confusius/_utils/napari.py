"""Napari layer-construction helpers shared by the CLI, GUI panels, and file readers."""

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from napari.layers.utils.layer_utils import calc_data_range
from napari.utils.colormaps import DirectLabelColormap, ensure_colormap
from napari.utils.notifications import show_warning

from confusius._utils.atlas import build_atlas_cmap_and_norm
from confusius._utils.coordinates import get_coordinate_spacings_best_effort


def infer_layer_type(dtype: npt.DTypeLike) -> Literal["image", "labels"]:
    """Infer the napari layer type to use for an array's dtype.

    Parameters
    ----------
    dtype : numpy.typing.DTypeLike
        Dtype of the array to be displayed.

    Returns
    -------
    {"image", "labels"}
        `"labels"` for integer dtypes (e.g. atlas annotations, ROI masks), which
        should render with per-label colors rather than a continuous colormap.
        `"image"` for all other dtypes.
    """
    return "labels" if np.issubdtype(np.dtype(dtype), np.integer) else "image"


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
        Colormap mapping each unique non-zero label in `data` to its atlas color,
        with unknown/background labels transparent. `None` if `data.attrs` carries
        neither a `cmap`/`norm` pair nor `rgb_lookup`.
    """
    cmap_attr = data.attrs.get("cmap")
    norm_attr = data.attrs.get("norm")
    if (cmap_attr is None or norm_attr is None) and "rgb_lookup" in data.attrs:
        cmap_attr, norm_attr = build_atlas_cmap_and_norm(data.attrs["rgb_lookup"])
    if cmap_attr is None or norm_attr is None:
        return None

    color_dict: defaultdict[int | None, np.ndarray] = defaultdict(
        lambda: np.zeros(4, dtype=np.float32)  # unknown labels → transparent.
    )
    for label in np.unique(data.values):
        if label == 0:
            continue  # background_value=0 is always transparent.
        color_dict[int(label)] = np.array(
            cmap_attr(norm_attr(int(label))), dtype=np.float32
        )
    return DirectLabelColormap(color_dict=color_dict, background_value=0)


def get_napari_scale_translate_units(
    data: xr.DataArray,
) -> tuple[
    list[float], list[float], list[str], list[str | None], list[str], dict[str, float]
]:
    """Return napari layer geometry metadata for `data`.

    `data` must carry a voxel-to-world index: both callers guarantee this before
    reaching this point (`plot_napari` checks it explicitly;
    `convert_dataarray_to_layer_data` only ever receives ConfUSIus-loaded or
    ConfUSIus-constructed data). Spacing is read from `.fusi.spacing`, falling back to
    the plain-coordinate best-effort median only for irregularly spaced coordinates
    (`.fusi.spacing` returns `None` there).

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to compute napari layer geometry for.

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
    for dim in all_dims:
        world_name = world_dim.get(dim, dim)
        world_coord = data.coords.get(world_name)
        if world_coord is not None and world_coord.dims == (dim,):
            values = np.asarray(world_coord.values, dtype=float)
            scale.append(spacing[dim])
            translate.append(np.float64(values[0]).item())
            axis_labels.append(world_name)
        else:
            scale.append(spacing[dim])
            translate.append(
                origin[world_name]
                if world_name in origin
                else (
                    np.float64(
                        np.asarray(data.coords[dim].values, dtype=float)[0]
                    ).item()
                    if dim in data.coords
                    else 0.0
                )
            )
            axis_labels.append(world_name if world_name in data.coords else dim)
    units: list[str | None] = []
    for dim in all_dims:
        world_name = world_dim.get(dim, dim)
        # world_name only ever differs from dim for k/j/i, and .fusi.spacing above
        # already requires a real VoxelToWorldIndex covering every present voxel
        # dim -- so a voxel dim's world coordinate is never missing while the dim
        # itself has one; the two checks can't diverge.
        if world_name in data.coords:
            units.append(data.coords[world_name].attrs.get("units"))
        else:
            units.append(None)
    return scale, translate, axis_labels, units, non_uniform, spacing


def convert_dataarray_to_layer_data(
    da: xr.DataArray, name: str
) -> tuple[Any, dict[str, Any], Literal["image", "labels"]]:
    """Convert a ConfUSIus DataArray to a napari FullLayerData tuple.

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
    from confusius.plotting._utils import resample_to_axis_aligned_world_grid

    # Resampling an oblique input onto an axis-aligned world grid keeps the
    # voxel-to-world index (see resample_to_axis_aligned_world_grid); already
    # axis-aligned input is returned unchanged. Either way `da` stays on its native
    # dims here. Layers are always positioned in world space (scale/translate are
    # real physical spacing/origin), so axis_labels shows world names (z/y/x) to
    # honestly reflect what the dims slider actually scrubs through -- matching
    # plot_napari's display convention, so a file dragged into napari and the same
    # data passed to plot_napari look identical.
    source_da = da
    da = resample_to_axis_aligned_world_grid(da)

    scale, translate, axis_labels, all_units, non_uniform, spacing = (
        get_napari_scale_translate_units(da)
    )
    for dim in non_uniform:
        show_warning(
            f"'{dim}' has non-uniform spacing; using median {spacing[dim]:.4g} "
            "(positions along this axis may be approximate)."
        )

    kwargs: dict[str, Any] = {
        "name": name,
        "scale": scale,
        "translate": translate,
        "axis_labels": axis_labels,
        "metadata": {"xarray": da, "source_xarray": source_da},
    }
    if any(u is not None for u in all_units):
        kwargs["units"] = all_units

    layer_type = infer_layer_type(da.dtype)
    if layer_type == "labels":
        colormap = build_direct_label_colormap(da)
        if colormap is not None:
            kwargs["colormap"] = colormap
        if (roi_labels := da.attrs.get("roi_labels")) is not None:
            kwargs["features"] = build_roi_labels_features(roi_labels)
    else:
        colormap = da.attrs.get("cmap", "gray")
        try:
            ensure_colormap(colormap)
        except (KeyError, TypeError):
            show_warning(
                f"{colormap!r} is not a valid napari colormap; falling back to 'gray'."
            )
            colormap = "gray"
        kwargs["colormap"] = colormap
        kwargs["blending"] = "additive"
        kwargs["contrast_limits"] = calc_data_range(da.data)

    return da.data, kwargs, layer_type


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
