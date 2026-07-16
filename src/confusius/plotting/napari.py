"""Napari-based visualization utilities for fUSI data."""

import warnings
from typing import TYPE_CHECKING, Literal, cast

import napari
import numpy as np
import xarray as xr

from confusius._utils.coordinates import get_coordinate_spacings_best_effort
from confusius._utils.napari import (
    build_direct_label_colormap,
    build_roi_labels_features,
)
from confusius._utils.stack import find_stack_level
from confusius.plotting._utils import (
    coerce_complex_to_magnitude,
    sort_coords_for_plot,
)

if TYPE_CHECKING:
    from napari import Viewer
    from napari.layers import Image, Labels


def plot_napari(
    data: xr.DataArray,
    show_colorbar: bool = True,
    show_scale_bar: bool = True,
    dim_order: tuple[str, ...] | None = None,
    viewer: "Viewer | None" = None,
    layer_type: Literal["image", "labels"] = "image",
    **layer_kwargs,
) -> "tuple[Viewer, Image | Labels]":
    """Display fUSI data using the napari viewer.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to visualize. Expected dimensions are (time, z, y, x) where
        z is the elevation/stacking axis, y is depth, and x is lateral. Use
        `dim_order` to specify a different dimension ordering. Can be image data
        or label/mask data (e.g., ROIs, segmentations).
    show_colorbar : bool, default: True
        Whether to show the colorbar. Only applies to image layers.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    dim_order : tuple[str, ...], optional
        Dimension ordering for the spatial axes (last three dimensions). If not
        provided, the ordering of the last three dimensions in `data` is used.
    viewer : napari.Viewer, optional
        Existing napari viewer to add the layer to. If not provided, a new viewer
        is created.
    layer_type : {"image", "labels"}, default: "image"
        Type of layer to create. Use "image" for fUSI data and "labels" for
        ROI masks, segmentations, or other label data.
    **layer_kwargs
        Additional keyword arguments passed to the layer creation method
        (`napari.imshow` for images or `viewer.add_labels` for labels).
        For image layers, if `data.attrs` contains `"cmap"` and `"colormap"`
        is not in `layer_kwargs`, the attribute is used as the colormap.
        For labels layers, if `data.attrs` contains `"cmap"` and `"norm"`
        (as set by atlas functions) and `"colormap"` is not in `layer_kwargs`,
        a per-label color dict is built automatically from those attributes.

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance with the layer added.
    layer : napari.layers.Image or napari.layers.Labels
        The layer added to the viewer.

    Notes
    -----
    Complex-valued data is converted to magnitude (`abs(data)`) before display.

    If all spatial dimensions have coordinates, their spacing is used as the scale
    parameter for napari to ensure correct physical scaling. If any spatial dimension is
    missing coordinates, no scaling is applied. The spacing is computed as the median
    difference between consecutive coordinate values.

    When spatial coordinates carry a `units` attribute (e.g. `"m"`), the unit list is
    forwarded to napari as the `units` layer parameter, which populates the status bar
    with physical coordinates and sets the scale bar unit if units are consistent across
    displayed axes.

    For unitary dimensions (e.g., a single-slice elevation axis in 2D+t data), the
    spacing cannot be inferred from coordinates. In that case, the function looks for a
    `voxdim` attribute on the coordinate variable
    (`data.coords[dim].attrs["voxdim"]`) and uses it as the spacing. If no such
    attribute is found, unit spacing is assumed and a warning is emitted.

    The first coordinate value of each spatial dimension is used as the `translate`
    parameter so that the image is positioned at its correct physical origin. For
    dimensions without coordinates, a translate of `0.0` is used. This ensures that
    multiple datasets with different fields of view overlay correctly when added to the
    same viewer.

    Examples
    --------
    >>> import xarray as xr
    >>> from confusius.plotting import plot_napari
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> viewer, layer = plot_napari(data)

    >>> # Custom contrast limits
    >>> viewer, layer = plot_napari(data, contrast_limits=(0, 100))

    >>> # Different dimension ordering (e.g., depth, elevation, lateral)
    >>> viewer, layer = plot_napari(data, dim_order=("y", "z", "x"))

    >>> # Add a second dataset as a new layer in an existing viewer
    >>> viewer, layer = plot_napari(data1)
    >>> viewer, layer = plot_napari(data2, viewer=viewer)

    >>> # Display ROI labels (e.g., segmentation mask)
    >>> roi_mask = xr.open_zarr("output.zarr")["roi_mask"]
    >>> viewer, layer = plot_napari(roi_mask, layer_type="labels")

    >>> # Overlay labels on existing image
    >>> viewer, layer = plot_napari(data)
    >>> viewer, layer = plot_napari(roi_mask, viewer=viewer, layer_type="labels")
    """
    if layer_type not in ("image", "labels"):
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. Expected 'image' or 'labels'."
        )

    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    data = sort_coords_for_plot(data, spatial_dims)

    if dim_order is not None and set(dim_order) != set(spatial_dims):
        raise ValueError(
            f"dim_order {dim_order} does not match spatial dimensions {spatial_dims}. "
            "Ensure 'dim_order' contains all spatial dimension names."
        )

    spacing, non_uniform = get_coordinate_spacings_best_effort(data)
    for dim in non_uniform:
        warnings.warn(
            f"'{dim}' has non-uniform spacing; using median {spacing[dim]:.4g} "
            "(positions along this axis may be approximate).",
            stacklevel=find_stack_level(),
        )
    scale = [spacing[str(dim)] for dim in all_dims]

    # .origin falls back to 0.0 for dimensions without coordinates.
    origin = data.fusi.origin
    coord_translates = [origin[dim] for dim in all_dims]

    # napari requires units to cover ALL dims. Build in all_dims order so each
    # unit aligns with the correct dimension; passing None is accepted for
    # unlabelled axes.
    all_units: list[str | None] = [
        data.coords[dim].attrs.get("units") if dim in data.coords else None
        for dim in all_dims
    ]

    layer_kwargs.setdefault("name", data.name)
    if any(u is not None for u in all_units):
        layer_kwargs.setdefault("units", all_units)

    if layer_type == "image":
        plot_data = coerce_complex_to_magnitude(data, caller="plot_napari")

        # The last 2 (2D) or 3 (3D) dimensions are the displayed spatial axes.
        if dim_order is not None:
            order = []
            if time_dim:
                order.append(all_dims.index(time_dim))
            for dim in dim_order:
                if dim in all_dims:
                    order.append(all_dims.index(dim))
            layer_kwargs["order"] = tuple(order)

        layer_kwargs.setdefault("axis_labels", all_dims)

        if "colormap" not in layer_kwargs:
            cmap_attr = data.attrs.get("cmap")
            if cmap_attr is not None:
                layer_kwargs["colormap"] = cmap_attr

        # fUSI arrays are scalar fields; prevent napari from auto-interpreting a
        # trailing axis of length 3/4 as RGB channels.
        layer_kwargs.setdefault("rgb", False)

        layer_kwargs.setdefault("translate", coord_translates)

        # Pass the underlying array (numpy or Dask) rather than the DataArray. napari's
        # rendering loop adds overhead on every frame when given an xarray DataArray,
        # making time scrubbing noticeably slow for lazy (Dask-backed) data.
        layer_kwargs.setdefault("metadata", {})["xarray"] = data
        viewer, layer = napari.imshow(
            plot_data.data,
            scale=scale,
            viewer=viewer,
            **layer_kwargs,
        )
        # napari.imshow stubs declare list[Image] but at runtime returns Image
        # directly: cast to silence the type checker.
        layer = cast("Image", layer)

        # Workaround for napari 0.6.6+: non-numpy data (xarray DataArray / Dask) defers
        # contrast-limit computation to the async slice worker. The worker fires AFTER
        # _should_calc_clims is set, but in napari 0.6.6 the initial viewer refresh
        # triggered by the `inserted` event completes before that flag is raised, so
        # contrast limits stay at (0, 1) for float data until the user manually clicks
        # "once". Explicitly computing them here is robust across napari versions. See
        # https://github.com/napari/napari/pull/8756.
        if "contrast_limits" not in layer_kwargs:
            layer.reset_contrast_limits_range()
            layer.reset_contrast_limits("data")

        if show_colorbar:
            layer.colorbar.visible = True

    elif layer_type == "labels":
        layer_kwargs.setdefault("translate", coord_translates)
        layer_kwargs.setdefault("metadata", {})["xarray"] = data
        if viewer is None:
            viewer = napari.Viewer()
        values = data.values
        if not np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.int32)

        # Build a DirectLabelColormap from attrs when the caller has not already
        # supplied one. This lets atlas annotations and masks carry their colormap
        # automatically into the viewer.
        if "colormap" not in layer_kwargs:
            colormap = build_direct_label_colormap(data)
            if colormap is not None:
                layer_kwargs["colormap"] = colormap

        layer = viewer.add_labels(  # type: ignore
            values,
            scale=scale,
            **layer_kwargs,
        )

        if (roi_labels := data.attrs.get("roi_labels")) is not None:
            layer.features = build_roi_labels_features(roi_labels)

    assert viewer is not None
    viewer.scale_bar.visible = show_scale_bar

    return viewer, layer


def draw_napari_labels(
    data: xr.DataArray,
    labels_layer_name: str = "labels",
    viewer: "Viewer | None" = None,
    **kwargs,
) -> "tuple[Viewer, Labels]":
    """Open a napari viewer to interactively paint integer labels over fUSI data.

    Displays the data as an image layer and adds an empty Labels layer on top. The user
    can paint integer labels directly on the image using napari's brush tool. After
    painting, call [`labels_from_layer`][confusius.plotting.labels_from_layer] with the
    returned Labels layer and the original data to obtain an integer label map as a
    DataArray with the same spatial coordinates.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to display as the background image. Typically a time-averaged
        power Doppler frame, e.g. `data.mean("time")`.
    labels_layer_name : str, default: "labels"
        Name assigned to the Labels layer added to the viewer.
    viewer : napari.Viewer, optional
        Existing napari viewer to add layers to. If not provided, a new viewer
        is created via [`plot_napari`][confusius.plotting.plot_napari].
    **kwargs
        Additional keyword arguments forwarded to
        [`plot_napari`][confusius.plotting.plot_napari] for the image layer
        (e.g. `colormap`, `contrast_limits`).

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance with the image and Labels layers.
    labels_layer : napari.layers.Labels
        The empty Labels layer initialised to zeros. After the user paints
        labels in the viewer, pass this layer to
        [`labels_from_layer`][confusius.plotting.labels_from_layer] to obtain
        an integer label map.

    Notes
    -----
    The Labels layer is initialised with the same `scale` and `translate`
    parameters as the image layer so that the napari canvas shows a consistent
    physical coordinate frame regardless of voxel spacing or data origin.

    Examples
    --------
    >>> import xarray as xr
    >>> import confusius  # Register accessor.
    >>> pwd = xr.open_zarr("output.zarr")["power_doppler"].compute()
    >>> # Display the time-averaged image and add an interactive Labels layer.
    >>> viewer, labels_layer = draw_napari_labels(pwd.mean("time"))
    >>> # … paint labels in the viewer …
    >>> # Convert painted labels to an integer label map DataArray.
    >>> label_map = labels_from_layer(labels_layer, pwd.mean("time"))
    """
    viewer, _ = plot_napari(data, viewer=viewer, **kwargs)

    # Reuse the same spatial scale and translate that plot_napari computed for
    # the image layer so the Labels layer overlays correctly.
    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    spacing = data.fusi.spacing
    spatial_scale = [
        s if (s := spacing[dim]) is not None else 1.0 for dim in spatial_dims
    ]
    spatial_translate = [
        float(data.coords[dim].values[0]) if dim in data.coords else 0.0
        for dim in spatial_dims
    ]
    spatial_shape = tuple(data.sizes[d] for d in spatial_dims)

    labels_array = np.zeros(spatial_shape, dtype=np.int32)
    labels_layer = viewer.add_labels(  # type: ignore
        labels_array,
        scale=spatial_scale,
        translate=spatial_translate,
        name=labels_layer_name,
    )

    return viewer, labels_layer


def labels_from_layer(
    labels_layer: "Labels",
    data: xr.DataArray,
) -> xr.DataArray:
    """Convert a napari Labels layer to an integer label map DataArray.

    Reads the integer array painted in `labels_layer` and wraps it in a DataArray whose
    spatial dimensions and coordinates match those of `data`. The result is compatible
    with [`extract_with_labels`][confusius.extract.extract_with_labels],
    [`plot_contours`][confusius.plotting.plot_contours], and
    [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours].

    Parameters
    ----------
    labels_layer : napari.layers.Labels
        A Labels layer populated by the user (e.g. via
        [`draw_napari_labels`][confusius.plotting.draw_napari_labels]). Integer values
        identify distinct regions; zero is the background and is excluded from
        downstream analyses.
    data : xarray.DataArray
        Reference data array. Its spatial dimensions and coordinates define the shape
        and labelling of the output. A time dimension, if present, is ignored: the
        label map is purely spatial.

    Returns
    -------
    xarray.DataArray
        Stacked integer DataArray with dims `["mask", *spatial_dims]`, where the
        `mask` coordinate holds each unique non-zero label integer. Each layer
        `mask=k` has values `k` where the user painted label `k` and `0` elsewhere.
        This format is directly compatible with
        [`extract_with_labels`][confusius.extract.extract_with_labels],
        [`plot_contours`][confusius.plotting.plot_contours], and
        [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours],
        and can be sliced by label (e.g. `label_map.sel(mask=2)`) for per-label
        display. The `attrs` dict carries:

        - `"long_name"` — human-readable name.
        - `"labels_layer_name"` — name of the source napari layer.
        - `"rgb_lookup"` — `dict[int, list[int]]` mapping each non-zero
          label to its `[r, g, b]` color (0–255) as painted in napari.

    Notes
    -----
    The label array is taken directly from `labels_layer.data`. No
    rasterisation is performed: this is a direct read of the painted values.

    Per-label colors are read from `labels_layer.get_color(label)`, which works for both
    the default cyclic colormap and any `DirectLabelColormap` set on the layer.

    Examples
    --------
    >>> import xarray as xr
    >>> import confusius  # Register accessor.
    >>> pwd = xr.open_zarr("output.zarr")["power_doppler"].compute()
    >>> viewer, labels_layer = draw_napari_labels(pwd.mean("time"))
    >>> # … paint labels in the viewer …
    >>> label_map = labels_from_layer(labels_layer, pwd.mean("time"))
    >>> label_map.dims
    ('mask', 'z', 'y', 'x')
    >>> # Slice a single label for display alongside a seed map.
    >>> label_map.sel(mask=2)
    >>> # Use the label map for region-based analysis.
    >>> from confusius.extract import extract_with_labels
    >>> signals = extract_with_labels(pwd, label_map)
    """
    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    coords = {dim: data.coords[dim] for dim in spatial_dims if dim in data.coords}

    # Build a color lookup from the napari layer so downstream consumers
    # (plot_napari, VolumePlotter.add_volume, add_contours) can render each
    # label with exactly the color the user painted in the viewer.
    # get_color() returns RGBA in [0, 1] for any non-zero label and works for
    # both the default CyclicLabelColormap and any DirectLabelColormap.
    label_array = np.asarray(labels_layer.data)
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels != 0]
    rgb_lookup: dict[int, list[int]] = {}
    for label in unique_labels:
        rgba = labels_layer.get_color(int(label))
        if rgba is not None:
            # Store 0-255 RGB (drop alpha) to match the atlas annotation convention.
            rgb_lookup[int(label)] = [int(round(c * 255)) for c in rgba[:3]]

    # Build one layer per label so the output matches the stacked mask format
    # returned by the atlas accessor's get_masks: dims=["mask", *spatial_dims] with the
    # mask coordinate holding integer label IDs. This allows per-label slicing
    # (e.g. label_map.sel(mask=2)) and is directly accepted by
    # extract_with_labels, plot_contours, and add_contours.
    layers = [np.where(label_array == k, k, 0).astype(np.int32) for k in unique_labels]
    stacked = (
        np.stack(layers, axis=0)
        if len(layers) > 0
        else np.empty((0, *label_array.shape), dtype=np.int32)
    )

    return xr.DataArray(
        stacked,
        dims=["mask", *spatial_dims],
        coords={"mask": unique_labels.astype(np.int32), **coords},
        attrs={
            "long_name": "Drawn label map",
            "labels_layer_name": labels_layer.name,
            "rgb_lookup": rgb_lookup,
        },
    )
