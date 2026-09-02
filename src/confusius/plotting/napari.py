"""Napari-based visualization utilities for fUSI data.

ConfUSIus loads SCAN and NIfTI volumes with native voxel dimensions such as `k/j/i`
and linked world coordinates such as `z/y/x`, defined by a voxel-to-world affine.
Displayed data keeps its native voxel dims and voxel-to-world index throughout --
napari's `axis_labels` layer parameter supplies the world-name (`z`/`y`/`x`) display
labels, so there is no need to rename the array's own dims. Oblique/sheared data is
resampled onto an axis-aligned world grid first, since napari's `scale`/`translate`
model can't represent oblique geometry; axis-aligned data needs no resampling.
"""

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import napari
import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import CORE_DIMS, POSE_DIM, TIME_DIM, WORLD_DIMS
from confusius._utils.geometry import (
    get_voxel_to_world_affine,
    get_voxel_to_world_units,
)
from confusius._utils.napari import (
    build_direct_label_colormap,
    build_roi_labels_features,
    get_napari_layer_geometry,
)
from confusius._utils.plotting import resample_to_axis_aligned_world_grid
from confusius._utils.stack import find_stack_level
from confusius.atlas import get_atlas_mesh
from confusius.plotting._utils import (
    coerce_complex_to_magnitude,
    sort_coords_for_plot,
)
from confusius.validation import ensure_voxeldata, validate_atlas
from confusius.xarray import create_voxeldata

if TYPE_CHECKING:
    from napari import Viewer
    from napari.layers import Image, Labels, Surface


def plot_napari(
    data: xr.DataArray,
    show_colorbar: bool = True,
    show_scale_bar: bool = True,
    dim_order: tuple[str, ...] | None = None,
    viewer: "Viewer | None" = None,
    layer_type: Literal["image", "labels"] = "image",
    resample_interpolation: Literal["linear", "nearest", "bspline"] | None = None,
    resample_fill_value: float | None = None,
    **layer_kwargs,
) -> "tuple[Viewer, Image | Labels]":
    """Display fUSI data using the napari viewer.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to visualize. Must carry a `VoxelToWorldIndex` (native
        voxel dimensions `(..., time, pose, k, j, i)` deriving world `z/y/x`
        coordinates). Use `dim_order` to specify a different displayed spatial
        ordering. Can be image data or label/mask data (e.g., ROIs, segmentations).
    show_colorbar : bool, default: True
        Whether to show the colorbar. Only applies to image layers.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    dim_order : tuple[str, ...], optional
        Dimension ordering for the spatial axes (last three dimensions). If not
        provided, singleton spatial dimensions (e.g. the elevation axis of a
        single-slice acquisition) are placed first so the canvas always shows the
        two axes that actually vary; otherwise the dimensions' native ordering in
        `data` is used.
    viewer : napari.Viewer, optional
        Existing napari viewer to add the layer to. If not provided, a new viewer
        is created.
    layer_type : {"image", "labels"}, default: "image"
        Type of layer to create. Use "image" for fUSI data and "labels" for
        ROI masks, segmentations, or other label data.
    resample_interpolation : {"linear", "nearest", "bspline"}, optional
        Interpolation method used when resampling oblique (non-axis-aligned)
        voxel-to-world `data` onto an axis-aligned world grid for display. If not
        provided, defaults to `"nearest"` for `layer_type="labels"` (blending
        distinct integer labels together is never meaningful) and `"linear"`
        otherwise.
    resample_fill_value : float, optional
        Value assigned to voxels outside `data`'s field of view after resampling
        oblique data. If not provided, defaults to `data`'s own minimum value.
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

    Napari's axis labels always show world `z`/`y`/`x` names, even though the
    displayed layer keeps `data`'s native voxel dims and index. Oblique or sheared
    data is resampled to an axis-aligned world grid first, because this display path
    does not yet pass `data.fusi.affine`'s rotation/shear through as a napari layer
    `affine`.

    If all displayed dimensions have coordinates, their spacing is used as the scale
    parameter for napari to ensure correct world scaling. If any displayed dimension
    is missing coordinates, no scaling is applied for that dimension. The spacing is
    computed as the median difference between consecutive coordinate values.

    When spatial coordinates carry a `units` attribute (e.g. `"m"`), the unit list is
    forwarded to napari as the `units` layer parameter, which populates the status bar
    with world coordinates and sets the scale bar unit if units are consistent across
    displayed axes.

    For unitary voxel dimensions (e.g., a single-slice elevation axis in 2D+t data),
    the spacing cannot be inferred from consecutive coordinate differences. In that
    case, `.fusi.spacing` derives it from the voxel-to-world affine column norm
    instead. For unitary non-voxel dimensions with no affine to fall back on, unit
    spacing is assumed and a warning is emitted.

    The first coordinate value of each displayed dimension is used as the `translate`
    parameter so that the image is positioned at its correct world origin. For
    dimensions without coordinates, a translate of `0.0` is used. This ensures that
    multiple datasets with different fields of view overlay correctly when added to
    the same viewer.

    Examples
    --------
    >>> import confusius as cf
    >>> from confusius.plotting import plot_napari
    >>> data = cf.load("output.nii.gz")
    >>> viewer, layer = plot_napari(data)

    >>> # Custom contrast limits
    >>> viewer, layer = plot_napari(data, contrast_limits=(0, 100))

    >>> # Different dimension ordering (e.g., depth, elevation, lateral)
    >>> viewer, layer = plot_napari(data, dim_order=("y", "z", "x"))

    >>> # Add a second dataset as a new layer in an existing viewer
    >>> viewer, layer = plot_napari(data1)
    >>> viewer, layer = plot_napari(data2, viewer=viewer)

    >>> # Display ROI labels (e.g., segmentation mask)
    >>> roi_mask = cf.load("roi_mask.nii.gz")
    >>> viewer, layer = plot_napari(roi_mask, layer_type="labels")

    >>> # Overlay labels on existing image
    >>> viewer, layer = plot_napari(data)
    >>> viewer, layer = plot_napari(roi_mask, viewer=viewer, layer_type="labels")
    """
    if layer_type not in ("image", "labels"):
        raise ValueError(
            f"Unknown layer_type: {layer_type!r}. Expected 'image' or 'labels'."
        )
    data = ensure_voxeldata(data)

    resolved_interpolation = resample_interpolation or (
        "nearest" if layer_type == "labels" else "linear"
    )
    source_data = data
    data = resample_to_axis_aligned_world_grid(
        data, interpolation=resolved_interpolation, fill_value=resample_fill_value
    )

    all_dims = [str(dim) for dim in data.dims]
    time_dim = TIME_DIM if TIME_DIM in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    data = sort_coords_for_plot(data, spatial_dims)

    if dim_order is not None:
        if set(dim_order) != set(spatial_dims):
            raise ValueError(
                f"dim_order {dim_order} does not match spatial dimensions "
                f"{spatial_dims}. Ensure 'dim_order' contains all spatial "
                "dimension names."
            )
    else:
        # Planar data has one or more singleton spatial dims (e.g. the elevation
        # axis of a single-slice acquisition). Default to displaying those as
        # sliders rather than relying on napari's "last two axes are the canvas"
        # convention, so the canvas always shows the two axes that actually vary,
        # regardless of how the voxel-to-world affine maps them.
        dim_order = tuple(sorted(spatial_dims, key=lambda d: data.sizes[d] != 1))

    scale, coord_translates, axis_labels, all_units, non_uniform, spacing = (
        get_napari_layer_geometry(data)
    )
    for dim in non_uniform:
        warnings.warn(
            f"'{dim}' has non-uniform spacing; using median {spacing[dim]:.4g} "
            "(positions along this axis may be approximate).",
            stacklevel=find_stack_level(),
        )

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

        # Layers are always positioned in world space (scale/translate are real
        # physical spacing/origin, and oblique input is resampled onto an
        # axis-aligned world grid), so axis_labels shows world names (z/y/x) to
        # honestly reflect what the dims slider actually scrubs through, even
        # though `data` itself stays on its native voxel dims with its index intact.
        layer_kwargs.setdefault("axis_labels", axis_labels)

        if "colormap" not in layer_kwargs:
            cmap_attr = data.attrs.get("cmap")
            if cmap_attr is not None:
                layer_kwargs["colormap"] = cmap_attr

        # VoxelData arrays are scalar fields; prevent napari from auto-interpreting a
        # trailing axis of length 3/4 as RGB channels.
        layer_kwargs.setdefault("rgb", False)

        layer_kwargs.setdefault("translate", coord_translates)

        # Pass the underlying array (numpy or Dask) rather than the DataArray. napari's
        # rendering loop adds overhead on every frame when given an xarray DataArray,
        # making time scrubbing noticeably slow for lazy (Dask-backed) data.
        layer_kwargs.setdefault("metadata", {})["xarray"] = data
        layer_kwargs["metadata"].setdefault("source_xarray", source_data)
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
        layer_kwargs["metadata"].setdefault("source_xarray", source_data)
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
    viewer.canvas.overlays.scale_bar.visible = show_scale_bar

    return viewer, layer


def plot_surface(
    mesh: tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]],
    values: npt.NDArray[np.floating] | None = None,
    viewer: "Viewer | None" = None,
    show_scale_bar: bool = True,
    **layer_kwargs,
) -> "tuple[Viewer, Surface]":
    """Display a triangular mesh as a napari surface layer.

    A thin wrapper over `napari.Viewer.add_surface`. To display an atlas region,
    prefer [`plot_atlas_mesh`][confusius.plotting.plot_atlas_mesh], which pulls the
    mesh, name, and color from the atlas and calls this function.

    Parameters
    ----------
    mesh : tuple[(N, 3) numpy.ndarray, (M, 3) numpy.ndarray]
        A `(vertices, faces)` pair, as held in each value of the dict returned by
        [`get_mesh`][confusius.atlas.AtlasAccessor.get_mesh]. `vertices` holds the
        vertex coordinates and `faces` holds zero-indexed triangle vertex indices.
    values : (N,) or (N, T) numpy.ndarray, optional
        Per-vertex scalar values used to color the surface through the layer's
        colormap. If not provided, the surface is rendered as a flat color.
    viewer : napari.Viewer, optional
        Existing napari viewer to add the layer to. If not provided, a new viewer
        is created.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    **layer_kwargs
        Additional keyword arguments passed to `napari.Viewer.add_surface`
        (e.g. `colormap`, `name`, `opacity`, `shading`, `units`).

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance with the surface layer added.
    layer : napari.layers.Surface
        The surface layer added to the viewer.

    Notes
    -----
    For a 3D mesh, the viewer is switched to 3D rendering (`ndisplay = 3`) and the
    view is reset to frame the mesh.

    Examples
    --------
    >>> import confusius as cf
    >>> atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_25um")
    >>> viewer, layer = cf.plotting.plot_surface(
    ...     atlas.atlas.get_mesh("VISp")["VISp"], colormap="magenta"
    ... )
    """
    vertices, faces = mesh
    surface_data = (vertices, faces) if values is None else (vertices, faces, values)

    if viewer is None:
        viewer = napari.Viewer()

    layer = viewer.add_surface(surface_data, **layer_kwargs)  # type: ignore

    # napari opens in 2D, where a Surface layer is drawn only as its cross-section
    # with the current slice plane, so a 3D mesh is all but invisible. Switch to 3D
    # rendering so the full mesh shows.
    if vertices.shape[1] >= 3:
        viewer.dims.ndisplay = 3
        viewer.reset_view()

    viewer.canvas.overlays.scale_bar.visible = show_scale_bar

    return viewer, layer


def plot_atlas_mesh(
    atlas: xr.Dataset,
    regions: int | str | Sequence[int | str],
    sides: (
        Literal["left", "right", "both"] | Sequence[Literal["left", "right", "both"]]
    ) = "both",
    *,
    clip: bool = True,
    values: npt.NDArray[np.floating] | None = None,
    viewer: "Viewer | None" = None,
    show_scale_bar: bool = True,
    **layer_kwargs,
) -> "tuple[Viewer, Surface]":
    """Display one or more atlas region surface meshes as a napari surface layer.

    The atlas-aware counterpart to
    [`plot_napari`][confusius.plotting.plot_napari] for meshes: pass an atlas Dataset
    and one or more regions, and the layer's mesh, name, colors, and scale bar unit are
    all pulled from the atlas before handing off to
    [`plot_surface`][confusius.plotting.plot_surface]. The meshes come from
    [`get_mesh`][confusius.atlas.AtlasAccessor.get_mesh], the per-vertex colors from the
    atlas' RGB lookup table, and the units from the atlas coordinates.

    Every requested region is merged into a single surface layer, each region keeping its
    own atlas color. The layer is named after the region (its full name and acronym) when
    a single region is requested, and after the requested acronyms otherwise.

    Parameters
    ----------
    atlas : xarray.Dataset
        Atlas Dataset with an `.atlas` accessor, as returned by
        [`fetch_brainglobe_atlas`][confusius.datasets.fetch_brainglobe_atlas] or
        [`load_atlas`][confusius.io.load_atlas].
    regions : int or str or sequence of int or str
        One or more regions, each given as a structure index or acronym.
    sides : {"left", "right", "both"} or sequence thereof, default: "both"
        Hemisphere filter. Pass a scalar to apply the same side to all regions, or a
        sequence of the same length as `regions` for per-region control.
        Faces are kept only when all three of their vertices survive, so the cut face is
        not closed.
    clip : bool, default: True
        Whether to clip the mesh to the reference grid, forwarded to
        [`get_mesh`][confusius.atlas.AtlasAccessor.get_mesh].
    values : (N,) or (N, T) numpy.ndarray, optional
        Per-vertex scalar values, over the concatenated vertices of all requested regions,
        used to color the surface through the layer's colormap. If not provided, each
        region is drawn in the atlas' designated color for it.
    viewer : napari.Viewer, optional
        Existing napari viewer to add the layer to. If not provided, a new viewer
        is created.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    **layer_kwargs
        Additional keyword arguments passed through to
        [`plot_surface`][confusius.plotting.plot_surface] and on to
        `napari.Viewer.add_surface` (e.g. `colormap`, `name`, `opacity`, `shading`).
        Passing `name`, `colormap`, `vertex_colors`, `axis_labels`, or `units`
        explicitly overrides the value derived from the atlas.

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance with the surface layer added.
    layer : napari.layers.Surface
        The surface layer added to the viewer.

    Raises
    ------
    KeyError
        If any requested region is not found in the atlas.
    ValueError
        If `atlas` is not a well-formed atlas, if `sides` is a sequence whose length does
        not match `regions`, or if a region's mesh file cannot be located.

    Examples
    --------
    >>> import confusius as cf
    >>> atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_25um")
    >>> # Overlay a region mesh, in its atlas assigned color, on the reference template.
    >>> viewer, _ = cf.plotting.plot_napari(atlas.atlas.reference)
    >>> viewer, layer = cf.plotting.plot_atlas_mesh(atlas, "VISp", viewer=viewer)

    >>> # Several regions at once, each in its own atlas color, one on a single side.
    >>> viewer, layer = cf.plotting.plot_atlas_mesh(
    ...     atlas, ["VISp", "AUDp", "MOp"], sides=["left", "both", "both"]
    ... )
    """
    validate_atlas(atlas)
    mesh_dict = get_atlas_mesh(atlas, regions, sides, clip=clip)
    structures = atlas.atlas.structures

    # get_atlas_mesh suffixes one-hemisphere keys with _L/_R; structure lookups need the
    # bare acronym.
    acronyms = {
        label: label if label in structures.acronym_to_id_map else label[:-2]
        for label in mesh_dict
    }

    vertices_parts, faces_parts, colors_parts = [], [], []
    n_vertices = 0
    for label, (region_vertices, region_faces) in mesh_dict.items():
        # Offset each region's faces by the running vertex count so the concatenated
        # meshes keep pointing at their own vertices.
        faces_parts.append(region_faces + n_vertices)
        vertices_parts.append(region_vertices)
        rgb = np.asarray(structures[acronyms[label]]["rgb_triplet"]) / 255.0
        colors_parts.append(np.tile(rgb, (len(region_vertices), 1)))
        n_vertices += len(region_vertices)
    mesh = (np.concatenate(vertices_parts), np.concatenate(faces_parts))
    vertex_colors = np.concatenate(colors_parts)

    if "colormap" not in layer_kwargs and values is None:
        layer_kwargs.setdefault("vertex_colors", vertex_colors)

    # Mesh vertices live in world space, so the layer's axes are the world z/y/x
    # coordinates, not the atlas's native k/j/i voxel dims.
    all_units: list[str | None] = [
        atlas.coords[dim].attrs.get("units") if dim in atlas.coords else None
        for dim in WORLD_DIMS
    ]
    layer_kwargs.setdefault("axis_labels", list(WORLD_DIMS))

    if any(u is not None for u in all_units):
        layer_kwargs.setdefault("units", all_units)

    labels = list(mesh_dict)
    if len(labels) == 1:
        name = f"{structures[acronyms[labels[0]]]['name']} ({labels[0]})"
    else:
        # The layer list is narrow, so a long region list is truncated rather than
        # pushing the rest of the name out of view.
        head = ", ".join(labels[:3])
        name = head if len(labels) <= 3 else f"{head}, +{len(labels) - 3}"
    layer_kwargs.setdefault("name", name)

    return plot_surface(
        mesh,
        values=values,
        viewer=viewer,
        show_scale_bar=show_scale_bar,
        **layer_kwargs,
    )


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
    world coordinate frame regardless of voxel spacing or data origin.

    Examples
    --------
    >>> import confusius as cf
    >>> pwd = cf.load("power_doppler.nii.gz")
    >>> # Display the time-averaged image and add an interactive Labels layer.
    >>> viewer, labels_layer = draw_napari_labels(pwd.mean("time"))
    >>> # … paint labels in the viewer …
    >>> # Convert painted labels to an integer label map DataArray.
    >>> label_map = labels_from_layer(labels_layer, pwd.mean("time"))
    """
    data = ensure_voxeldata(data)
    viewer, image_layer = plot_napari(data, viewer=viewer, **kwargs)
    display_data = cast(xr.DataArray, image_layer.metadata["xarray"])

    all_dims = [str(dim) for dim in display_data.dims]
    time_dim = TIME_DIM if TIME_DIM in all_dims else None
    spatial_dims = [dim for dim in all_dims if dim != time_dim]
    spatial_indices = [all_dims.index(dim) for dim in spatial_dims]
    scale, translate, axis_labels, all_units, _, _ = get_napari_layer_geometry(
        display_data
    )

    labels_array = np.zeros(
        tuple(display_data.sizes[dim] for dim in spatial_dims), dtype=np.int32
    )
    labels_kwargs: dict[str, Any] = {
        "scale": [scale[i] for i in spatial_indices],
        "translate": [translate[i] for i in spatial_indices],
        "axis_labels": [axis_labels[i] for i in spatial_indices],
        "metadata": {"xarray": display_data, "source_xarray": data},
    }
    units = [all_units[i] for i in spatial_indices]
    if any(unit is not None for unit in units):
        labels_kwargs["units"] = units

    labels_layer = viewer.add_labels(  # type: ignore
        labels_array, name=labels_layer_name, **labels_kwargs
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
        Stacked integer VoxelData array where the `mask` coordinate holds each unique
        non-zero label integer. Each layer has values `m` where the user painted label
        `m` and `0` elsewhere. This format is directly compatible with
        [`extract_with_labels`][confusius.extract.extract_with_labels],
        [`plot_contours`][confusius.plotting.plot_contours], and
        [`VolumePlotter.add_contours`][confusius.plotting.VolumePlotter.add_contours],
        and can be sliced by label (e.g. `label_map.sel(mask=2)`) for per-label display.
        The `attrs` dict carries:

        - `"long_name"`: "Drawn label map"
        - `"labels_layer_name"`: name of the source napari layer.
        - `"rgb_lookup"`: `dict[int, list[int]]` mapping each non-zero label to its
          `[r, g, b]` color (0–255) as painted in napari.

    Raises
    ------
    ValueError
        If `labels_layer` does not contain any non-zero labels.

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
    ('mask', 'k', 'j', 'i')
    >>> # Slice a single label for display alongside a seed map.
    >>> label_map.sel(mask=2)
    >>> # Use the label map for region-based analysis.
    >>> from confusius.extract import extract_with_labels
    >>> signals = extract_with_labels(pwd, label_map)
    """
    data = ensure_voxeldata(data)

    label_array = np.asarray(labels_layer.data)
    all_dims = [str(dim) for dim in data.dims]
    time_dim = TIME_DIM if TIME_DIM in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]
    if label_array.shape != tuple(data.sizes[dim] for dim in spatial_dims):
        metadata_data = getattr(labels_layer, "metadata", {}).get("xarray")
        if isinstance(metadata_data, xr.DataArray):
            data = ensure_voxeldata(metadata_data)
            all_dims = [str(dim) for dim in data.dims]
            time_dim = TIME_DIM if TIME_DIM in all_dims else None
            spatial_dims = [d for d in all_dims if d != time_dim]

    # Copy auxiliary spatial coordinates only. Native voxel coords and `pose` are
    # passed through `create_voxeldata`; derived `z`/`y`/`x` are rebuilt from the index.
    aux_coords = {
        name: coord
        for name, coord in data.coords.items()
        if set(coord.dims).issubset(spatial_dims)
        and name not in WORLD_DIMS
        and name not in CORE_DIMS
    }

    # Build a color lookup from the napari layer so downstream consumers
    # (plot_napari, VolumePlotter.add_volume, add_contours) can render each
    # label with exactly the color the user painted in the viewer.
    # get_color() returns RGBA in [0, 1] for any non-zero label and works for
    # both the default CyclicLabelColormap and any DirectLabelColormap.
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels != 0]
    rgb_lookup: dict[int, list[int]] = {}
    for label in unique_labels:
        label_id = np.int64(label).item()
        rgba = labels_layer.get_color(label_id)
        if rgba is not None:
            # Store 0-255 RGB (drop alpha) to match the atlas annotation convention.
            rgb_lookup[label_id] = [round(c * 255) for c in rgba[:3]]

    # Build one layer per label so the output matches the stacked mask format
    # returned by the atlas accessor's get_masks: dims=["mask", *spatial_dims] with the
    # mask coordinate holding integer label IDs. This allows per-label slicing
    # (e.g. label_map.sel(mask=2)) and is directly accepted by
    # extract_with_labels, plot_contours, and add_contours.
    if len(unique_labels) == 0:
        raise ValueError("labels_layer does not contain any non-zero labels.")

    layers = [np.where(label_array == k, k, 0).astype(np.int32) for k in unique_labels]
    stacked = np.stack(layers, axis=0)

    label_map = create_voxeldata(
        stacked,
        dims=["mask", *spatial_dims],
        extra_coords={"mask": unique_labels.astype(np.int32)},
        pose=data.coords.get(POSE_DIM) if POSE_DIM in spatial_dims else None,
        k=data.coords.get("k") if "k" in spatial_dims else None,
        j=data.coords.get("j") if "j" in spatial_dims else None,
        i=data.coords.get("i") if "i" in spatial_dims else None,
        voxel_to_world=get_voxel_to_world_affine(data),
        units=get_voxel_to_world_units(data),
        attrs={
            "long_name": "Drawn label map",
            "labels_layer_name": labels_layer.name,
            "rgb_lookup": rgb_lookup,
        },
    )
    return label_map.assign_coords(aux_coords) if aux_coords else label_map
