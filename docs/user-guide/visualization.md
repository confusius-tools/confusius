---
icon: lucide/wallpaper
---

# Visualization

ConfUSIus provides tools for both **interactive exploration** and **static figure
generation**:

| Tool | Backend | Best for |
|---|---|---|
| [`plot_napari`][confusius.plotting.plot_napari] / [`.fusi.plot.napari()`][confusius.xarray.FUSIPlotAccessor.napari] | Napari | Interactive exploration of 3D+t datasets |
| [`plot_volume`][confusius.plotting.plot_volume] / [`.fusi.plot.volume()`][confusius.xarray.FUSIPlotAccessor.volume] | Matplotlib | Static slice grids |
| [`plot_contours`][confusius.plotting.plot_contours] / [`.fusi.plot.contours()`][confusius.xarray.FUSIPlotAccessor.contours] | Matplotlib | Atlas or mask contour overlays |
| [`plot_carpet`][confusius.plotting.plot_carpet] / [`.fusi.plot.carpet()`][confusius.xarray.FUSIPlotAccessor.carpet] | Matplotlib | Voxel time-series raster (quality control) |

All functions accept DataArrays and use physical coordinates for axis scaling
automatically. The [Xarray accessor](xarray.md) variants (`.fusi.plot.*`) offer a more
concise syntax; both call the same underlying functions.

## Interactive Exploration with Napari

ConfUSIus relies on [Napari](https://napari.org) for interactive visualization of 3D+t
fUSI data. Napari handles large datasets efficiently through lazy loading, allowing you
to explore even beamformed IQ data without running out of memory. You may find it
helpful to read through the [tour of the Napari
viewer](https://napari.org/dev/getting_started/viewer.html) to familiarize yourself with
the controls and features.

### Basic Usage

=== "Xarray accessor"

    ```python
    import xarray as xr
    import confusius

    pwd = xr.open_zarr("sub-01_task-rest_pwd.zarr")["power_doppler"]
    viewer = pwd.fusi.plot.napari()
    ```

=== "Function API"

    ```python
    import xarray as xr
    import confusius as cf

    pwd = xr.open_zarr("sub-01_task-rest_pwd.zarr")["power_doppler"]
    viewer = cf.plotting.plot_napari(pwd)
    ```

This opens a Napari viewer with a scale bar, colorbar, and correct physical scaling
across axes. The viewer is fully interactive: you can zoom, pan, and scroll through time
and elevation slices with the sliders or mouse wheel.

![Napari viewer with a 2D+t power Doppler dataset](../images/visualization/napari-overview.png)

!!! tip "Using Napari's annotation tools and plugins"
    Napari's annotation tools let you [draw regions of
    interest](https://napari.org/dev/howtos/layers/shapes.html) and [place
    markers](https://napari.org/dev/howtos/layers/points.html). These annotations can be
    saved and loaded for later analysis. Additionally, the [Napari
    Hub](https://napari-hub.org/) hosts hundreds of plugins that extend functionality —
    from segmentation algorithms to integration with other imaging modalities like
    microscopy or MRI.

### Napari's Parameters

By default Napari auto-scales contrast to the data range. For power Doppler, working
in decibel scale with explicit limits is often more informative:

```python
# dB-scaled power Doppler with fixed contrast window.
viewer = cf.plotting.plot_napari(
    pwd.fusi.scale.db(),
    contrast_limits=(-30, 0),
    colormap="hot",
)
```

`contrast_limits`, `colormap`, and any other keyword arguments are forwarded directly to
[`napari.imshow`](https://napari.org/stable/api/napari.html#napari.imshow).

### Overlaying an Atlas as a Labels Layer

Napari's **labels** layer renders integer-labeled masks as filled or contoured regions,
ideal for visualizing an atlas alongside your power Doppler data. Here, we use
[BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html)
to look up the official Allen atlas colors, then add the labels layer directly on the
existing viewer:

```python
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.utils.colormaps import DirectLabelColormap
import confusius as cf
import numpy as np
import xarray as xr

# Load power Doppler mean volume and open viewer.
mean_vol = pwd.mean("time").compute()
viewer = cf.plotting.plot_napari(
    mean_vol.fusi.scale.db(),
    contrast_limits=(-15, 0),
)

# Build RGBA colormap from the Allen atlas.
atlas = BrainGlobeAtlas("allen_mouse_25um")
allen_colors = {
    s["id"]: tuple(c / 255 for c in s["rgb_triplet"])
    for s in atlas.structures_list
}
label_colormap = DirectLabelColormap(color_dict={
    0: np.zeros(4),       # Transparent background.
    None: np.zeros(4),    # Transparent default for unlisted labels.
    **{lbl: list(rgb) + [0.7] for lbl, rgb in allen_colors.items()},
})

# Load atlas mask and add as a labels layer on the existing viewer.
atlas_mask = xr.open_zarr("allen_atlas_mask.zarr")["atlas"]
viewer = atlas_mask.fusi.plot.napari(
    viewer=viewer,
    layer_type="labels",
    colormap=label_colormap,
    name="Allen atlas",
    opacity=0.6,
)
```

![Napari viewer with Allen atlas Labels layer overlaid](../images/visualization/napari-labels.png)

!!! tip "Adding Multiple Image Layers"
    The `viewer` argument works for any `plot_napari` call, letting you load two image
    datasets into the same viewer for direct comparison; for example, before and after
    motion correction:

    ```python
    viewer = pwd_before.fusi.plot.napari(name="Before correction")
    viewer = pwd_after.fusi.plot.napari(viewer=viewer, name="After correction")
    ```

### Slicing Across Different Spatial Dimensions

By default, Napari shows the last two spatial dimensions as a 2D plane and the remaining
dimensions (e.g., `time`, `z`) as sliders. If you prefer a different default slicing
axis—for example slicing along `y` (depth) instead of `z` (elevation)—use
`dim_order` to reorder the spatial axes:

```python
# Swap z and y: depth becomes the slider axis.
viewer = pwd_3d.fusi.plot.napari(dim_order=("y", "z", "x"))
```

### 3D Rendering

For volumetric datasets, Napari can render the data in full 3D using volume rendering
(accessible by clicking the second icon in the bottom-left controls). In the 3D view you
can drag to orbit, scroll to zoom, and use the Napari controls to adjust the rendering.

![3D orbit of the angiography volume](../images/visualization/napari-3d-orbit.gif)

## Static Volume Plots

[`plot_volume`][confusius.plotting.plot_volume] generates a static Matplotlib grid of 2D
slices—one panel per coordinate along the chosen slicing dimension. It is the standard
tool in ConfUSIus for generating static figures of 3D volumes, functional activation
maps, or 3D angiography data. 

### Basic Usage

=== "Xarray accessor"

    ```python
    import xarray as xr
    import confusius

    pwd = xr.open_zarr("sub-01_task-rest_pwd.zarr")["power_doppler"]

    # All elevation slices in an auto-sized grid.
    plotter = pwd.fusi.plot.volume()
    ```

=== "Function API"

    ```python
    import xarray as xr
    import confusius as cf

    pwd = xr.open_zarr("sub-01_task-rest_pwd.zarr")["power_doppler"]

    # All elevation slices in an auto-sized grid.
    plotter = cf.plotting.plot_volume(pwd)
    ```

The function returns a [`VolumePlotter`][confusius.plotting.VolumePlotter] object that
manages the underlying Matplotlib [`Figure`][matplotlib.figure.Figure] and
[`Axes`][matplotlib.axes.Axes] and supports overlay operations (see [Overlaying
Contours](#overlaying-contours)).

![Mean power Doppler volume](../images/visualization/plot-volume-grid-light.png#only-light)
![Mean power Doppler volume](../images/visualization/plot-volume-grid-dark.png#only-dark)

When the data has multiple slices along the sliced dimension, `plot_volume` lays them
out automatically in an approximately square grid:

```python
angio = xr.open_zarr("sub-01_angio.zarr")["angio"]

plotter = angio.fusi.plot.volume(slice_mode="z", show_colorbar=False)
```

![3D angiography volume shown as an elevation-slice grid](../images/visualization/plot-volume-3d-light.png#only-light)
![3D angiography volume shown as an elevation-slice grid](../images/visualization/plot-volume-3d-dark.png#only-dark)

### Selecting Slices and Colormap

By default all coordinates along `slice_mode` are shown. Use `slice_coords` to pick
specific ones and `cmap`/`vmin`/`vmax` to control the colormap and contrast. The grid
layout can be controlled using `nrows` and `ncols`, or by specifying axes directly with
`axes` (see the [API reference](../api/plotting.md#confusius.plotting.plot_volume) for
details).

```python
plotter = angio.fusi.plot.volume(
    nrows=1,
    slice_mode="z",
    slice_coords=(1.6, 2.4, 3.2),
    cmap="inferno",
    vmin=-30,
    vmax=0,
    show_colorbar=False,
)
```

![Three selected elevation slices of the angiography volume](../images/visualization/plot-sliced-volume-3d-light.png#only-light)
![Three selected elevation slices of the angiography volume](../images/visualization/plot-sliced-volume-3d-dark.png#only-dark)

### Thresholding

For functional activation maps or data where you want to suppress noise, `threshold`
sets a cutoff value. Subthreshold voxels are rendered transparently:

```python
# Hide values where |data| < 3.0 (noise floor suppression).
plotter = stat_map.fusi.plot.volume(
    slice_mode="z",
    threshold=3.0,
    threshold_mode="lower",
    cmap="RdBu_r",
    vmin=-6,
    vmax=6,
)
```

`threshold_mode="upper"` masks values *above* the threshold instead—useful for
removing saturation artifacts or thresholding decibel-scaled data.

### Saving and Closing

```python
plotter = pwd.fusi.plot.volume(slice_mode="z", cmap="hot")
plotter.savefig("sub-01_power_doppler_slices.png", dpi=150)
plotter.close()
```

Pass any keyword argument accepted by
[`matplotlib.figure.Figure.savefig`][matplotlib.figure.Figure.savefig] (e.g.,
`bbox_inches="tight"`, `transparent=True`).

## Overlaying Contours

Atlas outlines or regions of interest (ROIs) boundaries can be drawn on top of a volume
plot to provide anatomical context. ConfUSIus represents masks as **integer-labeled
DataArrays** where 0 is background and each positive integer identifies a distinct
region.

### Overlaying Contours on a Volume

The most common use case is to draw atlas outlines on top of a fUSI volume.
[`plot_volume`][confusius.plotting.plot_volume] returns a
[`VolumePlotter`][confusius.plotting.VolumePlotter] that remembers the
coordinate-to-axis mapping; calling
[`add_contours`][confusius.plotting.VolumePlotter.add_contours] on it draws outlines
on the matching panels. Here, we use [BrainGlobe Atlas
API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html) to color each
region with its official Allen atlas color:

```python
from brainglobe_atlasapi import BrainGlobeAtlas

atlas = BrainGlobeAtlas("allen_mouse_25um")
allen_colors = {
    s["id"]: tuple(c / 255 for c in s["rgb_triplet"])
    for s in atlas.structures_list
}

# Step 1: display an average power Doppler volume.
plotter = pwd.fusi.scale.db().fusi.plot.volume(
    slice_mode="z",
    cmap="gray",
    vmin=-15,
    vmax=0,
    cbar_label="Power Doppler (dB)",
)

# Step 2: overlay atlas contours with Allen colors.
plotter.add_contours(atlas_mask, colors=allen_colors)
plotter.savefig("sub-01_functional_with_atlas.png", dpi=150)
```

![Power Doppler volume with Allen atlas region contours overlaid](../images/visualization/volume-with-contours-light.png#only-light)
![Power Doppler volume with Allen atlas region contours overlaid](../images/visualization/volume-with-contours-dark.png#only-dark)

Coordinate matching is done in physical units, matching contour coordinates with those
of the previously plotted volume. Slices present in the mask but absent from the volume
are skipped with a warning.

## Carpet Plots

A **carpet plot** (also called a grayplot or Power plot[^power2017]) displays every
voxel's time-series as a row in a 2D raster image with time on the x-axis. Z-scored by
default, it makes motion artifacts, global signal transients, and outlier volumes
immediately visible as vertical stripes or abrupt intensity changes.

Carpet plots are primarily used for quality control—for a deeper discussion of QC
metrics, see the [Quality Control](qc.md) guide.

### Basic Usage

=== "Xarray accessor"

    ```python
    import xarray as xr
    import confusius

    brain_mask = xr.open_zarr("brain_mask.zarr")["mask"]

    fig, ax = pwd.fusi.plot.carpet(mask=brain_mask)
    ```

=== "Function API"

    ```python
    import xarray as xr
    import confusius as cf

    brain_mask = xr.open_zarr("brain_mask.zarr")["mask"]

    fig, ax = cf.plotting.plot_carpet(pwd, mask=brain_mask)
    ```

Without a `mask`, all non-zero voxels are included. With a mask, only voxels where `mask
== True` (or `mask > 0` for integer-labeled masks) are shown. See the API reference of
[`plot_carpet`][confusius.plotting.plot_carpet] for more options.

![Carpet plot of power Doppler voxel time-series](../images/visualization/carpet-plot-light.png#only-light)
![Carpet plot of power Doppler voxel time-series](../images/visualization/carpet-plot-dark.png#only-dark)

## Next Steps

Now that you can visualize your data, you're ready for:

1. **[Registration](registration.md)**: Correct for motion and align acquisitions to an
   anatomical template.
2. **[Quality Control](qc.md)**: Assess data quality and identify artifacts.
3. **[Signal Processing](signal.md)**: Extract regional signals and apply denoising.

## API Reference

For full parameter documentation, see the [Plotting API reference](../api/plotting.md)
and the [Xarray Integration API reference](../api/xarray.md).

[^power2017]:
    Power, Jonathan D. "A Simple but Useful Way to Assess fMRI Scan Qualities."
    NeuroImage, vol. 154, July 2017, pp. 150–58. DOI.org (Crossref),
    <https://doi.org/10.1016/j.neuroimage.2016.08.009>.
