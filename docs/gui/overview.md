---
icon: lucide/layout-dashboard
---

# ConfUSIus Graphical Interface

ConfUSIus ships a [napari](https://napari.org) plugin that provides an interactive
graphical interface for fUSI data exploration, signals inspection, and quality
control—no scripting required.

!!! question "New to napari?"
    This guide only covers ConfUSIus-specific features. If you are new to napari, the
    official [napari tutorials](https://napari.org/dev/tutorials/index.html) are a great
    starting point for learning the viewer, layer system, and annotation tools. napari
    also has a rich [plugin hub](https://napari-hub.org) with plugins for loading videos,
    multimodal recordings, and much more—all of which can be used alongside ConfUSIus.

![ConfUSIus plugin overview](../images/gui/plugin-signals.png)

## Launching the Plugin

The `confusius` command is available in any Python environment where ConfUSIus is
installed—see the [Installation guide](../user-guide/installation.md) for details.
Alternatively, [`uvx`](https://docs.astral.sh/uv/guides/tools/) can fetch and run
ConfUSIus without a prior install step.

There are two ways to start the plugin:

=== "From inside napari"

    Open napari, then go to **Plugins > ConfUSIus**. The widget docks on the right side
    of the viewer automatically.

=== "From the terminal"

    Run the `confusius` command. This opens napari with the ConfUSIus widget ready to
    use:

    ```bash
    confusius
    ```

    With `uvx`, no prior installation is needed:

    ```bash
    uvx -p 3.13 confusius
    ```

The widget contains three collapsible panels—[Data I/O](plugin.md#data-io-panel),
[Signals](plugin.md#signals-panel), and [QC](plugin.md#qc-panel)—that can each be
expanded or collapsed independently. If you want a quick walkthrough inside napari
itself, click **Take a Tour** in the upper-right corner of the sidebar header.

!!! tip "Running napari programmatically"
    If you prefer to open napari from a Python script or Jupyter notebook, see the
    [Visualization](../user-guide/visualization.md#interactive-exploration-with-napari)
    guide, which covers [`plot_napari`][confusius.plotting.plot_napari] and related
    functions.

## Opening Data

### Recommended: Data Panel and CLI

The recommended way to open files is through the [Data Panel](plugin.md#data-panel) or
by passing a path to the `confusius` command on launch:

```bash
confusius path/to/data.zarr
```

Both routes use [`confusius.load()`][confusius.load] under the hood, which produces a
fully-labeled DataArray with named dimensions, physical coordinates, and all file
metadata preserved. The DataArray is attached to the layer and used automatically by the
Signals and QC panels. By default the full array is loaded into memory for
responsive time scrubbing. For files that don't fit in RAM, lazy loading keeps the array
Dask-backed—the layer appears instantly and slices are read from disk on demand. Enable
it via the **Load lazily** checkbox in the Data Panel, or with the `--lazy` flag on the
CLI:

```bash
confusius --lazy path/to/data.zarr
```

### Alternative: napari file readers

ConfUSIus also registers native napari file readers for NIfTI, Iconeus SCAN, and Zarr
files. These let you open files by dragging them onto the canvas or using **File >
Open** (++ctrl+o++) without switching to the Data Panel.

!!! warning "Axis labels are not propagated to the viewer"
    Due to a current napari limitation, files opened through the native readers have
    axis labels stored on the layer but not propagated to `viewer.dims.axis_labels`.
    This means dimension names will not appear in the napari status bar. Use the Data
    Panel or the `confusius` command instead to get correct axis labels everywhere.

## Saving Data

### Recommended: Data I/O Panel

The [Data I/O Panel](plugin.md#saving-data) save section lets you export any layer to
NIfTI or Zarr. Select the layer, optionally pick a template layer to borrow physical
coordinates from, enter an output path, and click **Save**.

### Alternative: napari file writers

ConfUSIus also registers native napari writers for NIfTI and Zarr files. These let you
save any image or labels layer via **File → Save Selected Layer** (++ctrl+s++) without
switching to the Data I/O Panel. Note that the native writer only supports saving the
raw array data, so physical coordinates and metadata will not be preserved if they're
not already present on the layer.

## Next Steps

- [Using the Plugin](plugin.md): a walkthrough of each panel, including loading and
  saving.
- [Visualization](../user-guide/visualization.md): programmatic napari and Matplotlib
  plotting from Python.
- [Quality Control](../user-guide/quality-control.md): background and interpretation of
  the QC metrics computed by the plugin.
- [napari tutorials](https://napari.org/dev/tutorials/index.html): official guides
  covering the viewer, layer system, and annotation tools.
- [napari hub](https://napari-hub.org): community plugins for loading additional data
  types and extending the viewer.
