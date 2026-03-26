---
icon: lucide/app-window
---

# Using the Plugin

The ConfUSIus sidebar contains three collapsible panels. Each panel operates
independently and can be expanded or collapsed by clicking its header.

## Data I/O Panel

The Data I/O Panel handles both loading and saving fUSI files without leaving the
viewer.

![ConfUSIus Data I/O panel](../images/gui/plugin-data-io.png)

### Loading data

Click **Browse** to pick a file—it loads immediately on selection. Or paste a path
directly in the text field and press ++enter++. Enable **Load lazily** beforehand to
keep the array Dask-backed for large files. A progress bar animates during loading, and
any error is reported in the napari notification bar.

### Saving data

1. Select the layer to save from the **Save layer** dropdown.
2. Optionally select a layer in the **Coordinates from** dropdown to borrow its physical
   coordinates and attributes. This is useful when saving a labels layer drawn on top of
   a fUSI image: selecting the image layer as the template preserves the full physical
   coordinate system. If the labels layer has fewer dimensions than the template (e.g. a
   3D labels layer against a 4D image), the trailing spatial dimensions are used
   automatically.
3. Type an output path or click **Browse**. The format is inferred from the extension:
   `.nii` / `.nii.gz` for NIfTI and `.zarr` for Zarr.
4. Click **Save**. A notification confirms success.

Three save modes are applied automatically depending on what is available:

| Mode | When applied |
|------|-------------|
| **Direct** | The layer was loaded via ConfUSIus (DataArray in metadata). Saved verbatim, all coordinates and attributes preserved. |
| **Template** | A template layer is selected. Coordinates are borrowed from the template DataArray. |
| **Reconstruct** | No template and no DataArray in metadata (e.g. a freshly drawn labels layer). Coordinates are reconstructed from the napari layer state (`scale`, `translate`, `axis_labels`). |

## Signals Panel

The Signals Panel plots signals extracted from image layers along any non-spatial
dimension (time, lag, feature, etc.). The plot appears in a bottom dock that is created
the first time you click **Show Signal Plot**.

### Choosing a data source

Pick one of three source modes in the **Source** group:

**Mouse**
: Hold ++shift++ and move the mouse over the canvas. The plot updates live with the
  single-voxel signal at the cursor position.

![ConfUSIus Signals panel — hover mode](../images/gui/plugin-signals.png)

**Points**
: Select a Points layer from the dropdown (or click **+** to create one). Each point
  is plotted as a separate line colored by its face color. Add or remove points in
  napari and the plot updates automatically.

![ConfUSIus Signals panel — points mode](../images/gui/plugin-signals-points.png)

**Labels**
: Select a Labels layer from the dropdown (or click **+** to create one). The mean
  signal is extracted for each distinct non-zero label and plotted as a separate line,
  colored by the label's color in the napari colormap. This is useful for quickly
  comparing region-averaged signals after painting ROIs with napari's brush tool.

![ConfUSIus Signals panel — labels mode](../images/gui/plugin-signals-labels.png)

In Points and Labels modes, the **Reference** dropdown selects which image layer to
extract signals from. It defaults to **All image layers**, which plots each layer as a
separate line (distinguished by line style).

### Axis parameters

| Option | Description |
|--------|-------------|
| ***x*-axis** | Choose which non-spatial dimension to plot on the horizontal axis. Defaults to `time` when available, otherwise the first non-spatial dimension. |
| ***y* min / *y* max** | Manual limits for the vertical axis (disabled while autoscale is on). |
| **Autoscale *y*-axis** | When enabled, the vertical axis rescales to fit the data automatically. Disabling it captures the current limits so you can fine-tune them. |

### Display options

| Option | Description |
|--------|-------------|
| **Show grid** | Show or hide the background grid. |
| **Show x-axis cursor** | Draw a vertical line on the plot that follows the napari dimension slider for the selected *x*-axis dimension. |
| **Z-score signal** | Normalize each signal to zero mean and unit variance before plotting. The *y*-axis label changes from "Intensity" to "Z-score". |

### Managing signals

Click **Manage Signals** to open a floating dialog where you can customize both live
signals (from the current source mode) and imported signals:

- **Rename**: Double-click a signal's name to edit it.
- **Recolor**: Click the color swatch to pick a new color. Changes are synced back to
  the napari layer (point face color or label colormap).
- **Show / Hide**: Toggle individual signal visibility with the checkbox.

### Importing and exporting signals

**Import**
: In the Manage Signals dialog, click **Import** to load signals from a CSV or TSV
  file. The file must contain a column whose header matches the current *x*-axis
  dimension name (e.g. `time`) plus one or more numeric value columns. Each value
  column becomes a separate signal overlaid on the plot.

**Export**
: Click the **Export** button in the plot toolbar to export all currently plotted
  signals—both live and imported—to a CSV or TSV file.

## QC Panel

The QC Panel computes quality control metrics for a selected image layer.

![ConfUSIus QC panel](../images/gui/plugin-qc.png)

Select a layer from the **Layer** dropdown, check the metrics you want, and click
**Compute**.

=== "Temporal metrics"

    Temporal metrics are rendered as plots in the bottom dock (the same dock used by the
    Signals Panel, in separate tabs). Computed plots are cached and survive dock
    closure: closing and reopening the bottom dock restores the last computed result.

    **DVARS**
    : Plots the standardized temporal derivative of variance over time. A vertical cursor
      follows the napari time slider. See [DVARS](../user-guide/quality-control.md#dvars)
      for interpretation.

    **Carpet plot**
    : Displays the full voxel time series as a 2D raster (time × voxels). See [Carpet
      Plot](../user-guide/quality-control.md#carpet-plot) for interpretation.

=== "Spatial metrics"

    Spatial map metrics are added as new image layers in the napari layer list, with
    correct physical scale and origin preserved.

    **CV**
    : Coefficient of variation map.

    **tSNR**
    : Temporal signal-to-noise ratio map.

    !!! warning "Prefer CV over tSNR for fUSI power Doppler data"
        tSNR is misleading for power Doppler: low-signal regions such as gel layers and
        shadow zones behind the skull can appear bright. CV correctly highlights regions
        with high temporal variability. See the [Quality Control
        guide](../user-guide/quality-control.md#temporal-snr) for a full explanation.
