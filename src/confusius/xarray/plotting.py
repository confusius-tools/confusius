"""Xarray accessor for plotting."""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from confusius.plotting import (
    VolumePlotter,
    draw_napari_labels,
    labels_from_layer,
    plot_carpet,
    plot_composite,
    plot_contours,
    plot_napari,
    plot_stat_map,
    plot_volume,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image, Labels


class FUSIPlotAccessor:
    """Accessor for plotting fUSI data.

    This accessor provides convenient plotting methods for functional
    ultrasound imaging data, with specialized support for napari visualization.

    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to wrap.

    Examples
    --------
    >>> import xarray as xr
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> viewer, layer = data.fusi.plot.napari()
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def __call__(self, **kwargs) -> "tuple[Viewer, Image | Labels]":
        """Call the napari plotting method by default.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `napari`.
        """
        return self.napari(**kwargs)

    def napari(
        self,
        show_colorbar: bool = True,
        show_scale_bar: bool = True,
        dim_order: tuple[str, ...] | None = None,
        viewer: "Viewer | None" = None,
        layer_type: Literal["image", "labels"] = "image",
        **layer_kwargs,
    ) -> "tuple[Viewer, Image | Labels]":
        """Display data in napari viewer.

        Parameters
        ----------
        show_colorbar : bool, default: True
            Whether to show the colorbar. Only applies to image layers.
        show_scale_bar : bool, default: True
            Whether to show the scale bar.
        dim_order : tuple[str, ...], optional
            Dimension ordering for the spatial axes (last three dimensions). If not
            provided, the ordering of the last three dimensions in `data` is used.
        viewer : napari.Viewer, optional
            Existing napari viewer to add the layer to. If not provided, a new
            viewer is created.
        layer_type : {"image", "labels"}, default: "image"
            Type of layer to create. Use "image" for fUSI data and "labels" for
            ROI masks, segmentations, or other label data.
        **layer_kwargs
            Additional keyword arguments passed to the layer creation method.
            For image layers, if `data.attrs` contains `"cmap"` and `"colormap"`
            is not in `layer_kwargs`, the attribute is used as the colormap.

        Returns
        -------
        viewer : napari.Viewer
            The napari viewer instance with the layer added.
        layer : napari.layers.Image or napari.layers.Labels
            The layer added to the viewer.

        Notes
        -----
        If all spatial dimensions have coordinates, their spacing is used as the scale
        parameter for napari to ensure correct physical scaling. If any spatial dimension
        is missing coordinates, no scaling is applied. The spacing is computed as the
        median difference between consecutive coordinate values.

        For unitary dimensions (e.g., a single-slice elevation axis in 2D+t data), the
        spacing cannot be inferred from coordinates. In that case, the function looks for
        a `voxdim` attribute on the coordinate variable
        (`data.coords[dim].attrs["voxdim"]`) and uses it as the spacing. If no such
        attribute is found, unit spacing is assumed and a warning is emitted.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> viewer, layer = data.fusi.plot.napari()

        >>> # Custom contrast limits
        >>> viewer, layer = data.fusi.plot.napari(contrast_limits=(0, 100))

        >>> # Different dimension ordering (e.g., depth, elevation, lateral)
        >>> viewer, layer = data.fusi.plot.napari(dim_order=("y", "z", "x"))

        >>> # Add a second dataset as a new layer in an existing viewer
        >>> viewer, layer = data1.fusi.plot.napari()
        >>> viewer, layer = data2.fusi.plot.napari(viewer=viewer)

        >>> # Display ROI labels (e.g., segmentation mask)
        >>> roi_mask = xr.open_zarr("output.zarr")["roi_mask"]
        >>> viewer, layer = roi_mask.fusi.plot.napari(layer_type="labels")

        >>> # Overlay labels on existing image
        >>> viewer, layer = data.fusi.plot.napari()
        >>> viewer, layer = roi_mask.fusi.plot.napari(viewer=viewer, layer_type="labels")
        """
        return plot_napari(
            self._obj,
            show_colorbar=show_colorbar,
            show_scale_bar=show_scale_bar,
            dim_order=dim_order,
            viewer=viewer,
            layer_type=layer_type,
            **layer_kwargs,
        )

    def draw_napari_labels(
        self,
        labels_layer_name: str = "labels",
        viewer: "Viewer | None" = None,
        **plot_kwargs,
    ) -> "tuple[Viewer, Labels]":
        """Open napari to interactively paint integer labels over fUSI data.

        Displays the data as an image layer and adds an empty Labels layer on
        top. The user paints integer labels directly on the image using
        napari's brush tool. After painting, pass the returned Labels layer to
        [`labels_from_layer`][confusius.plotting.FUSIPlotAccessor.labels_from_layer]
        to obtain an integer label map in the same spatial coordinates as the
        data.

        Parameters
        ----------
        labels_layer_name : str, default: "labels"
            Name assigned to the Labels layer added to the viewer.
        viewer : napari.Viewer, optional
            Existing napari viewer to add layers to. If not provided, a new
            viewer is created.
        **plot_kwargs
            Additional keyword arguments forwarded to
            [`plot_napari`][confusius.plotting.plot_napari] for the image layer
            (e.g. `colormap`, `contrast_limits`).

        Returns
        -------
        viewer : napari.Viewer
            The napari viewer instance with the image and Labels layers.
        labels_layer : napari.layers.Labels
            The empty Labels layer initialised to zeros. After painting labels
            in the viewer, pass it to
            [`labels_from_layer`][confusius.plotting.FUSIPlotAccessor.labels_from_layer]
            to convert the paintings to an integer label map.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> pwd = xr.open_zarr("output.zarr")["power_doppler"].compute()
        >>> # Display time-averaged image with an interactive Labels layer.
        >>> viewer, labels_layer = pwd.mean("time").fusi.plot.draw_napari_labels()
        >>> # … paint labels in the viewer …
        >>> # Convert painted labels to an integer label map.
        >>> label_map = pwd.mean("time").fusi.plot.labels_from_layer(labels_layer)
        """
        return draw_napari_labels(
            self._obj,
            labels_layer_name=labels_layer_name,
            viewer=viewer,
            **plot_kwargs,
        )

    def labels_from_layer(
        self,
        labels_layer: "Labels",
    ) -> xr.DataArray:
        """Convert a napari Labels layer to an integer label map DataArray.

        Reads the integer array painted in `labels_layer` and wraps it in an
        [`xarray.DataArray`][xarray.DataArray] whose spatial dimensions and
        coordinates match those of the data.

        Parameters
        ----------
        labels_layer : napari.layers.Labels
            A Labels layer populated by the user (e.g. via
            [`draw_napari_labels`][confusius.plotting.FUSIPlotAccessor.draw_napari_labels]).
            Integer values identify distinct regions; zero is the background.

        Returns
        -------
        xarray.DataArray
            Integer DataArray with the same spatial dimensions and coordinates
            as the data. Zero values indicate background (unlabelled) voxels.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> pwd = xr.open_zarr("output.zarr")["power_doppler"].compute()
        >>> viewer, labels_layer = pwd.mean("time").fusi.plot.draw_napari_labels()
        >>> # … paint labels in the viewer …
        >>> label_map = pwd.mean("time").fusi.plot.labels_from_layer(labels_layer)
        >>> # Use the label map for region-based analysis.
        >>> from confusius.extract import extract_with_labels
        >>> signals = extract_with_labels(pwd, label_map)
        """
        return labels_from_layer(labels_layer, self._obj)

    def carpet(
        self,
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
        intensities as a 2D raster image with time on the x-axis and voxels on the
        y-axis. Each row represents one voxel's time series, typically standardized to
        z-scores.

        Parameters
        ----------
        mask : xarray.DataArray, optional
            Boolean mask with same spatial dimensions and coordinates as `data`.
            `True` values indicate voxels to include. If not provided, all non-zero
            voxels from the data are included.
        detrend_order : int, optional
            Polynomial order for detrending:

            - `0`: Remove mean (constant detrending).
            - `1`: Remove linear trend using least squares regression (default).
            - `2+`: Remove polynomial trend of specified order.

            If not provided, no detrending is applied.
        standardize : bool, default: True
            Whether to standardize each voxel's time series to z-scores.
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
            Base font size for text elements. Title uses `fontsize` directly;
            axis labels and colorbar label use `0.9 * fontsize`; tick labels use
            `0.85 * fontsize`. If not provided, uses the active Matplotlib defaults.
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
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> fig, ax = data.fusi.plot.carpet()

        >>> # With linear detrending.
        >>> fig, ax = data.fusi.plot.carpet(detrend_order=1)

        >>> # With mask.
        >>> mask = np.abs(data.isel(time=0)) > threshold
        >>> fig, ax = data.fusi.plot.carpet(mask=mask)
        """
        return plot_carpet(
            self._obj,
            mask=mask,
            detrend_order=detrend_order,
            standardize=standardize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            decimation_threshold=decimation_threshold,
            figsize=figsize,
            title=title,
            fontsize=fontsize,
            bg_color=bg_color,
            fg_color=fg_color,
            ax=ax,
        )

    def volume(
        self,
        slice_coords: list[Hashable] | None = None,
        slice_mode: str = "z",
        nrows: int | None = None,
        ncols: int | None = None,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        cmap: "str | Colormap | None" = None,
        norm: "Normalize | None" = None,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: "float | xr.DataArray | None" = None,
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
        axes: "npt.NDArray[Any] | None" = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot 2D slices of a volume as a matplotlib subplot grid.

        See [`confusius.plotting.plot_volume`][confusius.plotting.plot_volume] for full
        details.

        Parameters
        ----------
        slice_coords : list[collections.abc.Hashable], optional
            Coordinate values along `slice_mode` at which to extract slices.
            Slices are selected by nearest-neighbour lookup. If not provided,
            all coordinate values along `slice_mode` are used.
        slice_mode : str, default: "z"
            Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`,
            `"time"`). After slicing, each panel must be 2D.
        nrows : int, optional
            Number of rows in the subplot grid. If not provided, computed
            automatically together with `ncols` to produce a near-square layout.
        ncols : int, optional
            Number of columns in the subplot grid. If not provided, computed
            automatically together with `nrows`.
        threshold : float, optional
            Threshold applied to `|data|`. See `threshold_mode` for the
            masking direction. If not provided, no thresholding is applied.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Controls how `threshold` is applied:

            - `"lower"`: set pixels where `|data| < threshold` to NaN.
            - `"upper"`: set pixels where `|data| > threshold` to NaN.

        cmap : str or matplotlib.colors.Colormap, optional
            Colormap. When not provided, falls back to `data.attrs["cmap"]`
            if present, otherwise `"gray"`.
        norm : matplotlib.colors.Normalize, optional
            Normalization instance (e.g. `BoundaryNorm` for integer label
            maps). When not provided, falls back to `data.attrs["norm"]` if
            present. When a norm is active, `vmin` and `vmax` are ignored.
        vmin : float, optional
            Lower bound of the colormap. Defaults to the 2nd percentile. Ignored
            when `norm` is provided explicitly (that is, not just inherited from data
            attributes).
        vmax : float, optional
            Upper bound of the colormap. Defaults to the 98th percentile. Ignored
            when `norm` is provided explicitly (that is, not just inherited from data
            attributes).
        alpha : float or xarray.DataArray, optional
            Opacity of the image: a single scalar value, or a 3D DataArray sharing
            this DataArray's dims, shape, and coordinates (for independent
            per-slice, per-voxel opacity). A per-voxel opacity must be a DataArray,
            not a bare array, so it can be validated and aligned against this
            DataArray. If not provided, the colormap's own alpha channel is
            respected.
        show_colorbar : bool, default: True
            Whether to add a shared colorbar to the figure.
        cbar_label : str, optional
            Label for the colorbar.
        cbar_kwargs : dict, optional
            Additional keyword arguments forwarded to
            [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
            (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the
            colorbar when it spans a multi-panel grid, since the defaults are sized
            for a single axes.
        show_titles : bool, default: True
            Whether to display subplot titles showing the slice coordinate.
        show_axis_labels : bool, default: True
            Whether to display axis labels (with units when available).
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When
            `False`, overrides `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels and colorbar label use `0.9 * fontsize`; tick
            labels use `0.85 * fontsize`. If not provided, uses the active Matplotlib
            defaults.
        yincrease : bool, default: False
            Whether the y-axis increases upward (`True`) or downward (`False`).
        xincrease : bool, default: True
            Whether the x-axis increases to the right (`True`) or left
            (`False`).
        bg_color : str, default: "black"
            Background color for the figure and axes. Any matplotlib-compatible color
            string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
        fg_color : str, optional
            Color for text, labels, ticks, and spines. If not provided, derived
            automatically from `bg_color` using the WCAG relative luminance formula
            (white on dark backgrounds, black on light ones).
        figure : matplotlib.figure.Figure, optional
            Existing figure to draw into. If not provided, a new figure is
            created.
        axes : numpy.ndarray, optional
            Existing 2D array of `matplotlib.axes.Axes` to draw into. If not
            provided, new axes are created inside `figure`.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when `figure` is
            provided.

        Returns
        -------
        VolumePlotter
            Object managing the figure, axes, and coordinate mapping for overlays.

        Raises
        ------
        ValueError
            If `slice_mode` is not a dimension of the data.
        ValueError
            If the data is not 3D after squeezing unitary dimensions.
        ValueError
            If `axes` is provided but does not contain enough elements for all
            slices.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> data = xr.open_zarr("output.zarr")["power_doppler"]
        >>> plotter = data.fusi.plot.volume(slice_mode="z")

        >>> # dB-scale data with upper threshold to suppress far-field noise.
        >>> plotter = data.fusi.scale.db().fusi.plot.volume(
        ...     slice_mode="z",
        ...     threshold=-60,
        ...     threshold_mode="upper",
        ...     cmap="hot",
        ...     bg_color="black",
        ... )
        """
        return plot_volume(
            self._obj,
            slice_coords=slice_coords,
            slice_mode=slice_mode,
            nrows=nrows,
            ncols=ncols,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            show_colorbar=show_colorbar,
            cbar_label=cbar_label,
            cbar_kwargs=cbar_kwargs,
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
            dpi=dpi,
        )

    def contours(
        self,
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
        axes: "npt.NDArray[Any] | None" = None,
        **kwargs,
    ) -> "VolumePlotter":
        """Plot mask contours as a grid of 2D slice panels.

        Displays contour lines for each labeled region across a grid of subplots. See
        [`confusius.plotting.plot_contours`][confusius.plotting.plot_contours] for full
        details.

        Parameters
        ----------
        colors : dict[int | str, str] or str, optional
            Color specification for contour lines. A `dict` maps each label
            (integer index or region acronym string) to a color; a `str` applies
            one color to all regions. If not provided, colors are derived from
            `attrs["cmap"]` and `attrs["norm"]` when present, otherwise
            from the `tab10`/`tab20` colormap.
        linewidths : float, default: 1.5
            Width of contour lines in points.
        linestyles : str, default: "solid"
            Line style for contour lines (e.g. `"solid"`, `"dashed"`).
        slice_mode : str, default: "z"
            Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`).
            After slicing, each panel must be 2D.
        slice_coords : list[collections.abc.Hashable], optional
            Coordinate values along `slice_mode` at which to extract slices.
            Slices are selected by nearest-neighbour lookup. If not provided, all
            coordinate values along `slice_mode` are used.
        fontsize : float, optional
            Base font size for text elements. Subplot titles use `fontsize`
            directly; axis labels use `0.9 * fontsize`; tick labels use
            `0.85 * fontsize`. If not provided, uses the active Matplotlib defaults.
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
        axes : numpy.ndarray, optional
            Existing 2D array of [`matplotlib.axes.Axes`][matplotlib.axes.Axes] to draw
            into. If not provided, new axes are created inside `figure`.
        **kwargs
            Additional keyword arguments passed to
            [`matplotlib.axes.Axes.plot`][matplotlib.axes.Axes.plot].

        Returns
        -------
        VolumePlotter
            Object managing the figure, axes, and coordinate mapping for overlays.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> mask = xr.open_zarr("output.zarr")["roi_mask"]
        >>> plotter = mask.fusi.plot.contours(colors={1: "red", 2: "blue"})

        >>> # Overlay contours on an existing volume plot.
        >>> volume = xr.open_zarr("output.zarr")["power_doppler"]
        >>> plotter = volume.fusi.plot.volume(slice_mode="z")
        >>> plotter.add_contours(mask, colors="yellow")
        """

        return plot_contours(
            self._obj,
            colors=colors,
            linewidths=linewidths,
            linestyles=linestyles,
            slice_mode=slice_mode,
            slice_coords=slice_coords,
            fontsize=fontsize,
            yincrease=yincrease,
            xincrease=xincrease,
            bg_color=bg_color,
            fg_color=fg_color,
            figure=figure,
            axes=axes,
            **kwargs,
        )

    def composite(
        self,
        other: xr.DataArray,
        resample: bool = True,
        resample_kwargs: "dict[str, Any] | None" = None,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        normalize_strategy: Literal["per_volume", "per_slice", "shared"] = "per_volume",
        slice_coords: list[Hashable] | None = None,
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
        axes: "npt.NDArray[Any] | None" = None,
        nrows: int | None = None,
        ncols: int | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot a red/cyan composite of this volume against `other`.

        Self drives the red channel; `other` drives the cyan channel. See
        [`confusius.plotting.plot_composite`][confusius.plotting.plot_composite]
        for full details.

        Parameters
        ----------
        other : xarray.DataArray
            Second volume, plotted in cyan. When `resample=True`, resampled onto
            this DataArray's grid before blending.
        resample : bool, default: True
            Whether to resample `other` onto this DataArray's grid using an
            identity transform before blending. When `False`, the two arrays
            must already share dims and shape, and their coordinates must match
            within `rtol`/`atol`; once validated, `other`'s coordinates are
            replaced with this DataArray's so the two volumes share an exact
            coordinate frame downstream.
        resample_kwargs : dict, optional
            Extra keyword arguments forwarded to
            [`resample_like`][confusius.registration.resample_like] when
            `resample=True`. Ignored when `resample=False`.
        rtol : float, default: 1e-5
            Relative tolerance used to validate that this DataArray and `other`
            share coordinates when `resample=False`. Widen to accept
            acquisitions on slightly offset grids known to be equivalent.
            Ignored when `resample=True`.
        atol : float, default: 1e-8
            Absolute tolerance used to validate that this DataArray and `other`
            share coordinates when `resample=False`. Ignored when
            `resample=True`.
        normalize_strategy : {"per_volume", "per_slice", "shared"}, default: "per_volume"
            Intensity normalisation strategy.

            - `"per_volume"`: rescale each input to `[0, 1]` independently over its
              full volume.
            - `"per_slice"`: rescale each 2D slice independently.
            - `"shared"`: rescale both volumes together using a shared
              `[min, max]` range, preserving the absolute-intensity
              relationship between the two inputs.
        slice_coords : list[collections.abc.Hashable], optional
            Coordinate values along `slice_mode` at which to extract slices.
            Slices are selected by nearest-neighbour lookup. If not provided,
            all coordinate values from this DataArray are used.
        slice_mode : str, default: "z"
            Dimension along which to slice (e.g. `"x"`, `"y"`, `"z"`). After
            slicing, each panel must be 2D.
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
        yincrease : bool, default: False
            Whether the y-axis increases upward (`True`) or downward (`False`).
        xincrease : bool, default: True
            Whether the x-axis increases to the right (`True`) or left
            (`False`).
        bg_color : str, default: "black"
            Background color for the figure and axes. Any matplotlib-compatible
            color string (e.g. `"black"`, `"white"`, `"#1a1a2e"`).
        fg_color : str, optional
            Color for text, labels, ticks, and spines. If not provided, derived
            automatically from `bg_color` using the WCAG relative luminance
            formula (white on dark backgrounds, black on light ones).
        figure : matplotlib.figure.Figure, optional
            Existing figure to draw into. If not provided, a new figure is
            created.
        axes : numpy.ndarray, optional
            Existing 2D array of `matplotlib.axes.Axes` to draw into. If not
            provided, new axes are created inside `figure`.
        nrows : int, optional
            Number of rows in the subplot grid. If not provided, computed
            automatically.
        ncols : int, optional
            Number of columns in the subplot grid. If not provided, computed
            automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when `figure` is
            provided.

        Returns
        -------
        VolumePlotter
            Object managing the figure, axes, and coordinate mapping for
            overlays.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> fixed = xr.open_zarr("fixed.zarr")["power_doppler"]
        >>> moving = xr.open_zarr("moving.zarr")["power_doppler"]
        >>> plotter = fixed.fusi.plot.composite(moving)

        >>> # Same-grid comparison without resampling, with joint scaling.
        >>> plotter = fixed.fusi.plot.composite(
        ...     registered_moving, resample=False, normalize_strategy="shared"
        ... )
        """
        return plot_composite(
            self._obj,
            other,
            resample=resample,
            resample_kwargs=resample_kwargs,
            rtol=rtol,
            atol=atol,
            normalize_strategy=normalize_strategy,
            slice_coords=slice_coords,
            slice_mode=slice_mode,
            alpha=alpha,
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
        )

    def stat_map(
        self,
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
    ) -> "VolumePlotter":
        """Plot this statistical map, optionally over `bg_volume`.

        Self is the statistical map. See
        [`confusius.plotting.plot_stat_map`][confusius.plotting.plot_stat_map] for full
        details.

        Parameters
        ----------
        bg_volume : xarray.DataArray, optional
            Background anatomical volume, plotted underneath this DataArray. When
            `alpha` is not provided, this DataArray fully covers `bg_volume`
            wherever it has a value; `bg_volume` only shows through where this
            DataArray is masked out by `threshold`. Lower `alpha` to blend the two
            layers instead. Must share `slice_mode` and, after squeezing, the same
            display dimensions as this DataArray. If not provided, this DataArray is
            plotted on its own.
        slice_coords : list[collections.abc.Hashable], optional
            Coordinate values along `slice_mode` at which to extract slices. Slices
            are selected by nearest-neighbour lookup. If not provided, all coordinate
            values from `bg_volume` (or this DataArray when `bg_volume` is not
            provided) along `slice_mode` are used.
        slice_mode : str, default: "z"
            Dimension along which to slice (e.g., `"x"`, `"y"`, `"z"`, `"time"`).
            After slicing, each panel must be 2D.
        bg_kwargs : dict, optional
            Additional keyword arguments forwarded to
            [`plot_volume`][confusius.plotting.plot_volume] for the background layer
            (e.g. `cmap`, `vmin`, `vmax`, `norm`, `alpha`, `roi_labels`). Ignored when
            `bg_volume` is not provided. Layout and text styling (`slice_coords`,
            `slice_mode`, `show_titles`, `fontsize`, etc.) are controlled by this
            method's own parameters instead, so that both layers share consistent
            styling.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap for this DataArray. If not provided, the default depends on
            `auto_range` and the sign of this DataArray (see below); an explicit
            `cmap` is always used as-is regardless of `auto_range`.
        norm : matplotlib.colors.Normalize, optional
            Normalization instance (e.g. `TwoSlopeNorm`, `BoundaryNorm`, `LogNorm`)
            for cases `vmin`/`vmax`/`auto_range` can't express. When provided,
            `vmin`, `vmax`, and `auto_range`'s range computation are bypassed
            entirely; `cmap` still follows the usual rules above.
        vmin : float, optional
            Lower bound of the colormap. If not provided, defaults to the minimum
            value of this DataArray, computed over the full array rather than just
            the displayed slices. Ignored when `norm` is provided, or when
            `auto_range` resolves to a range anchored at zero (see below).
        vmax : float, optional
            Upper bound of the colormap. If not provided, defaults to the maximum
            value of this DataArray, computed over the full array rather than just
            the displayed slices. Ignored when `norm` is provided, or when
            `auto_range=True` and this DataArray has only non-positive values.
        auto_range : bool, default: True
            Whether to pick the colormap range and default colormap automatically
            based on the sign of this DataArray:

            - Both positive and negative values: diverging, symmetric `[-m, m]`
              range where `m = max(|vmin|, |vmax|)` (using the resolved bounds
              above), with `cmap` defaulting to `"coolwarm"` — the right choice
              for diverging statistics where the sign is meaningful (e.g.
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
        alpha : float or xarray.DataArray, optional
            Opacity of this DataArray's overlay: a single scalar value, or a 3D
            DataArray sharing this DataArray's dims, shape, and coordinates (for
            independent per-slice, per-voxel opacity, e.g. to fade out
            low-magnitude voxels instead of masking them out with `threshold`). A
            per-voxel opacity must be a DataArray, not a bare array, so it can be
            validated and aligned against this DataArray; note it is validated
            against self, not `bg_volume`. If not provided, the colormap's own alpha
            channel is respected.
        threshold : float, optional
            Threshold applied to the absolute value of this DataArray. See
            `threshold_mode` for the masking direction. If not provided, no
            thresholding is applied.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Controls how `threshold` is applied:

            - `"lower"`: set pixels below `threshold` (in absolute value) to NaN.
            - `"upper"`: set pixels above `threshold` (in absolute value) to NaN.

        show_colorbar : bool, default: True
            Whether to add a shared colorbar to the figure.
        cbar_label : str, optional
            Label for the colorbar.
        cbar_kwargs : dict, optional
            Additional keyword arguments forwarded to
            [`matplotlib.figure.Figure.colorbar`][matplotlib.figure.Figure.colorbar]
            (e.g. `shrink`, `fraction`, `pad`, `aspect`). Useful to shrink the
            colorbar when it spans a multi-panel grid, since the defaults are sized
            for a single axes.
        show_titles : bool, default: True
            Whether to display subplot titles showing the slice coordinate.
        show_axis_labels : bool, default: True
            Whether to display axis labels (with units when available).
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When
            `False`, overrides `show_axis_labels` and `show_axis_ticks`.
        fontsize : float, optional
            Base font size for all text elements. Subplot titles use `fontsize`
            directly; axis labels and the colorbar label use `0.9 * fontsize`; tick
            labels use `0.85 * fontsize`. If not provided, uses the active Matplotlib
            defaults.
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
            [`matplotlib.axes.Axes`][matplotlib.axes.Axes] or a 2D array of them.
            Must contain exactly as many elements as there are slices. A single
            `Axes` is wrapped automatically and limits the plot to one slice. If not
            provided, new axes are created inside `figure`.
        nrows : int, optional
            Number of rows in the subplot grid. If not provided, computed
            automatically.
        ncols : int, optional
            Number of columns in the subplot grid. If not provided, computed
            automatically.
        dpi : int, optional
            Figure resolution in dots per inch. Ignored when `figure` is provided.

        Returns
        -------
        VolumePlotter
            Object managing the figure, axes, and coordinate mapping for overlays.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> anatomical = xr.open_zarr("output.zarr")["power_doppler"]
        >>> t_map = xr.open_zarr("output.zarr")["t_stat"]
        >>> plotter = t_map.fusi.plot.stat_map(bg_volume=anatomical, slice_mode="z")

        >>> # Blend the overlay with the background instead of fully covering it.
        >>> plotter = t_map.fusi.plot.stat_map(bg_volume=anatomical, alpha=0.6)

        >>> # Non-diverging statistic (e.g. R²): sequential range and colormap
        >>> # picked automatically since r2_map has only non-negative values.
        >>> r2_map = xr.open_zarr("output.zarr")["r2"]
        >>> plotter = r2_map.fusi.plot.stat_map(anatomical)
        """
        return plot_stat_map(
            self._obj,
            bg_volume=bg_volume,
            slice_coords=slice_coords,
            slice_mode=slice_mode,
            bg_kwargs=bg_kwargs,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            auto_range=auto_range,
            alpha=alpha,
            threshold=threshold,
            threshold_mode=threshold_mode,
            show_colorbar=show_colorbar,
            cbar_label=cbar_label,
            cbar_kwargs=cbar_kwargs,
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
        )
