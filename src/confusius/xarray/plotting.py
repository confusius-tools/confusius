"""Xarray accessor for plotting."""

from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from confusius.plotting import (
    VolumePlotter,
    draw_napari_labels,
    labels_from_layer,
    plot_carpet,
    plot_contours,
    plot_napari,
    plot_volume,
)

if TYPE_CHECKING:
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
            Existing Napari viewer to add the layer to. If not provided, a new
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
            The Napari viewer instance with the layer added.
        layer : napari.layers.Image or napari.layers.Labels
            The layer added to the viewer.

        Notes
        -----
        If all spatial dimensions have coordinates, their spacing is used as the scale
        parameter for Napari to ensure correct physical scaling. If any spatial dimension
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
            Existing Napari viewer to add layers to. If not provided, a new
            viewer is created.
        **plot_kwargs
            Additional keyword arguments forwarded to
            [`plot_napari`][confusius.plotting.plot_napari] for the image layer
            (e.g. `colormap`, `contrast_limits`).

        Returns
        -------
        viewer : napari.Viewer
            The Napari viewer instance with the image and Labels layers.
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
        black_bg: bool = False,
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
        black_bg : bool, default: False
            Whether to use a black figure background with white foreground elements
            (spines, ticks, labels). Use `True` for dark-themed figures.
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
            black_bg=black_bg,
            ax=ax,
        )

    def volume(
        self,
        slice_coords: list[float] | None = None,
        slice_mode: str = "z",
        nrows: int | None = None,
        ncols: int | None = None,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
        cmap: "str | Colormap | None" = None,
        norm: "Normalize | None" = None,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 1.0,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        show_titles: bool = True,
        show_axis_labels: bool = True,
        show_axis_ticks: bool = True,
        show_axes: bool = True,
        yincrease: bool = False,
        xincrease: bool = True,
        black_bg: bool = True,
        figure: "Figure | None" = None,
        axes: "npt.NDArray[Any] | None" = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot 2D slices of a volume as a matplotlib subplot grid.

        See [`confusius.plotting.plot_volume`][confusius.plotting.plot_volume] for full
        details.

        Parameters
        ----------
        slice_coords : list[float], optional
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
        alpha : float, default: 1.0
            Opacity of the image.
        show_colorbar : bool, default: True
            Whether to add a shared colorbar to the figure.
        cbar_label : str, optional
            Label for the colorbar.
        show_titles : bool, default: True
            Whether to display subplot titles showing the slice coordinate.
        show_axis_labels : bool, default: True
            Whether to display axis labels (with units when available).
        show_axis_ticks : bool, default: True
            Whether to display axis tick labels.
        show_axes : bool, default: True
            Whether to show all axis decorations (spines, ticks, labels). When
            `False`, overrides `show_axis_labels` and `show_axis_ticks`.
        yincrease : bool, default: False
            Whether the y-axis increases upward (`True`) or downward (`False`).
        xincrease : bool, default: True
            Whether the x-axis increases to the right (`True`) or left
            (`False`).
        black_bg : bool, default: True
            Whether to set the figure background to black.
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
        ...     black_bg=True,
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
            show_titles=show_titles,
            show_axis_labels=show_axis_labels,
            show_axis_ticks=show_axis_ticks,
            show_axes=show_axes,
            yincrease=yincrease,
            xincrease=xincrease,
            black_bg=black_bg,
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
        slice_coords: list[float] | None = None,
        yincrease: bool = False,
        xincrease: bool = True,
        black_bg: bool = True,
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
        slice_coords : list[float], optional
            Coordinate values along `slice_mode` at which to extract slices.
            Slices are selected by nearest-neighbour lookup. If not provided, all
            coordinate values along `slice_mode` are used.
        yincrease : bool, default: False
            Whether the y-axis increases upward (`True`) or downward (`False`).
        xincrease : bool, default: True
            Whether the x-axis increases to the right (`True`) or left (`False`).
        black_bg : bool, default: True
            Whether to set the figure background to black.
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
            yincrease=yincrease,
            xincrease=xincrease,
            black_bg=black_bg,
            figure=figure,
            axes=axes,
            **kwargs,
        )
