"""Xarray accessor for plotting."""

from typing import TYPE_CHECKING, Any, Literal

import xarray as xr

from confusius.plotting import VolumePlotter, plot_carpet, plot_napari, plot_volume

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image


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

    def __call__(self, **kwargs) -> tuple[Any, Any]:
        """Call the napari plotting method by default.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `napari`.
        """
        return self.napari(**kwargs)

    def napari(
        self,
        scale_method: Literal["db", "log", "power"] | None = None,
        scale_kwargs: dict[str, Any] | None = None,
        show_colorbar: bool = True,
        show_scale_bar: bool = True,
        dim_order: tuple[str, ...] | None = None,
        viewer: "Viewer | None" = None,
        **imshow_kwargs,
    ) -> "tuple[Viewer, list[Image]]":
        """Display data in napari viewer.

        Parameters
        ----------
        scale_method : {"db", "log", "power"}, optional
            Scaling method to apply before display. Use ``"db"`` for decibel scaling,
            ``"log"`` for natural log, ``"power"`` for power scaling. If not provided,
            no scaling is applied.
        scale_kwargs : dict, optional
            Keyword arguments to pass to the scaling method. For example, ``{"factor":
            20}`` for db scaling or ``{"exponent": 0.5}`` for power scaling.
        show_colorbar : bool, default: True
            Whether to show the colorbar.
        show_scale_bar : bool, default: True
            Whether to show the scale bar.
        dim_order : tuple[str, ...], optional
            Dimension ordering for the spatial axes (last three dimensions). If not
            provided, the ordering of the last three dimensions in `data` is used.
        viewer : napari.Viewer, optional
            Existing Napari viewer to add the image layer to. If not provided, a new
            viewer is created.
        **imshow_kwargs
            Additional keyword arguments passed to `napari.imshow`, such as
            ``contrast_limits``, ``colormap``, etc.

        Returns
        -------
        viewer : napari.Viewer
            The Napari viewer instance.
        layer : napari.layers.Image
            The image layer added to the viewer.

        Notes
        -----
        If all spatial dimensions have coordinates, their spacing is used as the scale
        parameter for Napari to ensure correct physical scaling. If any spatial dimension
        is missing coordinates, no scaling is applied. The spacing is computed as the
        median difference between consecutive coordinate values.

        For unitary dimensions (e.g., a single-slice elevation axis in 2D+t data), the
        spacing cannot be inferred from coordinates. In that case, the function looks for
        a ``voxdim`` attribute on the coordinate variable
        (``data.coords[dim].attrs["voxdim"]``) and uses it as the spacing. If no such
        attribute is found, unit spacing is assumed and a warning is emitted.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> data = xr.open_zarr("output.zarr")["iq"]
        >>> viewer, layer = data.fusi.plot.napari()

        >>> # Custom contrast limits
        >>> viewer, layer = data.fusi.plot.napari(contrast_limits=(0, 100))

        >>> # Amplitude scaling (factor=20)
        >>> viewer, layer = data.fusi.plot.napari(
        ...     scale_method="db", scale_kwargs={"factor": 20}
        ... )

        >>> # Decibel scaling
        >>> viewer, layer = data.fusi.plot.napari(scale_method="db")

        >>> # Different dimension ordering (e.g., depth, elevation, lateral)
        >>> viewer, layer = data.fusi.plot.napari(dim_order=("y", "z", "x"))

        >>> # Add a second dataset as a new layer in an existing viewer
        >>> viewer, layer1 = data1.fusi.plot.napari()
        >>> viewer, layer2 = data2.fusi.plot.napari(viewer=viewer)
        """
        return plot_napari(
            self._obj,
            scale_method=scale_method,
            scale_kwargs=scale_kwargs,
            show_colorbar=show_colorbar,
            show_scale_bar=show_scale_bar,
            dim_order=dim_order,
            viewer=viewer,
            **imshow_kwargs,
        )

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
            ``True`` values indicate voxels to include. If not provided, all non-zero
            voxels from the data are included.
        detrend_order : int, optional
            Polynomial order for detrending:

            - ``0``: Remove mean (constant detrending).
            - ``1``: Remove linear trend using least squares regression (default).
            - ``2+``: Remove polynomial trend of specified order.

            If not provided, no detrending is applied.
        standardize : bool, default: True
            Whether to standardize each voxel's time series to z-scores.
        cmap : str, default: "gray"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for colormap. If not provided, uses ``mean - 2*std``.
        vmax : float, optional
            Maximum value for colormap. If not provided, uses ``mean + 2*std``.
        decimation_threshold : int or None, default: 800
            If the number of timepoints exceeds this value, data is downsampled
            along the time axis to improve plotting performance. Set to ``None`` to
            disable downsampling.
        figsize : tuple[float, float], default: (10, 5)
            Figure size in inches ``(width, height)``.
        title : str, optional
            Plot title.
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
        cmap: "str | Colormap" = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 1.0,
        black_bg: bool = True,
        figure: "Figure | None" = None,
        axes: "npt.NDArray[Any] | None" = None,
        show_colorbar: bool = True,
        cbar_label: str | None = None,
        dpi: int | None = None,
    ) -> "VolumePlotter":
        """Plot 2D slices of a volume as a matplotlib subplot grid.

        Parameters
        ----------
        slice_coords : list[float], optional
            Coordinate values along ``slice_mode`` at which to extract slices.
            Slices are selected by nearest-neighbour lookup. If not provided,
            all coordinate values along ``slice_mode`` are used.
        slice_mode : str, default: "z"
            Dimension along which to slice (e.g. ``"x"``, ``"y"``, ``"z"``,
            ``"time"``). After slicing, each panel must be 2D.
        nrows : int, optional
            Number of rows in the subplot grid. If not provided, computed
            automatically together with ``ncols`` to produce a near-square
            layout.
        ncols : int, optional
            Number of columns in the subplot grid. If not provided, computed
            automatically together with ``nrows``.
        threshold : float, optional
            Threshold applied to ``|data|``. See ``threshold_mode`` for the
            masking direction. If not provided, no thresholding is applied.
        threshold_mode : {"lower", "upper"}, default: "lower"
            Controls how ``threshold`` is applied:

            - ``"lower"``: set pixels where ``|data| < threshold`` to NaN.
            - ``"upper"``: set pixels where ``|data| > threshold`` to NaN.

        cmap : str, default: "gray"
            Matplotlib colormap name.
        vmin : float, optional
            Lower bound of the colormap. Defaults to the 2nd percentile.
        vmax : float, optional
            Upper bound of the colormap. Defaults to the 98th percentile.
        alpha : float, default: 1.0
            Opacity of the image (0 transparent, 1 opaque).
        black_bg : bool, default: True
            If ``True``, set backgrounds to black and text/spines to white.
        figure : matplotlib.figure.Figure, optional
            Existing figure to draw into.
        axes : numpy.ndarray, optional
            Existing 2D array of ``matplotlib.axes.Axes`` to draw into.
        show_colorbar : bool, default: True
            Whether to add a shared colorbar.
        cbar_label : str, optional
            Label for the colorbar.
        dpi : int, optional
            Figure resolution in dots per inch.

        Returns
        -------
        VolumePlotter
            Object managing the figure, axes, and coordinate mapping for overlays.

        Examples
        --------
        >>> import xarray as xr
        >>> import confusius  # Register accessor.
        >>> data = xr.open_zarr("output.zarr")["pwd"]
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
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            black_bg=black_bg,
            figure=figure,
            axes=axes,
            show_colorbar=show_colorbar,
            cbar_label=cbar_label,
            dpi=dpi,
        )
