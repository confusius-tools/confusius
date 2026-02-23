"""Image visualization utilities for fUSI data."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from confusius.extract import extract_with_mask
from confusius.signal import clean

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from napari import Viewer
    from napari.layers import Image

# Import napari at module level for backward compatibility and test mocking.
# This is a lightweight import (~111ms) compared to matplotlib/scipy.
import napari  # noqa: F401


def plot_napari(
    data: xr.DataArray,
    scale_method: Literal["db", "log", "power"] | None = None,
    scale_kwargs: dict[str, Any] | None = None,
    show_colorbar: bool = True,
    show_scale_bar: bool = True,
    dim_order: tuple[str, ...] | None = None,
    viewer: "Viewer | None" = None,
    **imshow_kwargs,
) -> tuple["Viewer", list["Image"]]:
    """Display fUSI data using the Napari viewer.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array to visualize. Expected dimensions are (time, z, y, x) where
        z is the elevation/stacking axis, y is depth, and x is lateral. Use
        ``dim_order`` to specify a different dimension ordering.
    scale_method : {"db", "log", "power"}, optional
        Scaling method to apply before display. Use ``"db"`` for decibel scaling,
        ``"log"`` for natural log, ``"power"`` for power scaling. If not provided,
        no scaling is applied.
    scale_kwargs : dict, optional
        Keyword arguments to pass to the scaling method. For example, ``{"factor": 20}``
        for db scaling or ``{"exponent": 0.5}`` for power scaling.
    show_colorbar : bool, default: True
        Whether to show the colorbar.
    show_scale_bar : bool, default: True
        Whether to show the scale bar.
    dim_order : tuple[str, ...], optional
        Dimension ordering for the spatial axes (last three dimensions). If not
        provided, the ordering of the last three dimensions in `data` is used.
    viewer : napari.Viewer, optional
        Existing Napari viewer to add the image layer to. If not provided, a new viewer
        is created.
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
    parameter for Napari to ensure correct physical scaling. If any spatial dimension is
    missing coordinates, no scaling is applied. The spacing is computed as the median
    difference between consecutive coordinate values.

    For unitary dimensions (e.g., a single-slice elevation axis in 2D+t data), the
    spacing cannot be inferred from coordinates. In that case, the function looks for a
    ``voxdim`` attribute on the coordinate variable
    (``data.coords[dim].attrs["voxdim"]``) and uses it as the spacing. If no such
    attribute is found, unit spacing is assumed and a warning is emitted.

    The first coordinate value of each spatial dimension is used as the ``translate``
    parameter so that the image is positioned at its correct physical origin. For
    dimensions without coordinates, a translate of ``0.0`` is used. This ensures that
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

    >>> # Amplitude scaling (factor=20)
    >>> viewer, layer = plot_napari(
    ...     data, scale_method="db", scale_kwargs={"factor": 20}
    ... )

    >>> # Decibel scaling
    >>> viewer, layer = plot_napari(data, scale_method="db")

    >>> # Different dimension ordering (e.g., depth, elevation, lateral)
    >>> viewer, layer = plot_napari(data, dim_order=("y", "z", "x"))

    >>> # Add a second dataset as a new layer in an existing viewer
    >>> viewer, layer1 = plot_napari(data1)
    >>> viewer, layer2 = plot_napari(data2, viewer=viewer)
    """
    if scale_method is not None:
        if scale_kwargs is None:
            scale_kwargs = {}

        if scale_method == "db":
            data = data.fusi.scale.db(**scale_kwargs)
        elif scale_method == "log":
            data = data.fusi.scale.log(**scale_kwargs)
        elif scale_method == "power":
            data = data.fusi.scale.power(**scale_kwargs)
        else:
            raise ValueError(
                f"Unknown scale_method: {scale_method}. "
                "Use 'db', 'log', 'power', or None."
            )

    all_dims = list(data.dims)
    time_dim = "time" if "time" in all_dims else None
    spatial_dims = [d for d in all_dims if d != time_dim]

    if dim_order is not None and set(dim_order) != set(spatial_dims):
        raise ValueError(
            f"dim_order {dim_order} does not match spatial dimensions {spatial_dims}. "
            "Ensure 'dim_order' contains all spatial dimension names."
        )

    # Build the Napari scale from coordinate spacing. Warnings for undefined dims are
    # emitted by .fusi.spacing; fall back to 1.0 so the scale bar still works.
    scale = None
    spacing = data.fusi.spacing
    coord_scales = [
        s if (s := spacing[dim]) is not None else 1.0 for dim in spatial_dims
    ]
    if coord_scales:
        scale = coord_scales

    # Build translate from the first coordinate value per spatial dim (physical origin).
    # Falls back to 0.0 for dimensions without coordinates.
    coord_translates = [
        float(data.coords[dim].values[0]) if dim in data.coords else 0.0
        for dim in spatial_dims
    ]

    # The last 2 (2D) or 3 (3D) dimensions are the displayed spatial axes.
    if dim_order is not None:
        order = []
        if time_dim:
            order.append(all_dims.index(time_dim))
        for dim in dim_order:
            if dim in all_dims:
                order.append(all_dims.index(dim))
        imshow_kwargs["order"] = tuple(order)

    imshow_kwargs.setdefault("axis_labels", all_dims)
    imshow_kwargs.setdefault("name", data.name)
    imshow_kwargs.setdefault("translate", coord_translates)
    viewer, image_layer = napari.imshow(
        data,
        scale=scale,
        viewer=viewer,
        **imshow_kwargs,
    )

    if show_colorbar:
        viewer.layers[0].colorbar.visible = True  # type: ignore[attr-defined]

    if show_scale_bar:
        viewer.scale_bar.visible = True
        viewer.scale_bar.gridded = True
        viewer.scale_bar.unit = "mm"

    return viewer, image_layer


def plot_carpet(
    data: xr.DataArray,
    mask: xr.DataArray | None = None,
    detrend_order: int | None = None,
    standardize: bool = True,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    decimation_threshold: int | None = 800,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
    ax: "Axes | None" = None,
) -> tuple["Figure | SubFigure", "Axes"]:
    """Plot voxel intensities across time as a raster image.

    A carpet plot (also known as "grayplot" or "Power plot") displays voxel
    intensities as a 2D raster image with time on the x-axis and voxels on
    the y-axis. Each row represents one voxel's time series, typically
    standardized to z-scores.

    Parameters
    ----------
    data : xarray.DataArray
        Input data array with a 'time' dimension.
    mask : xarray.DataArray, optional
        Boolean mask with same spatial dimensions and coordinates as `data`. True
        values indicate voxels to include. If not provided, all non-zero voxels
        from the data are included.
    detrend_order : int, optional
        Polynomial order for detrending:

        - ``0``: Remove mean (constant detrending).
        - ``1``: Remove linear trend using least squares regression.
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
    >>> from confusius.plotting import plot_carpet
    >>> data = xr.open_zarr("output.zarr")["iq"]
    >>> fig, ax = plot_carpet(data)

    >>> # With linear detrending
    >>> fig, ax = plot_carpet(data, detrend_order=1)

    >>> # With mask
    >>> import numpy as np
    >>> mask = xr.DataArray(
    ...     np.abs(data.isel(time=0)) > threshold,
    ...     dims=["z", "y", "x"],
    ... )
    >>> fig, ax = plot_carpet(data, mask=mask)
    """
    import matplotlib.pyplot as plt

    if np.iscomplexobj(data):
        data = xr.ufuncs.abs(data)

    if "time" not in data.dims or "time" not in data.coords:
        raise ValueError("Data must have 'time' dimension and coordinates.")

    n_timepoints = data.sizes["time"]

    non_zero_voxels = (data != 0).any(dim="time")
    if mask is None:
        mask = non_zero_voxels
    else:
        mask = mask & non_zero_voxels

    # extract_with_mask creates a MultiIndex on voxels that makes xarray plotting
    # fail; we need to drop the spatial coordinates.
    spatial_dims = [d for d in data.dims if d != "time"]
    signals = extract_with_mask(data, mask)
    signals = signals.drop_vars(spatial_dims + ["voxels"])

    signals = clean(
        signals,
        detrend_order=detrend_order,
        standardize_method="zscore" if standardize else None,
    )

    if vmin is None or vmax is None:
        std_val = float(signals.std(axis=0).mean().values)
        default_vmin = float(signals.mean().values - (2 * std_val))
        default_vmax = float(signals.mean().values + (2 * std_val))
        vmin = vmin or default_vmin
        vmax = vmax or default_vmax

    if decimation_threshold is not None and n_timepoints > decimation_threshold:
        # For decimation, we get the smallest power of 2 greater than the number of
        # volumes divided by the threshold.
        n_decimations = int(
            np.ceil(np.log2(np.ceil(n_timepoints / decimation_threshold)))
        )
        decimation_factor = 2**n_decimations
        signals = signals[::decimation_factor, :]

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    signals.T.plot(cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, yincrease=False)  # type: ignore[call-arg]

    ax.grid(False)
    ax.set_yticks([])
    ax.set_ylabel("Voxels")
    # TODO: we could get time units from attrs, or maybe use pint?
    ax.set_xlabel("Time (s)")

    if title:
        ax.set_title(title)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", 10))
    ax.spines["left"].set_position(("outward", 10))

    return figure, ax
