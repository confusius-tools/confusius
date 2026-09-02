"""The `.atlas.plot` namespace: napari plotting for atlas Datasets."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from napari import Viewer
    from napari.layers import Surface


class AtlasPlotAccessor:
    """Plotting methods for an atlas Dataset, reached via `ds.atlas.plot`.

    Parameters
    ----------
    ds : xarray.Dataset
        Atlas Dataset the plotting methods operate on.

    Examples
    --------
    >>> import confusius as cf
    >>> atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_25um")
    >>> viewer, layer = atlas.atlas.plot.mesh("VISp")
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def mesh(
        self,
        regions: int | str | Sequence[int | str],
        sides: (
            Literal["left", "right", "both"]
            | Sequence[Literal["left", "right", "both"]]
        ) = "both",
        *,
        clip: bool = True,
        values: npt.NDArray[np.floating] | None = None,
        viewer: "Viewer | None" = None,
        show_scale_bar: bool = True,
        **layer_kwargs,
    ) -> "tuple[Viewer, Surface]":
        """Display one or more region surface meshes in napari.

        Thin wrapper around
        [`plot_atlas_mesh`][confusius.plotting.plot_atlas_mesh] bound to this atlas.

        Parameters
        ----------
        regions : int or str or sequence of int or str
            One or more regions to display, each given as a structure index or acronym.
            All requested meshes are merged into a single surface layer.
        sides : {"left", "right", "both"} or sequence thereof, default: "both"
            Hemisphere filter, forwarded to
            [`get_mesh`][confusius.atlas.AtlasAccessor.get_mesh]. Pass a scalar to apply
            the same side to all regions, or a sequence of the same length as `regions`
            for per-region control.
        clip : bool, default: True
            Whether to clip the mesh to the reference grid, forwarded to
            [`get_mesh`][confusius.atlas.AtlasAccessor.get_mesh].
        values : (N,) or (N, T) numpy.ndarray, optional
            Per-vertex scalar values used to color the surface through the layer's
            colormap. If not provided, each region is drawn in its atlas color.
        viewer : napari.Viewer, optional
            Existing napari viewer to add the layer to. If not provided, a new viewer
            is created.
        show_scale_bar : bool, default: True
            Whether to show the scale bar.
        **layer_kwargs
            Additional keyword arguments passed through to
            [`plot_atlas_mesh`][confusius.plotting.plot_atlas_mesh].

        Returns
        -------
        viewer : napari.Viewer
            The napari viewer instance with the surface layer added.
        layer : napari.layers.Surface
            The surface layer added to the viewer.
        """
        # Imported here so `import confusius.atlas` does not eagerly pull in napari.
        from confusius.plotting import plot_atlas_mesh

        return plot_atlas_mesh(
            self._ds,
            regions,
            sides,
            clip=clip,
            values=values,
            viewer=viewer,
            show_scale_bar=show_scale_bar,
            **layer_kwargs,
        )
