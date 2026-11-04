"""Connects a napari layer to its lazily resampled transform preview."""

from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import xarray as xr
from napari.layers import Image, Labels

from confusius._dims import POSE_DIM, VOXEL_DIMS
from confusius._napari._transform._view import (
    AxisAlignedGrid,
    ResampledView,
    compute_sweep_grid,
    compute_tight_grid,
)
from confusius._utils.napari import build_napari_layer_data
from confusius.registration import resample_like


class CommittedLabels(NamedTuple):
    """What a Labels layer displays when no preview is running.

    Attributes
    ----------
    data : numpy.ndarray
        Layer data, including any paint.
    fill_value : float, optional
        Fill value the layer was built with. `None` means `build_napari_layer_data`'s
        default.
    """

    data: npt.NDArray[Any]
    fill_value: float | None


def _raise_if_paint_would_be_lost(
    source: xr.DataArray, committed: CommittedLabels | None
) -> None:
    """Refuse to rebuild a Labels layer from a source its paint never reached.

    A commit rebuilds the layer from `source`. Paint only survives that when the layer
    holds `source`'s own array, so this compares a rebuilt layer with the layer as it is.
    A selection-time check cannot do this, since painting can happen at any time after.

    Parameters
    ----------
    source : xarray.DataArray
        Committed VoxelData array.
    committed : CommittedLabels, optional
        Labels layer's committed data. If not provided, nothing is checked.

    Raises
    ------
    ValueError
        If the layer holds values that rebuilding it from `source` would lose.
    """
    if committed is None:
        return
    rebuilt, _, _ = build_napari_layer_data(
        source, layer_type="labels", resample_fill_value=committed.fill_value
    )
    if rebuilt is not committed.data and not np.array_equal(
        np.asarray(rebuilt), committed.data
    ):
        raise ValueError(
            "This layer holds paint that is not in its source data, and transforming it "
            "would lose that paint."
        )


def _has_unsaved_paint(layer: Labels, displayed: xr.DataArray) -> bool:
    """Return whether a Labels layer holds values its metadata does not.

    Parameters
    ----------
    layer : napari.layers.Labels
        Labels layer.
    displayed : xarray.DataArray
        DataArray the layer's metadata says it displays.

    Returns
    -------
    bool
        Whether `layer.data` differs from `displayed`. Only NumPy data can be painted,
        so anything else counts as unpainted without reading it.
    """
    if layer.data is displayed.data or not isinstance(layer.data, np.ndarray):
        return False
    return not np.array_equal(layer.data, np.asarray(displayed.data))


class PreviewSession:
    """Live preview of world-space transforms on one napari layer.

    The preview only touches `layer.data`, `layer.scale` and `layer.translate`. Both
    metadata arrays stay as they are until `adopt`, so everything reading them sees
    committed data.

    Parameters
    ----------
    layer : napari.layers.Image or napari.layers.Labels
        Layer to preview on.

    Attributes
    ----------
    layer : napari.layers.Image or napari.layers.Labels
        Layer being previewed on.
    source : xarray.DataArray
        Committed VoxelData array the preview resamples.

    Raises
    ------
    ValueError
        If `layer` was not built from a VoxelData array, holds a `pose` dim, or is a
        Labels layer with paint that its metadata does not hold. Applying a transform
        rebuilds the layer from its metadata, which would silently lose that paint.
    """

    def __init__(self, layer: Image | Labels) -> None:
        displayed = layer.metadata.get("xarray")
        if displayed is None:
            raise ValueError(f"Layer '{layer.name}' is not a ConfUSIus layer.")
        if POSE_DIM in displayed.dims:
            raise ValueError(
                f"Layer '{layer.name}' has a pose dimension, which the transform panel "
                "does not support."
            )
        self._layer_type: Literal["image", "labels"] = (
            "labels" if isinstance(layer, Labels) else "image"
        )
        if isinstance(layer, Labels) and _has_unsaved_paint(layer, displayed):
            raise ValueError(
                f"Layer '{layer.name}' holds paint that is not in its source data, and "
                "transforming it would lose that paint."
            )
        self.layer = layer
        self.source: xr.DataArray = layer.metadata.get("source_xarray", displayed)
        # Kept across a restore, because it caches a volume that may be slow to read.
        self._view: ResampledView | None = None
        self._stash: tuple[Any, npt.NDArray[Any], npt.NDArray[Any]] | None = None
        self._committed_fill: float | None = None

    @property
    def layer_type(self) -> Literal["image", "labels"]:
        """Return the kind of layer being previewed on.

        Returns
        -------
        {"image", "labels"}
            Kind of layer.
        """
        return self._layer_type

    @property
    def fill_value(self) -> float | None:
        """Return the value outside the field of view.

        Returns
        -------
        float or None
            Fill value, or `None` before any preview.
        """
        return self._view.fill_value if self._view is not None else None

    @property
    def committed_labels(self) -> CommittedLabels | None:
        """Return what a commit must not lose.

        Returns
        -------
        CommittedLabels or None
            Committed data of a Labels layer, or `None` for anything that cannot be
            painted: an image layer, or data that is not a NumPy array.
        """
        data = self._stash[0] if self._stash is not None else self.layer.data
        if self._layer_type != "labels" or not isinstance(data, np.ndarray):
            return None
        return CommittedLabels(data, self._committed_fill)

    def _get_view(self) -> ResampledView:
        """Return the preview's view, stashing the layer if no preview is running.

        Returns
        -------
        ResampledView
            View to display.
        """
        if self._stash is None:
            self._stash = (
                self.layer.data,
                self.layer.scale.copy(),
                self.layer.translate.copy(),
            )
        if self._view is None:
            self._view = ResampledView(self.source, layer_type=self._layer_type)
        return self._view

    def _set_layer_grid(self, grid: AxisAlignedGrid) -> bool:
        """Set the layer's spatial scale and translate, keeping leading dims.

        Each assignment makes napari slice the layer and upload it again, so only what
        changed gets assigned.

        Parameters
        ----------
        grid : AxisAlignedGrid
            Grid to position the layer on.

        Returns
        -------
        bool
            Whether anything changed, and so whether the layer was sliced again.
        """
        changed = False
        if not np.array_equal(self.layer.scale[-3:], grid.spacing):
            self.layer.scale = np.array([*self.layer.scale[:-3], *grid.spacing])
            changed = True
        if not np.array_equal(self.layer.translate[-3:], grid.origin):
            self.layer.translate = np.array([*self.layer.translate[:-3], *grid.origin])
            changed = True
        return changed

    def _display(
        self, transform: npt.NDArray[np.float64], grid: AxisAlignedGrid
    ) -> None:
        """Display `transform` on `grid`.

        Parameters
        ----------
        transform : (4, 4) numpy.ndarray
            World-space transform of the source.
        grid : AxisAlignedGrid
            Grid to display on.
        """
        view = self._get_view()
        needs_data = self.layer.data is not view or tuple(grid.shape) != view.shape[-3:]
        view.set_transform(transform, grid)
        # The view already has the new shape, so the geometry goes first: slicing it with
        # the old scale would resample a wrong plane for nothing.
        was_sliced = self._set_layer_grid(grid)
        if needs_data:
            # The data setter takes any array-like, which its stubs do not say.
            self.layer.data = view  # type: ignore
            # The data setter makes the layer editable again, and the view is read-only.
            if self._layer_type == "labels":
                self.layer.editable = False
        elif not was_sliced:
            self.layer.refresh(thumbnail=False, highlight=False, extent=False)

    def show(self, transform: npt.NDArray[np.float64]) -> None:
        """Display a transform on the tight grid `plot_napari` would use.

        Parameters
        ----------
        transform : (4, 4) numpy.ndarray
            World-space transform of the source. The identity restores the layer.
        """
        if np.allclose(transform, np.eye(4)):
            self.restore()
            return
        self._display(transform, compute_tight_grid(self.source, transform))

    def begin_rotation_drag(
        self,
        transform: npt.NDArray[np.float64],
        axis: npt.NDArray[np.float64],
        pivot: npt.NDArray[np.float64],
        *,
        max_voxels: int | None = None,
    ) -> None:
        """Switch to a grid enclosing a whole rotation drag.

        Parameters
        ----------
        transform : (4, 4) numpy.ndarray
            World-space transform when the drag starts.
        axis : (3,) numpy.ndarray
            Unit rotation axis, in world `(z, y, x)` order.
        pivot : (3,) numpy.ndarray
            World point the rotation axis goes through.
        max_voxels : int, optional
            Number of voxels to coarsen the grid down to, approximately. If not provided,
            the grid keeps full resolution.
        """
        self._display(
            transform,
            compute_sweep_grid(
                self.source, transform, axis, pivot, max_voxels=max_voxels
            ),
        )

    def drag_to(self, transform: npt.NDArray[np.float64]) -> None:
        """Display a transform on the current grid, at the cost of one slice.

        Parameters
        ----------
        transform : (4, 4) numpy.ndarray
            World-space transform of the source.
        """
        self._get_view().set_transform(transform)
        self.layer.refresh(thumbnail=False, highlight=False, extent=False)

    def restore(self) -> None:
        """Put back the layer as it was before the preview."""
        if self._stash is None:
            return
        data, scale, translate = self._stash
        # The data goes first and the stash goes last, so that a view that can no longer
        # be read cannot get in the way of getting the layer back.
        self.layer.data = data
        self.layer.scale = scale
        self.layer.translate = translate
        self._stash = None
        # Paint lands in the layer's array, so a cached labels volume would go stale.
        if self._layer_type == "labels":
            self._view = None

    def adopt(self, data: Any, kwargs: dict[str, Any]) -> None:
        """Make a rebuilt layer the committed one.

        Parameters
        ----------
        data : Any
            Layer data from `build_applied_layer`.
        kwargs : dict[str, Any]
            Layer keyword arguments from `build_applied_layer`.
        """
        self._committed_fill = self.fill_value
        self._view = None
        self._stash = None
        self.layer.scale = kwargs["scale"]
        self.layer.translate = kwargs["translate"]
        self.layer.data = data
        self.layer.metadata.update(kwargs["metadata"])
        self.source = kwargs["metadata"]["source_xarray"]


def build_applied_layer(
    source: xr.DataArray,
    transform: npt.NDArray[np.float64],
    *,
    layer_type: Literal["image", "labels"],
    fill_value: float | None,
    name: str,
    committed_labels: CommittedLabels | None = None,
) -> tuple[Any, dict[str, Any], Literal["image", "labels"]]:
    """Compose a transform into `source` and build its layer.

    This does no Qt work, so it is safe in a worker thread.

    Parameters
    ----------
    source : xarray.DataArray
        Committed VoxelData array.
    transform : (4, 4) numpy.ndarray
        World-space transform to compose.
    layer_type : {"image", "labels"}
        Type of layer to build.
    fill_value : float, optional
        Value outside the field of view. If not provided, `source`'s own minimum.
    name : str
        Layer name.
    committed_labels : CommittedLabels, optional
        Labels layer's committed data, to refuse a commit that would lose its paint. If
        not provided, nothing is checked.

    Returns
    -------
    data : Any
        Layer data.
    kwargs : dict[str, Any]
        Layer keyword arguments. `metadata` holds the new `xarray` and `source_xarray`.
    layer_type : {"image", "labels"}
        Type of layer built.

    Raises
    ------
    ValueError
        If this would lose paint held in `committed_labels`.
    """
    _raise_if_paint_would_be_lost(source, committed_labels)
    return build_napari_layer_data(
        source.fusi.affine.apply(transform),
        name=name,
        layer_type=layer_type,
        resample_fill_value=fill_value,
    )


def build_resampled_layer(
    source: xr.DataArray,
    transform: npt.NDArray[np.float64],
    *,
    layer_type: Literal["image", "labels"],
    fill_value: float | None,
    name: str,
    reference: xr.DataArray | None = None,
    committed_labels: CommittedLabels | None = None,
) -> tuple[Any, dict[str, Any], Literal["image", "labels"]]:
    """Resample `source` after a transform onto a new grid, for a new layer.

    This does no Qt work, so it is safe in a worker thread.

    Parameters
    ----------
    source : xarray.DataArray
        Committed VoxelData array.
    transform : (4, 4) numpy.ndarray
        World-space transform to compose.
    layer_type : {"image", "labels"}
        Type of layer to build.
    fill_value : float, optional
        Value outside the field of view. If not provided, `source`'s own minimum.
    name : str
        Layer name.
    reference : xarray.DataArray, optional
        VoxelData array whose spatial grid to resample onto. If not provided, the
        axis-aligned grid `plot_napari` would use.
    committed_labels : CommittedLabels, optional
        Labels layer's committed data, to refuse a commit that would lose its paint. If
        not provided, nothing is checked.

    Returns
    -------
    data : Any
        Layer data.
    kwargs : dict[str, Any]
        Layer keyword arguments. `metadata["source_xarray"]` holds the resampled voxels,
        so saving the layer writes those.
    layer_type : {"image", "labels"}
        Type of layer built.

    Raises
    ------
    ValueError
        If `reference` is not a VoxelData array with the same spatial units as `source`,
        or if this would lose paint held in `committed_labels`.
    """
    _raise_if_paint_would_be_lost(source, committed_labels)
    moved = source.fusi.affine.apply(transform)
    # A translation is not resampled, so the new layer could end up holding the original's
    # array, and painting it would edit the original.
    if layer_type == "labels" and isinstance(moved.data, np.ndarray):
        moved = moved.copy(deep=True)
    if reference is not None:
        grid = reference.isel(
            {dim: 0 for dim in reference.dims if dim not in VOXEL_DIMS}, drop=True
        )
        moved = resample_like(
            moved,
            grid,
            np.eye(4),
            interpolation="nearest" if layer_type == "labels" else "linear",
            fill_value=fill_value,
        )
    data, kwargs, resolved_type = build_napari_layer_data(
        moved,
        name=name,
        layer_type=layer_type,
        resample_fill_value=fill_value,
    )
    kwargs["metadata"]["source_xarray"] = kwargs["metadata"]["xarray"]
    return data, kwargs, resolved_type
