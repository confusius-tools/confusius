"""Lazily resampled layer data for the live transform preview."""

import operator
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import xarray as xr
from scipy.spatial.transform import Rotation

from confusius._dims import TIME_DIM, VOXEL_DIMS, WORLD_DIMS
from confusius._utils.geometry import (
    get_voxel_to_world_affine,
    get_voxel_to_world_direction_matrix,
    get_voxel_to_world_index_origin,
    get_voxel_to_world_index_spacing,
)
from confusius._utils.plotting import (
    compute_axis_aligned_grid_geometry,
    sort_coords_for_plot,
)
from confusius.validation import ensure_voxeldata

OUTPUT_DIRECTION = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
"""SimpleITK direction of an output image with axes `(i, j, k)` along world `(x, y, z)`.

Rows follow world `(z, y, x)`. These axes make the array SimpleITK returns C-ordered
`(k, j, i)`, so neither direction needs a memory reorder.
"""

SWEEP_SAMPLES = 145
"""Number of rotations `compute_sweep_grid` samples.

Over a full turn this is a 2.5 degree step, where an arc strays less than 0.03% of its
radius from the samples. The grid's one voxel margin covers that for radii up to several
thousand voxels.
"""


class AxisAlignedGrid(NamedTuple):
    """World-axis-aligned voxel grid, in `(z, y, x)` order.

    Attributes
    ----------
    shape : (3,) numpy.ndarray
        Number of voxels along each world axis.
    spacing : (3,) numpy.ndarray
        World distance between consecutive voxels.
    origin : (3,) numpy.ndarray
        World location of the first voxel.
    """

    shape: npt.NDArray[np.int64]
    spacing: npt.NDArray[np.float64]
    origin: npt.NDArray[np.float64]


def _get_voxel_bounds(source: xr.DataArray) -> list[tuple[float, float]]:
    """Return the lowest and highest voxel coordinate along `k`, `j` and `i`.

    Parameters
    ----------
    source : xarray.DataArray
        VoxelData array.

    Returns
    -------
    list of tuple of float
        One `(low, high)` pair per voxel dim.
    """
    return [
        (float(source.coords[dim].values.min()), float(source.coords[dim].values.max()))
        for dim in VOXEL_DIMS
    ]


def compute_tight_grid(
    source: xr.DataArray, transform: npt.NDArray[np.float64]
) -> AxisAlignedGrid:
    """Return the grid `plot_napari` would display `source` on after `transform`.

    Parameters
    ----------
    source : xarray.DataArray
        VoxelData array.
    transform : (4, 4) numpy.ndarray
        World-space transform applied to `source`.

    Returns
    -------
    AxisAlignedGrid
        Grid enclosing the transformed voxel centers.
    """
    shape, spacing, origin = compute_axis_aligned_grid_geometry(
        transform @ get_voxel_to_world_affine(source), _get_voxel_bounds(source)
    )
    return AxisAlignedGrid(np.array(shape), np.array(spacing), np.array(origin))


def compute_sweep_grid(
    source: xr.DataArray,
    transform: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    pivot: npt.NDArray[np.float64],
    max_angle: float = 180.0,
    max_voxels: int | None = None,
) -> AxisAlignedGrid:
    """Return one grid enclosing `source` over a whole rotation drag.

    Keeping one grid during a drag keeps the layer's shape and the dimension sliders
    still, and swapping `layer.data` on a same-shape layer costs about a third.

    Parameters
    ----------
    source : xarray.DataArray
        VoxelData array.
    transform : (4, 4) numpy.ndarray
        World-space transform when the drag starts.
    axis : (3,) numpy.ndarray
        Unit rotation axis, in world `(z, y, x)` order.
    pivot : (3,) numpy.ndarray
        World point the rotation axis goes through.
    max_angle : float, default: 180.0
        Largest rotation either way from `transform`, in degrees. The default encloses
        a full turn, which is as far as a dial can go in one drag.
    max_voxels : int, optional
        Number of voxels to coarsen the grid down to, approximately. If not provided, the
        grid keeps full resolution. A pivot far from the volume, like the default grid
        origin, sweeps a grid several times the volume's size.

    Returns
    -------
    AxisAlignedGrid
        Grid with the spacing of the tight grid at drag start, unless `max_voxels` asks
        for coarser, and one voxel of margin.
    """
    affine = transform @ get_voxel_to_world_affine(source)
    bounds = _get_voxel_bounds(source)
    spacing = compute_tight_grid(source, transform).spacing

    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for angle in np.linspace(-max_angle, max_angle, SWEEP_SAMPLES):
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(np.deg2rad(angle) * axis).as_matrix()
        step[:3, 3] = pivot - step[:3, :3] @ pivot
        shape, turned_spacing, origin = compute_axis_aligned_grid_geometry(
            step @ affine, bounds
        )
        lower = np.minimum(lower, origin)
        upper = np.maximum(
            upper, np.array(origin) + (np.array(shape) - 1) * np.array(turned_spacing)
        )

    lower = lower - spacing
    upper = upper + spacing
    shape = np.ceil((upper - lower) / spacing - 1e-6).astype(np.int64) + 1
    if max_voxels is not None and np.prod(shape) > max_voxels:
        spacing = spacing * (np.prod(shape) / max_voxels) ** (1 / 3)
        shape = np.ceil((upper - lower) / spacing - 1e-6).astype(np.int64) + 1
    return AxisAlignedGrid(shape, spacing, lower)


class ResampledView:
    """Array-like resampling a VoxelData array only where napari reads it.

    Assigned to `layer.data`, it makes a 2D update cost one resampled plane, however
    many time points or voxels `source` holds. It caches the latest time point as one
    SimpleITK image, so a new transform only changes that image's geometry. It follows
    the grid rule and interpolation of `plot_napari`, so a read returns what `plot_napari`
    would display.

    Leading dims (`time` and any extra ones) are indexed directly. A read spanning several
    of their elements reads `source` once, because reading a compressed recording frame by
    frame would decompress it each time.

    Parameters
    ----------
    source : xarray.DataArray
        VoxelData array without a `pose` dim.
    layer_type : {"image", "labels"}, default: "image"
        Kind of layer displaying the view. `"labels"` uses nearest-neighbor interpolation
        and integer values, `"image"` uses linear interpolation.
    fill_value : float, optional
        Value outside the field of view. If not provided, `0` for labels, otherwise the
        minimum of the first volume read, kept for the view's lifetime.

    Raises
    ------
    ValueError
        If `source` is not a VoxelData array without a `pose` dim, or a voxel dim has no
        well-defined spacing.
    """

    def __init__(
        self,
        source: xr.DataArray,
        *,
        layer_type: Literal["image", "labels"] = "image",
        fill_value: float | None = None,
    ) -> None:
        source = ensure_voxeldata(
            source, require_time=False, allow_pose=False, allow_extra_dims=True
        )
        # `build_napari_layer_data` sorts these, so a slider index must select the same
        # element here as on the committed layer.
        self._source = sort_coords_for_plot(
            source, [dim for dim in source.dims[:-3] if dim != TIME_DIM]
        )
        self._array = self._source.data
        self._fill = (
            0.0 if fill_value is None and layer_type == "labels" else fill_value
        )
        self._interpolator = (
            sitk.sitkNearestNeighbor if layer_type == "labels" else sitk.sitkLinear
        )
        self._is_labels = layer_type == "labels"

        kind = self._source.dtype.kind
        if kind == "c":
            self.dtype = np.abs(np.empty(0, dtype=self._source.dtype)).dtype
        elif kind == "b" or (self._is_labels and kind not in "iu"):
            self.dtype = np.dtype(np.int32)
        else:
            self.dtype = np.dtype(self._source.dtype)
        # SimpleITK rejects non-native byte orders, which NIfTI files can have.
        self.dtype = self.dtype.newbyteorder("=")

        self._position_affine = self._compute_position_affine()
        self._moving: sitk.Image | None = None
        self._moving_lead: tuple[int, ...] | None = None
        self._memo_key: tuple[int | slice, ...] | None = None
        self._memo_value: npt.NDArray[Any] | None = None
        self.set_transform(np.eye(4), compute_tight_grid(self._source, np.eye(4)))

    def _compute_position_affine(self) -> npt.NDArray[np.float64]:
        """Return the affine from array position to world, for any voxel coordinates.

        The voxel-to-world affine takes voxel coordinate values, which stay put when an
        array is cropped or strided, whereas SimpleITK needs array positions.

        Returns
        -------
        (4, 4) numpy.ndarray
            Homogeneous position-to-world affine.

        Raises
        ------
        ValueError
            If a voxel dim has no well-defined spacing.
        """
        spacing = get_voxel_to_world_index_spacing(self._source)
        if any(spacing[dim] is None for dim in VOXEL_DIMS):
            raise ValueError(
                "Cannot resample irregularly spaced voxel coordinates for display."
            )
        origin = get_voxel_to_world_index_origin(self._source)
        affine = np.eye(4)
        affine[:3, :3] = get_voxel_to_world_direction_matrix(self._source) @ np.diag(
            [spacing[dim] for dim in VOXEL_DIMS]
        )
        affine[:3, 3] = [origin[name] for name in WORLD_DIMS]
        return affine

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the displayed array.

        Returns
        -------
        tuple of int
            Size of each dimension, leading dims first.
        """
        lead = self._array.shape[:-3]
        return (*lead, *(int(size) for size in self._grid.shape))

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the displayed array.

        Returns
        -------
        int
            Number of dimensions.
        """
        return self._array.ndim

    @property
    def size(self) -> int:
        """Return the number of displayed voxels.

        Returns
        -------
        int
            Product of `shape`. napari's contrast range buttons read it.
        """
        return int(np.prod(self.shape))

    @property
    def fill_value(self) -> float:
        """Return the value outside the field of view.

        Returns
        -------
        float
            Fill value, reading a volume if it is not known yet.
        """
        if self._fill is None:
            self._load_volume((0,) * (self.ndim - 3))
        assert self._fill is not None
        return self._fill

    def __len__(self) -> int:
        """Return the size of the first dimension.

        Returns
        -------
        int
            Size of the first dimension.
        """
        return self.shape[0]

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]:
        """Return the whole displayed array.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike, optional
            Requested dtype. If not provided, the view's own.
        copy : bool, optional
            Part of the NumPy protocol, and ignored.

        Returns
        -------
        numpy.ndarray
            The whole displayed array, in memory. It may be the view's own last result,
            so it must not be written to.
        """
        return np.asarray(self[(slice(None),) * self.ndim], dtype=dtype)

    def set_transform(
        self,
        transform: npt.NDArray[np.float64],
        grid: AxisAlignedGrid | None = None,
    ) -> None:
        """Change what the view displays.

        Parameters
        ----------
        transform : (4, 4) numpy.ndarray
            World-space transform applied to `source`.
        grid : AxisAlignedGrid, optional
            Grid to display on. If not provided, the current one is kept.
        """
        self._matrix = transform @ self._position_affine
        if grid is not None:
            self._grid = grid
        if self._moving is not None:
            self._set_geometry(self._moving)
        self._memo_key = None
        self._memo_value = None

    def _set_geometry(self, image: sitk.Image) -> None:
        """Point a `(i, j, k)` SimpleITK image at the current transform.

        Parameters
        ----------
        image : SimpleITK.Image
            Image to update in place.
        """
        linear = self._matrix[:3, :3]
        norms = np.linalg.norm(linear, axis=0)
        image.SetSpacing(norms[::-1].tolist())
        image.SetOrigin(self._matrix[:3, 3].tolist())
        image.SetDirection((linear / norms)[:, ::-1].ravel().tolist())

    def _to_display_values(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Convert values read from `source` to what the layer displays.

        Parameters
        ----------
        values : numpy.ndarray
            Values as read.

        Returns
        -------
        numpy.ndarray
            Magnitude of complex values, and integers for labels.
        """
        if np.iscomplexobj(values):
            values = np.abs(values)
        return values.astype(self.dtype, copy=False)

    def _note_fill(self, volume: npt.NDArray[Any]) -> None:
        """Settle the fill value from the first volume read, if it is not known yet.

        Parameters
        ----------
        volume : numpy.ndarray
            Volume just read.
        """
        if self._fill is None:
            self._fill = float(np.nanmin(volume))

    def _to_image(self, volume: npt.NDArray[Any]) -> sitk.Image:
        """Build a SimpleITK image from a `(k, j, i)` volume.

        Parameters
        ----------
        volume : numpy.ndarray
            Volume in display values.

        Returns
        -------
        SimpleITK.Image
            Image with axes `(i, j, k)` matching the volume's C-ordered memory, unlike
            the `.T` used elsewhere, which forces a reorder of every voxel.
        """
        image = sitk.GetImageFromArray(volume)
        self._set_geometry(image)
        return image

    def _load_volume(self, lead: tuple[int, ...]) -> sitk.Image:
        """Return the image for a leading-dim index, reading `source` once per index.

        Parameters
        ----------
        lead : tuple of int
            Non-negative index along each leading dim.

        Returns
        -------
        SimpleITK.Image
            Image at the current transform.
        """
        if self._moving is None or self._moving_lead != lead:
            volume = self._to_display_values(np.asarray(self._array[lead]))
            self._note_fill(volume)
            self._moving = self._to_image(volume)
            self._moving_lead = lead
        return self._moving

    def _normalize_key(self, key: Any) -> tuple[int | slice, ...]:
        """Expand a NumPy-style key to one non-negative index or slice per dimension.

        Parameters
        ----------
        key : int or slice or Ellipsis or tuple
            Key as given to `__getitem__`.

        Returns
        -------
        tuple of (int or slice)
            One entry per dimension, with concrete slice bounds.

        Raises
        ------
        IndexError
            If `key` has too many indices, an out-of-bounds index, or a negative step.
        TypeError
            If an index is neither an integer, a slice nor an Ellipsis.
        """
        items = key if isinstance(key, tuple) else (key,)
        has_ellipsis = any(item is Ellipsis for item in items)
        missing = self.ndim - (len(items) - int(has_ellipsis))
        if missing < 0:
            raise IndexError(
                f"too many indices: the array is {self.ndim}-dimensional, got {key!r}."
            )
        expanded: list[Any] = []
        for item in items:
            if item is Ellipsis:
                expanded.extend([slice(None)] * missing)
            else:
                expanded.append(item)
        if not has_ellipsis:
            expanded.extend([slice(None)] * missing)

        normalized: list[int | slice] = []
        for axis, (item, size) in enumerate(zip(expanded, self.shape, strict=True)):
            if isinstance(item, slice):
                start, stop, step = item.indices(size)
                if step < 0:
                    raise IndexError(f"a negative step is not supported, got {item!r}.")
                normalized.append(slice(start, stop, step))
            else:
                index = operator.index(item)
                if not -size <= index < size:
                    raise IndexError(
                        f"index {index} is out of bounds for axis {axis} with size "
                        f"{size}."
                    )
                normalized.append(index % size)
        return tuple(normalized)

    def _get_result_shape(self, key: tuple[int | slice, ...]) -> tuple[int, ...]:
        """Return the shape a normalized key selects.

        Parameters
        ----------
        key : tuple of (int or slice)
            Normalized key.

        Returns
        -------
        tuple of int
            Length of each slice in `key`.
        """
        return tuple(
            len(range(item.start, item.stop, item.step))
            for item in key
            if isinstance(item, slice)
        )

    def _resample_box(
        self, moving: sitk.Image, spatial: tuple[int | slice, ...]
    ) -> npt.NDArray[Any]:
        """Resample a box of the output grid.

        Parameters
        ----------
        moving : SimpleITK.Image
            Image to resample.
        spatial : tuple of (int or slice)
            Normalized key along `(k, j, i)`.

        Returns
        -------
        numpy.ndarray
            Resampled values. Integer keys drop their dimension.
        """
        starts = np.zeros(3, dtype=np.int64)
        steps = np.ones(3, dtype=np.int64)
        lengths = np.ones(3, dtype=np.int64)
        dropped: list[int] = []
        for axis, item in enumerate(spatial):
            if isinstance(item, slice):
                starts[axis], steps[axis] = item.start, item.step
                lengths[axis] = len(range(item.start, item.stop, item.step))
            else:
                starts[axis] = item
                dropped.append(axis)

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(lengths[::-1].tolist())
        resampler.SetOutputSpacing((self._grid.spacing * steps)[::-1].tolist())
        resampler.SetOutputOrigin(
            (self._grid.origin + starts * self._grid.spacing).tolist()
        )
        resampler.SetOutputDirection(OUTPUT_DIRECTION)
        resampler.SetInterpolator(self._interpolator)
        resampler.SetDefaultPixelValue(self.fill_value)
        # `GetArrayViewFromImage` would avoid this copy, but the view does not keep its
        # image alive and segfaults once the image is collected.
        result = sitk.GetArrayFromImage(resampler.Execute(moving))
        return np.squeeze(result, axis=tuple(dropped)) if dropped else result

    def _read_voxel_course(
        self, lead: tuple[int | slice, ...], spatial: tuple[int, ...]
    ) -> npt.NDArray[Any]:
        """Return one output voxel across leading dims, reading `source` once.

        This is the read behind the Signals panel's cursor readout. SimpleITK has no way
        to resample across time, and resampling frame by frame reads a compressed
        recording once per frame, so it interpolates here the way SimpleITK does.

        Parameters
        ----------
        lead : tuple of (int or slice)
            Normalized key along leading dims, with at least one slice.
        spatial : tuple of int
            Output voxel along `(k, j, i)`.

        Returns
        -------
        numpy.ndarray
            Values with one dimension per slice in `lead`.
        """
        world = self._grid.origin + np.array(spatial) * self._grid.spacing
        position = np.linalg.solve(self._matrix, [*world, 1.0])[:3]
        sizes = np.array(self._array.shape[-3:])
        # SimpleITK's own test for a point being inside the buffer.
        if np.any(position < -0.5) or np.any(position >= sizes - 0.5):
            return np.full(self._get_result_shape(lead), self.fill_value, self.dtype)

        if self._is_labels:
            low = high = np.clip(np.floor(position + 0.5).astype(int), 0, sizes - 1)
            weights = np.zeros(3)
        else:
            low = np.clip(np.floor(position).astype(int), 0, sizes - 1)
            high = np.minimum(low + 1, sizes - 1)
            weights = np.where(high > low, position - low, 0.0)

        box = tuple(slice(a, b + 1) for a, b in zip(low, high, strict=True))
        block = self._to_display_values(np.asarray(self._array[(*lead, *box)]))
        for weight in weights[::-1]:
            block = (1.0 - weight) * block[..., 0] + weight * block[..., -1]
        return block.astype(self.dtype, copy=False)

    def _resample_frames(
        self, lead: tuple[int | slice, ...], spatial: tuple[int | slice, ...]
    ) -> npt.NDArray[Any]:
        """Resample a box across leading dims, reading `source` once.

        Parameters
        ----------
        lead : tuple of (int or slice)
            Normalized key along leading dims, with at least one slice.
        spatial : tuple of (int or slice)
            Normalized key along `(k, j, i)`.

        Returns
        -------
        numpy.ndarray
            Values with one dimension per slice in the key.
        """
        block = self._to_display_values(np.asarray(self._array[lead]))
        lead_shape = block.shape[:-3]
        self._note_fill(block[(0,) * len(lead_shape)])
        result = np.empty(
            (*lead_shape, *self._get_result_shape(spatial)), dtype=self.dtype
        )
        for index in np.ndindex(lead_shape):
            result[index] = self._resample_box(self._to_image(block[index]), spatial)
        return result

    def __getitem__(self, key: Any) -> npt.NDArray[Any]:
        """Return the displayed values `key` selects, resampling only those.

        Parameters
        ----------
        key : int or slice or Ellipsis or tuple
            NumPy-style key, without negative steps.

        Returns
        -------
        numpy.ndarray
            Selected values.
        """
        normalized = self._normalize_key(key)
        result_shape = self._get_result_shape(normalized)
        if 0 in result_shape:
            return np.empty(result_shape, dtype=self.dtype)

        lead = normalized[:-3]
        spatial = normalized[-3:]
        lead_lengths = self._get_result_shape(lead)
        if any(length > 1 for length in lead_lengths):
            if not any(isinstance(item, slice) for item in spatial):
                return self._read_voxel_course(lead, cast("tuple[int, ...]", spatial))
            return self._resample_frames(lead, spatial)

        # One update refreshes the layer more than once, which matters when napari asks
        # for a whole volume in 3D.
        if self._memo_key == normalized and self._memo_value is not None:
            return self._memo_value
        # napari 0.9 selects the sliders' position with length-one slices rather than
        # integers. They must reach the cached volume too, or every update reads `source`.
        index = tuple(item.start if isinstance(item, slice) else item for item in lead)
        result = self._resample_box(self._load_volume(index), spatial)
        result = result[(np.newaxis,) * len(lead_lengths)]
        self._memo_key = normalized
        self._memo_value = result
        return result
