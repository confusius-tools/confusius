"""Utilities for loading and converting AUTC DAT files."""

import shutil
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import zarr

if TYPE_CHECKING:
    from rich.progress import Progress

from confusius._dims import SPATIAL_DIMS, TIME_DIM, VOXEL_DIMS
from confusius._utils.coordinates import get_probe_surface_origin
from confusius._utils.stack import find_stack_level
from confusius.io._utils import make_attrs_zarr_safe
from confusius.io.utils import check_path


class AUTCDATFileHeader(TypedDict):
    """TypedDict representing the header structure of an AUTC DAT file.

    Attributes
    ----------
    acquisition_block_index : int
        Index of the acquisition block.
    n_z : int
        Number of depth samples.
    n_x : int
        Number of lateral samples.
    n_frames : int
        Number of frames in the block.
    """

    acquisition_block_index: int
    n_z: int
    n_x: int
    n_frames: int


def _get_dat_dtype(
    block_shape: tuple[int, int, int],
    header_dtype: npt.DTypeLike = np.int32,
    data_dtype: npt.DTypeLike = np.complex64,
) -> np.dtype:
    """Get the Numpy dtype for an AUTC DAT file with a specific block shape.

    Parameters
    ----------
    block_shape : tuple[int, int, int]
        Block shape with `(n_x, n_z, n_frames)`.
    header_dtype : dtype_like, default: numpy.int32
        Data type used for the header fields.
    data_dtype : dtype_like, default: numpy.complex64
        Data type used for the data field.

    Returns
    -------
    numpy.dtype
        Numpy dtype representing the structure of the DAT file.
    """
    return np.dtype(
        [
            ("acquisition_block_index", header_dtype),
            ("n_data_items", header_dtype),
            ("n_z", header_dtype),
            ("n_x", header_dtype),
            ("n_frames", header_dtype),
            ("field5", header_dtype),
            ("field6", header_dtype),
            ("field7", header_dtype),
            ("field8", header_dtype),
            ("field9", header_dtype),
            ("data", data_dtype, block_shape),
        ]
    )


NumpyIndex: TypeAlias = (
    bool  # Single boolean
    | int  # Single integer
    | slice  # Slice object
    | list[int]  # List of integers
    | npt.NDArray[np.integer]  # Integer array
    | npt.NDArray[np.bool_]  # Boolean array
    | tuple[int | slice | list[int] | npt.NDArray, ...]  # Multi-dimensional
)
"""Type alias for numpy-style indexing objects.

This type represents valid indexing objects that can be used with numpy arrays,
including single values, slices, lists, arrays, and multi-dimensional combinations.
"""


class AUTCDAT:
    """Beamformed IQ data stored in AUTC DAT format.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the AUTC DAT file.
    n_header_items : int, default: 10
        Number of header items to read from each block.
    header_dtype : dtype_like, default: numpy.int32
        Data type used for header fields.
    data_dtype : dtype_like, default: numpy.complex64
        Data type used for data fields.

    Attributes
    ----------
    path : pathlib.Path
        Resolved path to the AUTC DAT file.
    shape : tuple[int, int, int, int]
        Shape of the data in the AUTC DAT file in the format `(blocks, x, z, frames)`.
    n_blocks : int
        Number of frame blocks in the AUTC DAT file.
    n_frames_per_block : int
        Number of frames per block in the AUTC DAT file.
    acquisition_block_indices : numpy.ndarray
        Array of acquisition block indices contained in the AUTC DAT file.
    """

    def __init__(
        self,
        path: str | Path,
        n_header_items: int = 10,
        header_dtype: npt.DTypeLike = np.int32,
        data_dtype: npt.DTypeLike = np.complex64,
    ):
        self.path = check_path(path, type="file")

        self._header_dtype = header_dtype
        first_header = self._read_first_header(
            dtype=header_dtype, n_items=n_header_items
        )

        n_x = first_header["n_x"]
        n_z = first_header["n_z"]
        n_frames = first_header["n_frames"]

        self._memmap = np.memmap(
            self.path,
            dtype=_get_dat_dtype(
                block_shape=(n_x, n_z, n_frames),
                header_dtype=header_dtype,
                data_dtype=data_dtype,
            ),
            mode="r",
        )

        self._acquisition_block_indices = self._read_acquisition_block_indices()

    def _read_first_header(
        self, dtype: npt.DTypeLike, n_items: int
    ) -> AUTCDATFileHeader:
        """Internal method to read the first block's header.

        Parameters
        ----------
        dtype : dtype_like
            Data type used for header fields.
        n_items : int
            Number of header items to read.

        Returns
        -------
        AUTCDATFileHeader
            Dictionary containing the first block's header information.
        """
        header = np.fromfile(self.path, dtype=dtype, count=n_items)
        return {
            # Frame index is 1-based in the AUTC DAT format. header[1] (n_data_items
            # in the on-disk format) is skipped over: it is not part of the parsed
            # header contract.
            "acquisition_block_index": header[0] - 1,
            "n_z": header[2],
            "n_x": header[3],
            "n_frames": header[4],
        }

    def _read_acquisition_block_indices(self) -> npt.NDArray:
        """Read acquisition block indices from the AUTC DAT file.

        We read the acquisition block indices directly from the file to avoid loading
        the entire memmap into memory.

        Returns
        -------
        numpy.ndarray
            Array of acquisition block indices.
        """
        acquisition_block_indices = np.empty(self.n_blocks, dtype=np.int32)
        with open(self.path, "r") as file:
            for block_index in range(self.n_blocks):
                file.seek(block_index * self._memmap.dtype.itemsize)

                header = np.fromfile(file, dtype=self._header_dtype, count=1)
                acquisition_block_indices[block_index] = header[0]
        return acquisition_block_indices

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of the data in this AUTC DAT file.

        Returns
        -------
        tuple[int, int, int, int]
            Tuple of the form `(blocks, x, z, frames)`.
        """
        return self._memmap["data"].shape

    @property
    def n_blocks(self) -> int:
        """Number of frame blocks in this AUTC DAT file.

        Returns
        -------
        int
            Number of blocks.
        """
        return self.shape[0]

    @property
    def n_frames_per_block(self) -> int:
        """Number of frames per block in this AUTC DAT file.

        Returns
        -------
        int
            Number of frames per block.
        """
        return self.shape[3]

    @property
    def acquisition_block_indices(self) -> npt.NDArray:
        """Acquisition-wide frame indices contained in this AUTC DAT file.

        Returns
        -------
        numpy.ndarray
            Array of acquisition block indices.
        """
        return self._acquisition_block_indices

    def __len__(self) -> int:
        """Number of blocks in this AUTC DAT file.

        Returns
        -------
        int
            Number of blocks.
        """
        return self.shape[0]

    def __getitem__(self, slice_object: NumpyIndex) -> npt.NDArray:
        """Index the data contained in this AUTC DAT file.

        Parameters
        ----------
        slice_object : NumpyIndex
            Indexing object for the data.

        Returns
        -------
        numpy.ndarray
            The requested data from the AUTC DAT file.
        """
        return self._memmap["data"][slice_object]


class AUTCDATsLoader:
    """Load a beamformed IQ acquisition from a set of AUTC DAT files.

    Parameters
    ----------
    root : str or pathlib.Path
        Path to the root directory containing the AUTC DAT files.
    pattern : str, default: "bf*part*.dat"
        Glob pattern used to search for AUTC DAT files inside `root`.
    show_progress : bool, default: True
        Whether to show progress during file indexing. If `False`, no progress bars
        are displayed.
    progress : rich.progress.Progress, optional
        External `rich.progress.Progress` instance to add tasks to. If provided and
        `show_progress` is `True`, a task will be added to this
        `rich.progress.Progress` instance instead of creating a new progress bar with
        `rich.progress.track`.
    track_kwargs : dict, optional
        Additional keyword arguments to pass to `rich.progress.track` if using internal
        progress tracking (only used if `show_progress` is `True` and `progress` is
        `None`).

    Attributes
    ----------
    root : pathlib.Path
        Resolved path to the root directory.
    pattern : str
        Glob pattern used for file searching.
    dats : dict[pathlib.Path, AUTCDAT]
        Dictionary mapping AUTC DAT file paths to `AUTCDAT` instances.
    block_index_to_file : dict[int, pathlib.Path]
        Mapping from block indices to their corresponding file paths.
    """

    def __init__(
        self,
        root: str | Path,
        pattern: str = "bf*part*.dat",
        show_progress: bool = True,
        progress: Optional["Progress"] = None,
        track_kwargs: dict[str, Any] | None = None,
    ):
        self.root = check_path(root, label="root", type="dir")
        self.pattern = pattern
        self.dats: dict[Path, AUTCDAT] = {}
        self.block_index_to_file: dict[int, Path] = {}
        self._index_dat_files(
            show_progress=show_progress, progress=progress, track_kwargs=track_kwargs
        )

    def _index_dat_files(
        self,
        show_progress: bool = True,
        progress: Optional["Progress"] = None,
        track_kwargs: dict[str, Any] | None = None,
    ):
        """Locate AUTCDAT files in the root folder and index their frames."""
        from rich.progress import track

        dat_paths = list(self.root.glob(self.pattern))
        if len(dat_paths) == 0:
            raise FileNotFoundError(
                f"No AUTC DAT files found in {self.root} with pattern {self.pattern}."
            )

        task_id = None

        if not show_progress:
            iterable = enumerate(dat_paths)
        elif progress is not None:
            task_id = progress.add_task(
                "Indexing AUTC DAT files...", total=len(dat_paths)
            )
            iterable = enumerate(dat_paths)
        else:
            kwargs = track_kwargs or {}
            kwargs.setdefault("description", "Indexing AUTC DAT files...")
            kwargs.setdefault("total", len(dat_paths))
            iterable = track(enumerate(dat_paths), **kwargs)

        for _, dat_path in iterable:
            if dat_path not in self.dats:
                n_total_bytes = dat_path.stat().st_size
                if n_total_bytes == 0:
                    warnings.warn(
                        f"Skipping empty AUTC DAT file: {dat_path}",
                        stacklevel=find_stack_level(),
                    )
                    continue

                dat = AUTCDAT(dat_path)
                self.dats[dat_path] = dat
                for frame_label in dat.acquisition_block_indices:
                    self.block_index_to_file[frame_label] = dat_path

                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)

    def __getitem__(
        self, slice_object: int | slice | tuple[int | slice, ...]
    ) -> npt.NDArray:
        """Index the data across all AUTC DAT files.

        Parameters
        ----------
        slice_object : int or slice or tuple[int or slice, ...]
            Indexing object for the data. Can be:
            - `int`: Single block index;
            - `slice`: Range of block indices;
            - `tuple`: Multi-dimensional indexing `(block, x, z, frame)`.

        Returns
        -------
        numpy.ndarray
            The requested data from the AUTC DAT files.
        """
        if isinstance(slice_object, tuple):
            block_slice = slice_object[0]
            remaining_slices = slice_object[1:]
        elif isinstance(slice_object, (int, slice)):
            block_slice = slice_object
            remaining_slices = ()
        else:
            raise TypeError(
                "slice_object must be an int, slice, or tuple of (int or slice)."
            )

        if isinstance(block_slice, int):
            block_slice = slice(block_slice, block_slice + 1)

        block_indices = self.blocks[block_slice]

        # Pre-allocate to avoid repeated allocations during loading.
        loader_shape = self.shape
        output_shape = [len(block_indices)]

        for i, slice_obj in enumerate(remaining_slices):
            if i + 1 < len(loader_shape):
                dim_size = loader_shape[i + 1]
                if isinstance(slice_obj, slice):
                    start, stop, step = slice_obj.indices(dim_size)
                    output_shape.append(len(range(start, stop, step)))
                elif isinstance(slice_obj, int):
                    output_shape.append(1)

        for i in range(len(remaining_slices) + 1, len(loader_shape)):
            output_shape.append(loader_shape[i])

        result = np.empty(tuple(output_shape), dtype=self.dtype)

        # We group blocks by AUTC DAT file for efficient access.
        blocks_by_file: dict[Path, list] = {}
        for i, acquisition_block_index in enumerate(block_indices):
            dat_path = self.block_index_to_file[acquisition_block_index]
            if dat_path not in blocks_by_file:
                blocks_by_file[dat_path] = []
            blocks_by_file[dat_path].append((i, acquisition_block_index))

        for dat_path, file_blocks in blocks_by_file.items():
            dat = self.dats[dat_path]

            for original_index, acquisition_block_index in file_blocks:
                block_position = np.where(
                    dat.acquisition_block_indices == acquisition_block_index
                )[0][0]

                full_slice = (block_position,) + remaining_slices
                result[original_index] = dat[full_slice]

        return result

    @property
    def blocks(self) -> npt.NDArray:
        """List of all blocks available in the acquisition.

        Returns
        -------
        numpy.ndarray
            Sorted array of all available block indices.
        """
        return np.asarray(sorted(self.block_index_to_file.keys()))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of the complete dataset across all AUTC DAT files.

        Returns
        -------
        tuple[int, int, int, int]
            Shape in the format `(n_blocks, n_x, n_z, n_frames)`.
        """
        first_dat_shape = next(iter(self.dats.values())).shape
        return (
            len(self.blocks),
            first_dat_shape[1],
            first_dat_shape[2],
            first_dat_shape[3],
        )

    @property
    def dtype(self) -> np.dtype:
        """Data type of the arrays in the AUTC DAT files.

        Returns
        -------
        numpy.dtype
            Data type of the stored arrays.
        """
        return next(iter(self.dats.values()))._memmap["data"].dtype


def convert_autc_dats_to_zarr(
    dats_root: str | Path,
    output_path: str | Path,
    spacing: Sequence[float],
    dats_pattern: str = "bf*part*.dat",
    frames_per_chunk: int | None = None,
    chunks_per_shard: int | None = None,
    batch_size: int = 100,
    overwrite: bool = False,
    zarr_kwargs: dict[str, Any] | None = None,
    origin: Sequence[float] | None = None,
    transmit_frequency: float | None = None,
    probe_n_elements: int | None = None,
    probe_pitch: float | None = None,
    beamforming_sound_velocity: float | None = None,
    plane_wave_angles: npt.ArrayLike | None = None,
    compound_sampling_frequency: float | None = None,
    pulse_repetition_frequency: float | None = None,
    beamforming_method: str | None = None,
    skip_first_blocks: int = 0,
    skip_last_blocks: int = 0,
    block_times: npt.ArrayLike | None = None,
    show_progress: bool = True,
    progress: Optional["Progress"] = None,
    track_kwargs: dict[str, Any] | None = None,
) -> "zarr.Group":
    """Convert AUTC DAT files to Zarr format compatible with Xarray.

    Beamformed IQ data is converted to a Zarr group with an `iq` array of shape
    `(time, k, j, i)` chunked along the first dimension. Voxel-to-world geometry is
    stored as `attrs["voxel_to_world"]` on the `iq` array rather than dense `z`/`y`/`x`
    coordinate arrays, matching [`confusius.io.save`][confusius.io.save]'s Zarr
    convention. Coordinates are stored as separate Zarr arrays following Xarray
    conventions, allowing the data to be opened directly with `xarray.open_zarr()`.

    Parameters
    ----------
    dats_root : str or pathlib.Path
        Path to the directory containing AUTC DAT files.
    output_path : str or pathlib.Path
        Path where the Zarr group will be saved.
    spacing : sequence[float]
        World voxel spacing in `(k, j, i)` order (elevation, axial/depth, lateral), in
        millimeters. AUTC's own file data carries no physical calibration for any
        axis (see the Notes section for the elevation placeholder), so this must
        always be supplied explicitly.
    dats_pattern : str, default: "bf*part*.dat"
        Glob pattern used to search for AUTC DAT files inside `dats_root`.
    frames_per_chunk : int, optional
        Number of frames to include in each Zarr chunk. If not provided, defaults to
        the number of frames per block in the AUTC DAT files.
    chunks_per_shard : int, optional
        Number of chunks to bundle into each shard. If provided, enables Zarr sharding
        to reduce the number of files on disk. If not provided, sharding is disabled.
    batch_size : int, default: 100
        Number of blocks to process in each batch.
    overwrite : bool, default: False
        Whether to overwrite existing Zarr group at the output path.
    zarr_kwargs : dict, optional
        Additional keyword arguments to pass to `zarr.create_array` for the main data
        array.
    origin : sequence[float], optional
        World position of voxel index 0 in `(k, j, i)` order, in millimeters in the
        probe-relative coordinate system (origin at the center of the probe face). If
        not provided, defaults to the same probe-centered/surface-referenced
        convention as [`create_voxeldata`][confusius.xarray.create_voxeldata]:
        elevation and lateral centered on the probe, depth referenced to the probe
        surface.
    transmit_frequency : float, optional
        Central frequency of the ultrasound probe in hertz.
    probe_n_elements : int, optional
        Number of probe transducers.
    probe_pitch : float, optional
        Inter-element pitch of the probe in millimeters.
    beamforming_sound_velocity : float, optional
        Speed of sound in meters per second.
    plane_wave_angles : array_like, optional
        Angles at which tilted plane waves are emitted in degrees.
    compound_sampling_frequency : float, optional
        Sampling frequency of the compounded frames in hertz.
    pulse_repetition_frequency : float, optional
        Pulse repetition frequency in hertz.
    beamforming_method : str, optional
        Beamforming method used (e.g. `"DAS"`).
    skip_first_blocks : int, default: 0
        Number of blocks to skip from the beginning of the acquisition. This is useful
        when the first blocks are known to be corrupted or unusable.
    skip_last_blocks : int, default: 0
        Number of blocks to skip from the end of the acquisition. This is useful
        when the last blocks are known to be corrupted or unusable.
    block_times : (n_blocks_after_skip,) array_like, optional
        Start time of each IQ block in seconds, for the retained blocks only. If
        provided, individual volume times will be computed and stored as a time
        coordinate. Requires `compound_sampling_frequency` to be provided. If not
        provided, time coordinate will be computed based on
        `compound_sampling_frequency` or set to frame indices.
    show_progress : bool, default: True
        Whether to show progress during conversion. If `False`, no progress bars are
        displayed.
    progress : rich.progress.Progress, optional
        External `rich.progress.Progress` instance to add tasks to. If provided and
        `show_progress` is `True`, a task will be added to this
        `rich.progress.Progress` instance instead of creating a new progress bar with
        `rich.progress.track`.
    track_kwargs : dict, optional
        Additional keyword arguments to pass to `rich.progress.track` if using internal
        progress tracking (only used if `show_progress` is `True` and `progress` is
        `None`).

    Returns
    -------
    zarr.Group
        The created Zarr group containing the `iq` data array and coordinate arrays.
        Can be opened directly with `xarray.open_zarr()`.

    Raises
    ------
    ValueError
        If `spacing` or `origin` does not have length 3, or if `block_times` is
        provided without `compound_sampling_frequency`.

    Notes
    -----
    The output Zarr group follows Xarray conventions and can be opened with::

        import xarray as xr
        ds = xr.open_zarr("output.zarr")
        iq = ds["iq"]

    `iq` above carries plain `k`/`j`/`i` voxel dims, not VoxelData --
    [`xarray.open_zarr`][xarray.open_zarr] doesn't know how to rebuild a
    `VoxelToWorldIndex` from `attrs["voxel_to_world"]`. Load with
    [`confusius.io.load`][confusius.io.load] instead to get the world coordinates back:

        import confusius as cf
        iq = cf.io.load("output.zarr")

    Metadata attributes (e.g., `transmit_frequency`, `beamforming_sound_velocity`) are stored
    on the `iq` DataArray (accessible via `iq.attrs`), consistent with how
    reduction functions return DataArrays with attributes.

    The group contains:

    - `iq`: The main data array with dimensions `(time, k, j, i)`. Voxel-to-world
      geometry is stored as `attrs["voxel_to_world"]`/`attrs["world_coord_attrs"]`
      rather than dense `z`/`y`/`x` coordinate arrays.
    - `time`: Time coordinate array.
    - `k`: Elevation voxel coordinate array (always `[0]` for 2D data).
    - `j`: Axial (depth) voxel coordinate array.
    - `i`: Lateral voxel coordinate array.

    Spatial coordinates (`z`, `y`, `x`) follow the ConfUSIus probe-relative
    coordinate system: world distances in millimeters along each voxel axis, with the
    origin at the center of the probe face. Unlike EchoFrame data (where coordinates are
    embedded in the metadata file), AUTC's own file data carries no physical
    calibration for any axis, so `spacing` must always be supplied by the caller --
    there is no meaningful default.

    Correctly computing elevation (`spacing[0]`) spacing requires the elevation beam
    width, which itself needs the elevation aperture and elevation focus; that
    computation isn't implemented here, so pass `spacing[0]` explicitly if that
    information is known, or an intentional placeholder (e.g. `0.4`) otherwise.
    """
    from rich.progress import track

    loader = AUTCDATsLoader(
        root=dats_root,
        pattern=dats_pattern,
        show_progress=show_progress,
        progress=progress,
        track_kwargs=track_kwargs,
    )

    output_path = Path(output_path)
    n_blocks, n_x, n_z, n_frames = loader.shape

    total_skip = skip_first_blocks + skip_last_blocks
    if total_skip >= n_blocks:
        raise ValueError(
            f"Cannot skip {total_skip} blocks (skip_first_blocks={skip_first_blocks}, "
            f"skip_last_blocks={skip_last_blocks}) from {n_blocks} total blocks."
        )

    n_blocks_after_skip = n_blocks - total_skip
    n_total_frames = n_blocks_after_skip * n_frames

    if frames_per_chunk is None:
        frames_per_chunk = n_frames

    output_shape = (n_total_frames, 1, n_z, n_x)
    chunks = (frames_per_chunk, 1, n_z, n_x)

    if chunks_per_shard is not None:
        shards = (frames_per_chunk * chunks_per_shard, 1, n_z, n_x)
    else:
        shards = None

    zarr_group = zarr.open_group(output_path, mode="w" if overwrite else "w-")

    # Dimension names required for Zarr v3 / Xarray compatibility.
    dim_names = [TIME_DIM, *VOXEL_DIMS]

    create_array_kwargs = zarr_kwargs.copy() if zarr_kwargs else {}
    handled_keys = {"shape", "chunks", "shards", "dtype", "dimension_names"}
    overridden_keys = handled_keys & create_array_kwargs.keys()
    if overridden_keys:
        warnings.warn(
            f"zarr_kwargs contains keys that are handled by function parameters and "
            f"will be overridden: {overridden_keys}.",
            stacklevel=find_stack_level(),
        )
    create_array_kwargs["shape"] = output_shape
    create_array_kwargs["chunks"] = chunks
    create_array_kwargs["dtype"] = loader.dtype
    create_array_kwargs["dimension_names"] = dim_names

    if shards is not None:
        create_array_kwargs["shards"] = shards

    zarr_iq = zarr_group.create_array("iq", **create_array_kwargs)

    if block_times is not None:
        if compound_sampling_frequency is None:
            raise ValueError(
                "compound_sampling_frequency is required when block_times is provided "
                "to compute individual volume times."
            )
        block_times_array = np.asarray(block_times)
        if block_times_array.size != n_blocks_after_skip:
            raise ValueError(
                f"block_times length ({block_times_array.size}) does not match "
                f"number of blocks after skipping ({n_blocks_after_skip})."
            )
        time_values = np.concatenate(
            [
                block_start + np.arange(n_frames) / compound_sampling_frequency
                for block_start in block_times_array
            ]
        )
    elif compound_sampling_frequency is not None:
        skipped_first_frames = skip_first_blocks * n_frames
        time_values = (
            np.arange(n_total_frames, dtype=np.float64) / compound_sampling_frequency
            + skipped_first_frames / compound_sampling_frequency
        )
    else:
        time_values = np.arange(n_total_frames, dtype=np.float64)

    zarr_group.create_array("time", data=time_values, dimension_names=["time"])
    zarr_group["time"].attrs["units"] = "s"
    zarr_group["time"].attrs["long_name"] = "Time"
    zarr_group["time"].attrs["volume_acquisition_reference"] = "start"
    if plane_wave_angles is not None and pulse_repetition_frequency is not None:
        zarr_group["time"].attrs["volume_acquisition_duration"] = float(
            np.asarray(plane_wave_angles).size / pulse_repetition_frequency
        )

    for dim, size in zip(VOXEL_DIMS, (1, n_z, n_x), strict=True):
        zarr_group.create_array(
            dim, data=np.arange(size, dtype=np.int64), dimension_names=[dim]
        )

    if len(spacing) != 3:
        raise ValueError(f"spacing must have length 3 (k, j, i), got {len(spacing)}.")
    spacing_k, spacing_j, spacing_i = (float(value) for value in spacing)
    resolved_spacing = (spacing_k, spacing_j, spacing_i)

    if origin is None:
        resolved_origin = get_probe_surface_origin(
            {"k": 1, "j": n_z, "i": n_x}, resolved_spacing
        )
    elif len(origin) != 3:
        raise ValueError(f"origin must have length 3 (k, j, i), got {len(origin)}.")
    else:
        origin_k, origin_j, origin_i = (float(value) for value in origin)
        resolved_origin = (origin_k, origin_j, origin_i)

    voxel_to_world = np.eye(4)
    voxel_to_world[:3, :3] = np.diag(resolved_spacing)
    voxel_to_world[:3, 3] = resolved_origin
    zarr_iq.attrs.update(
        make_attrs_zarr_safe(
            {
                "voxel_to_world": voxel_to_world,
                "world_coord_attrs": {name: {"units": "mm"} for name in SPATIAL_DIMS},
            }
        )
    )

    if transmit_frequency is not None:
        zarr_iq.attrs["transmit_frequency"] = transmit_frequency
    if probe_n_elements is not None:
        zarr_iq.attrs["probe_number_of_elements"] = probe_n_elements
    if probe_pitch is not None:
        zarr_iq.attrs["probe_pitch"] = probe_pitch
    if beamforming_sound_velocity is not None:
        zarr_iq.attrs["beamforming_sound_velocity"] = beamforming_sound_velocity
    if plane_wave_angles is not None:
        zarr_iq.attrs["plane_wave_angles"] = np.asarray(plane_wave_angles).tolist()
    if compound_sampling_frequency is not None:
        zarr_iq.attrs["compound_sampling_frequency"] = compound_sampling_frequency
    if pulse_repetition_frequency is not None:
        zarr_iq.attrs["pulse_repetition_frequency"] = pulse_repetition_frequency
    if beamforming_method is not None:
        zarr_iq.attrs["beamforming_method"] = beamforming_method

    first_block = skip_first_blocks
    last_block = n_blocks - skip_last_blocks

    n_batches = (n_blocks_after_skip + batch_size - 1) // batch_size
    batches = range(first_block, last_block, batch_size)
    task_id = None

    if not show_progress:
        iterable = batches
    elif progress is not None:
        task_id = progress.add_task(
            "Converting AUTC DAT files to Zarr...", total=n_batches
        )
        iterable = batches
    else:
        kwargs = track_kwargs or {}
        kwargs.setdefault("description", "Converting AUTC DAT files to Zarr...")
        iterable = track(batches, **kwargs)

    try:
        for idx, start_block in enumerate(iterable):
            end_block = min(start_block + batch_size, last_block)

            batch_data = np.transpose(
                loader[start_block:end_block], axes=(0, 3, 2, 1)
            ).reshape(-1, 1, n_z, n_x)

            output_start = (start_block - skip_first_blocks) * n_frames
            output_end = (end_block - skip_first_blocks) * n_frames
            zarr_iq[output_start:output_end] = batch_data

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

        # Consolidate metadata for faster opening with Xarray.
        # Suppress warning about consolidated metadata not being in Zarr v3 spec yet.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata")
            zarr.consolidate_metadata(output_path)

    except Exception:
        # Clean up incomplete zarr store to avoid leaving unconsolidated data.
        if output_path.exists():
            shutil.rmtree(output_path)
        raise

    return zarr_group
