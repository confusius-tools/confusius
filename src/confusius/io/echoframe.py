"""Utilities for loading and converting EchoFrame DAT files."""

import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import dask
import dask.array as da
import h5py as h5
import numpy as np
import numpy.typing as npt
import xarray as xr
import zarr

if TYPE_CHECKING:
    from rich.progress import Progress

from confusius._utils.stack import find_stack_level
from confusius.io.utils import check_path


class EchoFrameMetadata(TypedDict):
    """Metadata extracted from an EchoFrame MAT file.

    Uses fUSI-BIDS compliant field names for consistency with the BIDS specification.

    Attributes
    ----------
    lateral_coords : numpy.ndarray
        Lateral coordinates in millimeters, cropped if the sequence uses ROI cropping.
    axial_coords : numpy.ndarray
        Axial (depth) coordinates in millimeters, cropped if the sequence uses ROI
        cropping.
    transmit_frequency : float
        Central frequency of the ultrasound probe in hertz.
    probe_number_of_elements : int
        Number of probe transducer elements.
    probe_pitch : float
        Inter-element pitch of the probe in millimeters.
    beamforming_sound_velocity : float
        Speed of sound in meters per second used during beamforming.
    plane_wave_angles : numpy.ndarray
        Angles at which tilted plane waves are emitted in degrees.
    compound_sampling_frequency : float
        Sampling frequency of the compounded frames in hertz.
    pulse_repetition_frequency : float
        Single plane wave pulse repetition frequency in hertz.
    beamforming_method : str
        Beamforming method used (e.g. `"DAS"`).
    n_volumes_per_block : int
        Number of volumes per acquisition block.
    """

    lateral_coords: npt.NDArray[np.float64]
    axial_coords: npt.NDArray[np.float64]
    transmit_frequency: float
    probe_number_of_elements: int
    probe_pitch: float
    beamforming_sound_velocity: float
    plane_wave_angles: npt.NDArray[np.float64]
    compound_sampling_frequency: float
    pulse_repetition_frequency: float
    beamforming_method: str
    n_volumes_per_block: int


def load_echoframe_metadata(meta_path: str | Path) -> EchoFrameMetadata:
    """Load acquisition metadata from an EchoFrame MAT file.

    Parameters
    ----------
    meta_path : str or pathlib.Path
        Path to the EchoFrame sequence parameter file (MAT v7.3 / HDF5 format).

    Returns
    -------
    EchoFrameMATMetadata
        Dictionary containing the extracted metadata fields.

    Raises
    ------
    FileNotFoundError
        If `meta_path` does not exist or is not a file.
    """
    meta_path = check_path(meta_path, label="meta_path", type="file")

    with h5.File(meta_path, "r") as f:
        recon_spec = f["ReconSpec"]
        assert isinstance(recon_spec, h5.Group)
        receive_spec = f["ReceiveSpec"]
        assert isinstance(receive_spec, h5.Group)
        probe_spec = f["ProbeSpec"]
        assert isinstance(probe_spec, h5.Group)
        transmit_spec = f["TransmitSpec"]
        assert isinstance(transmit_spec, h5.Group)

        # Cropping information.
        crop = (
            bool(np.array(recon_spec["cropBF"][:])) if "cropBF" in recon_spec else False
        )

        # Spatial coordinates.
        # recon_spec["x_axis"] is lateral (x dimension in ConfUSIus).
        # recon_spec["z_axis"] is depth/axial (y dimension in ConfUSIus).
        x_axis_full = np.array(recon_spec["x_axis"][:]).flatten()
        z_axis_full = np.array(recon_spec["z_axis"][:]).flatten()

        if crop:
            # croppingROI is 1-indexed, convert to 0-indexed.
            cropping_roi = (
                np.array(recon_spec["croppingROI"][:]).flatten().astype(int) - 1
            )
            z_start, z_end, x_start, x_end = cropping_roi
            lateral_coords = x_axis_full[x_start : x_end + 1]
            axial_coords = z_axis_full[z_start : z_end + 1]
        else:
            lateral_coords = x_axis_full
            axial_coords = z_axis_full

        # Probe parameters.
        transmit_frequency = float(np.array(probe_spec["Fc"][:]).item())
        probe_n_elements = int(np.array(probe_spec["nElementsX"][:]).item())
        # Probe pitch is stored in meters in the EchoFrame metadata files.
        probe_pitch = float(np.array(probe_spec["pitchX"][:]).item()) * 1e3

        # Sequence parameters.
        speed_of_sound = float(np.array(recon_spec["c0"][:]).item())

        # Plane wave steering angles.
        plane_wave_angles = np.array(transmit_spec["steerX"][:]).flatten()

        # Sampling frequencies.
        pulse_repetition_frequency = float(
            1 / (receive_spec["transmitReceiveTimeMus"][:].item() * 1e-6)
        )
        compound_sampling_frequency = (
            pulse_repetition_frequency / plane_wave_angles.size
        )

        # Beamforming method.
        beamforming_method_bytes = np.array(recon_spec["method"][:]).flatten()
        beamforming_method = "".join(chr(int(c)) for c in beamforming_method_bytes)

        # Acquisition block structure.
        n_volumes_per_block = int(np.array(receive_spec["nRepeats"][:]).item(0))

    return EchoFrameMetadata(
        lateral_coords=lateral_coords,
        axial_coords=axial_coords,
        transmit_frequency=transmit_frequency,
        probe_number_of_elements=probe_n_elements,
        probe_pitch=probe_pitch,
        beamforming_sound_velocity=speed_of_sound,
        plane_wave_angles=plane_wave_angles,
        compound_sampling_frequency=compound_sampling_frequency,
        pulse_repetition_frequency=pulse_repetition_frequency,
        beamforming_method=beamforming_method,
        n_volumes_per_block=n_volumes_per_block,
    )


def _load_echoframe_block(
    dat_path: str | Path,
    block_idx: int,
    header_size: int,
    n_volumes_per_block: int,
    x: int,
    z: int,
    dat_dtype: npt.DTypeLike,
    padding_bytes: int,
) -> np.ndarray:
    """Load a single EchoFrame acquisition block from a DAT file.

    Opens a fresh, block-scoped `numpy.memmap` and copies the block out of it. Used as
    the per-chunk loader for the lazy Dask array returned by `load_echoframe_dat`; each
    block gets its own memory map so no single memmap spanning the whole file needs to
    be built or transferred across task boundaries.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to the EchoFrame DAT file containing beamformed IQ data.
    block_idx : int
        Index of the acquisition block to load.
    header_size : int
        Size in bytes of the DAT file header, preceding the first block.
    n_volumes_per_block : int
        Number of volumes per acquisition block.
    x : int
        Lateral dimension size.
    z : int
        Axial (depth) dimension size.
    dat_dtype : dtype_like
        Data type of the beamformed IQ data in the DAT file.
    padding_bytes : int
        Number of padding bytes following each block's data. `0` if blocks are
        contiguous.

    Returns
    -------
    (volumes, x, 1, z) numpy.ndarray
        Beamformed IQ data for the requested block.
    """
    block_shape = (n_volumes_per_block, x, 1, z)
    if padding_bytes > 0:
        # Block has trailing padding - use a structured dtype to skip it.
        block_dtype = np.dtype(
            [
                ("data", dat_dtype, block_shape),
                ("padding", np.uint8, (padding_bytes,)),
            ]
        )
        offset = header_size + block_idx * block_dtype.itemsize
        memmap = np.memmap(
            dat_path, dtype=block_dtype, mode="r", offset=offset, shape=(1,)
        )
        return np.array(memmap["data"][0])  # type: ignore
    else:
        itemsize = np.dtype(dat_dtype).itemsize
        offset = header_size + block_idx * int(np.prod(block_shape)) * itemsize
        memmap = np.memmap(
            dat_path, dtype=dat_dtype, mode="r", offset=offset, shape=block_shape
        )
        return np.array(memmap)


def _load_echoframe_dat_blocks(
    dat_path: str | Path,
    x: int,
    z: int,
    n_volumes_per_block: int,
    dat_dtype: npt.DTypeLike,
    header_dtype: npt.DTypeLike,
    n_header_items: int,
) -> da.Array:
    """Load all EchoFrame acquisition blocks as a lazy, EchoFrame-ordered Dask array.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to the EchoFrame DAT file containing beamformed IQ data.
    x : int
        Lateral dimension size.
    z : int
        Axial (depth) dimension size.
    n_volumes_per_block : int
        Number of volumes per acquisition block.
    dat_dtype : dtype_like
        Data type of the beamformed IQ data in the DAT file.
    header_dtype : dtype_like
        Data type of the DAT file header.
    n_header_items : int
        Number of items in the DAT file header.

    Returns
    -------
    (blocks, volumes, x, y, z) dask.array.Array
        Lazy array containing the beamformed IQ data, where `blocks` is the number of
        acquisition blocks, `volumes` is `n_volumes_per_block`, `x` is the lateral
        dimension, `y` is the elevation dimension (always `1`), and `z` is the axial
        dimension. Chunked one block per Dask chunk along the leading `blocks` axis;
        each block is only read from disk once its chunk is computed.
    """
    header = np.fromfile(dat_path, dtype=header_dtype, count=n_header_items)
    _, header_size, n_blocks, data_size, padding_bytes = (int(item) for item in header)

    load = dask.delayed(_load_echoframe_block)
    block_shape = (n_volumes_per_block, x, 1, z)
    blocks = [
        da.from_delayed(
            load(
                dat_path,
                block_idx,
                header_size,
                n_volumes_per_block,
                x,
                z,
                dat_dtype,
                padding_bytes,
            ),
            shape=block_shape,
            dtype=dat_dtype,
        )
        for block_idx in range(n_blocks)
    ]
    return da.stack(blocks, axis=0)


def load_echoframe_dat(
    dat_path: str | Path,
    meta_path: str | Path,
    dat_dtype: npt.DTypeLike = np.complex64,
    header_dtype: npt.DTypeLike = np.uint64,
    n_header_items: int = 5,
) -> xr.DataArray:
    """Load an EchoFrame DAT file as a lazy, ConfUSIus-ordered DataArray.

    Beamformed IQ data is loaded as a DataArray with dimensions `(time, z, y, x)`,
    matching ConfUSIus conventions. Coordinates (`time`, `z`, `y`, `x`) and acquisition
    metadata (e.g. `transmit_frequency`, `beamforming_sound_velocity`) are attached
    from the EchoFrame sequence parameter file. The `time` coordinate is computed from
    frame indices and the compound sampling frequency; callers with acquisition
    timestamps for each block should override it (e.g. via
    `data.assign_coords(time=...)`).

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to the EchoFrame DAT file containing beamformed IQ data.
    meta_path : str or pathlib.Path
        Path to the EchoFrame sequence parameter file (MAT v7.3 / HDF5 format).
    dat_dtype : dtype_like, default: numpy.complex64
        Data type of the beamformed IQ data in the DAT file.
    header_dtype : dtype_like, default: numpy.uint64
        Data type of the DAT file header.
    n_header_items : int, default: 5
        Number of items in the DAT file header.

    Returns
    -------
    xarray.DataArray
        Lazy DataArray with dimensions `(time, z, y, x)`, where `z` is a singleton
        elevation dimension. Data is wrapped in a Dask array, chunked so that each
        chunk corresponds to one acquisition block's volumes; individual blocks
        remain accessible via `data.isel(time=slice(i * n, (i + 1) * n))`, where `n`
        is `data.attrs["n_volumes_per_block"]`.

    Notes
    -----
    Acquisition metadata is stored in `da.attrs`, including `n_volumes_per_block`
    (number of volumes per acquisition block).
    """
    meta = load_echoframe_metadata(meta_path)
    x, z = len(meta["lateral_coords"]), len(meta["axial_coords"])
    n_volumes_per_block = meta["n_volumes_per_block"]

    dat_path = check_path(dat_path, label="dat_path", type="file")
    blocks = _load_echoframe_dat_blocks(
        dat_path, x, z, n_volumes_per_block, dat_dtype, header_dtype, n_header_items
    )
    n_blocks, n_volumes, _, _, _ = blocks.shape
    # EchoFrame axes are (blocks, volumes, x, y, z); transpose spatial axes to
    # ConfUSIus (z, y, x) and merge (blocks, volumes) into a single time axis. Each
    # Dask chunk is exactly one block, so this reshape never splits a chunk.
    iq = blocks.transpose((0, 1, 4, 3, 2)).reshape(n_blocks * n_volumes, 1, z, x)

    n_total_volumes = n_blocks * n_volumes
    time_values = (
        np.arange(n_total_volumes, dtype=np.float64)
        / meta["compound_sampling_frequency"]
    )
    volume_acquisition_duration = float(
        meta["plane_wave_angles"].size / meta["pulse_repetition_frequency"]
    )

    # TODO: we should compute the actual z-axis voxdim from the elevation beam width,
    # but we're currently missing some information for that, such as the elevation
    # aperture and elevation focus.
    coords = {
        "time": (
            "time",
            time_values,
            {
                "units": "s",
                "long_name": "Time",
                "volume_acquisition_reference": "start",
                "volume_acquisition_duration": volume_acquisition_duration,
            },
        ),
        "z": (
            "z",
            np.array([0.0]),
            {"units": "mm", "long_name": "Elevation", "voxdim": 0.4},
        ),
        "y": (
            "y",
            meta["axial_coords"],
            {
                "units": "mm",
                "long_name": "Depth",
                "voxdim": float(np.diff(meta["axial_coords"]).mean()),
            },
        ),
        "x": (
            "x",
            meta["lateral_coords"],
            {
                "units": "mm",
                "long_name": "Lateral",
                "voxdim": float(np.diff(meta["lateral_coords"]).mean()),
            },
        ),
    }
    attrs = {
        "transmit_frequency": meta["transmit_frequency"],
        "probe_number_of_elements": meta["probe_number_of_elements"],
        "probe_pitch": meta["probe_pitch"],
        "beamforming_sound_velocity": meta["beamforming_sound_velocity"],
        "plane_wave_angles": meta["plane_wave_angles"],
        "compound_sampling_frequency": meta["compound_sampling_frequency"],
        "pulse_repetition_frequency": meta["pulse_repetition_frequency"],
        "beamforming_method": meta["beamforming_method"],
        "n_volumes_per_block": n_volumes_per_block,
    }
    return xr.DataArray(
        iq, dims=("time", "z", "y", "x"), coords=coords, attrs=attrs, name="iq"
    )


def convert_echoframe_dat_to_zarr(
    dat_path: str | Path,
    meta_path: str | Path,
    output_path: str | Path,
    dat_dtype: npt.DTypeLike = np.complex64,
    header_dtype: npt.DTypeLike = np.uint64,
    n_header_items: int = 5,
    volumes_per_chunk: int | None = None,
    volumes_per_shard: int | None = None,
    batch_size: int = 100,
    overwrite: bool = False,
    zarr_kwargs: dict[str, Any] | None = None,
    skip_first_blocks: int = 0,
    skip_last_blocks: int = 0,
    block_times: npt.ArrayLike | None = None,
    show_progress: bool = True,
    progress: "Progress | None" = None,
    track_kwargs: dict[str, Any] | None = None,
) -> "zarr.Group":
    """Convert an EchoFrame DAT file to Zarr format compatible with Xarray.

    Beamformed IQ data is converted to a Zarr group with an `iq` array of shape
    `(time, z, y, x)` chunked along the first dimension. Coordinates are stored as
    separate Zarr arrays following Xarray conventions, allowing the data to be opened
    directly with `xarray.open_zarr()`.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to the EchoFrame DAT file containing beamformed IQ data.
    meta_path : str or pathlib.Path
        Path to the EchoFrame sequence parameter file (MAT v7.3 / HDF5 format).
    output_path : str or pathlib.Path
        Path where the Zarr group will be saved.
    dat_dtype : dtype_like, default: numpy.complex64
        Data type of the beamformed IQ data in the DAT file.
    header_dtype : dtype_like, default: numpy.uint64
        Data type of the DAT file header.
    n_header_items : int, default: 5
        Number of items in the DAT file header.
    volumes_per_chunk : int, optional
        Number of volumes to include in each Zarr chunk. If not provided, defaults to
        the number of volumes per block from the raw file.
    volumes_per_shard : int, optional
        Number of volumes to include in each shard. If provided, enables Zarr v3
        sharding to reduce the number of files on disk. Must be a multiple of
        `volumes_per_chunk`. If not provided, sharding is disabled.
    batch_size : int, default: 100
        Number of blocks to process in each batch.
    overwrite : bool, default: False
        Whether to overwrite existing Zarr group at the output path.
    zarr_kwargs : dict, optional
        Additional keyword arguments to pass to `zarr.create_array` for the main data
        array.
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

    Notes
    -----
    The output Zarr group follows Xarray conventions and can be opened with::

        import xarray as xr
        ds = xr.open_zarr("output.zarr")
        iq = ds["iq"]

    Metadata attributes (e.g., `transmit_frequency`, `beamforming_sound_velocity`) are stored
    on the `iq` DataArray (accessible via `iq.attrs`), consistent with how
    reduction functions return DataArrays with attributes.

    The group contains:

    - `iq`: The main data array with dimensions `(time, z, y, x)`.
    - `time`: Time coordinate array.
    - `z`: Elevation coordinate array (always `[0]` for 2D data).
    - `y`: Axial (depth) coordinate array (from metadata).
    - `x`: Lateral coordinate array (from metadata).
    """
    from rich.progress import track

    iq_da = load_echoframe_dat(
        dat_path=dat_path,
        meta_path=meta_path,
        dat_dtype=dat_dtype,
        header_dtype=header_dtype,
        n_header_items=n_header_items,
    )

    n_volumes = iq_da.attrs["n_volumes_per_block"]
    n_total_volumes_full = iq_da.sizes["time"]
    n_blocks = n_total_volumes_full // n_volumes

    total_skip = skip_first_blocks + skip_last_blocks
    if total_skip >= n_blocks:
        raise ValueError(
            f"Cannot skip {total_skip} blocks (skip_first_blocks={skip_first_blocks}, "
            f"skip_last_blocks={skip_last_blocks}) from {n_blocks} total blocks."
        )

    iq_da = iq_da.isel(
        time=slice(
            skip_first_blocks * n_volumes,
            n_total_volumes_full - skip_last_blocks * n_volumes,
        )
    )
    n_blocks_after_skip = n_blocks - total_skip
    n_total_volumes = iq_da.sizes["time"]
    n_z, n_x = iq_da.sizes["y"], iq_da.sizes["x"]

    if volumes_per_chunk is None:
        volumes_per_chunk = n_volumes

    output_shape = (n_total_volumes, 1, n_z, n_x)
    chunks = (volumes_per_chunk, 1, n_z, n_x)

    if volumes_per_shard is not None:
        if volumes_per_shard % volumes_per_chunk != 0:
            raise ValueError(
                f"volumes_per_shard ({volumes_per_shard}) must be a multiple of "
                f"volumes_per_chunk ({volumes_per_chunk})."
            )
        shards = (volumes_per_shard, 1, n_z, n_x)
    else:
        shards = None

    # Dimension names required for Zarr v3 / Xarray compatibility.
    dim_names = ["time", "z", "y", "x"]

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
    create_array_kwargs["dtype"] = iq_da.dtype
    create_array_kwargs["dimension_names"] = dim_names

    if shards is not None:
        create_array_kwargs["shards"] = shards

    if block_times is not None:
        block_times_array = np.asarray(block_times)
        if block_times_array.size != n_blocks_after_skip:
            raise ValueError(
                f"block_times length ({block_times_array.size}) does not match "
                f"number of blocks after skipping ({n_blocks_after_skip})."
            )

        time_values = np.concatenate(
            [
                block_start
                + np.arange(n_volumes) / iq_da.attrs["compound_sampling_frequency"]
                for block_start in block_times_array
            ]
        )
        iq_da = iq_da.assign_coords(time=("time", time_values))

    zarr_group = zarr.open_group(output_path, mode="w" if overwrite else "w-")
    zarr_iq = zarr_group.create_array("iq", **create_array_kwargs)

    for dim in ("time", "z", "y", "x"):
        zarr_group.create_array(
            dim, data=iq_da.coords[dim].values, dimension_names=[dim]
        )
        zarr_group[dim].attrs.update(iq_da.coords[dim].attrs)

    for key, value in iq_da.attrs.items():
        if key == "n_volumes_per_block":
            continue
        zarr_iq.attrs[key] = value.tolist() if isinstance(value, np.ndarray) else value

    first_block = skip_first_blocks
    last_block = n_blocks - skip_last_blocks

    n_batches = (n_blocks_after_skip + batch_size - 1) // batch_size
    batches = range(first_block, last_block, batch_size)
    task_id = None

    if not show_progress:
        iterable = batches
    elif progress is not None:
        task_id = progress.add_task(
            "Converting EchoFrame DAT to Zarr...", total=n_batches
        )
        iterable = batches
    else:
        kwargs = track_kwargs or {}
        kwargs.setdefault("description", "Converting EchoFrame DAT to Zarr...")
        iterable = track(batches, **kwargs)

    try:
        for idx, start_block in enumerate(iterable):
            end_block = min(start_block + batch_size, last_block)

            output_start = (start_block - skip_first_blocks) * n_volumes
            output_end = (end_block - skip_first_blocks) * n_volumes
            batch_data = iq_da.data[output_start:output_end].compute()

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
        output_path = Path(output_path)
        if output_path.exists():
            shutil.rmtree(output_path)
        raise

    if progress is not None and task_id is not None:
        progress.update(task_id, visible=False)

    return zarr_group
