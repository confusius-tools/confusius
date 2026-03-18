"""Utilities for loading and converting EchoFrame DAT files."""

import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import h5py as h5
import numpy as np
import numpy.typing as npt
import zarr

if TYPE_CHECKING:
    from rich.progress import Progress

from confusius._utils import find_stack_level
from confusius.io.utils import check_path


class EchoFrameMetadata(TypedDict):
    """Metadata extracted from an EchoFrame MAT file.

    Attributes
    ----------
    lateral_coords : numpy.ndarray
        Lateral coordinates in millimeters, cropped if the sequence uses ROI cropping.
    axial_coords : numpy.ndarray
        Axial (depth) coordinates in millimeters, cropped if the sequence uses ROI
        cropping.
    transmit_frequency : float
        Central frequency of the ultrasound probe in hertz.
    probe_n_elements : int
        Number of probe transducer elements.
    probe_pitch : float
        Inter-element pitch of the probe in millimeters.
    speed_of_sound : float
        Speed of sound in meters per second.
    plane_wave_angles : numpy.ndarray
        Angles at which tilted plane waves are emitted in degrees.
    compound_sampling_frequency : float
        Sampling frequency of the compounded frames in hertz.
    pulse_repetition_frequency : float
        Single plane wave pulse repetition frequency in hertz.
    beamforming_method : str
        Beamforming method used (e.g. `"DAS"`).
    """

    lateral_coords: npt.NDArray[np.float64]
    axial_coords: npt.NDArray[np.float64]
    transmit_frequency: float
    probe_n_elements: int
    probe_pitch: float
    speed_of_sound: float
    plane_wave_angles: npt.NDArray[np.float64]
    compound_sampling_frequency: float
    pulse_repetition_frequency: float
    beamforming_method: str


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
        probe_pitch = float(np.array(probe_spec["pitchX"][:]).item())

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

    return EchoFrameMetadata(
        lateral_coords=lateral_coords,
        axial_coords=axial_coords,
        transmit_frequency=transmit_frequency,
        probe_n_elements=probe_n_elements,
        probe_pitch=probe_pitch,
        speed_of_sound=speed_of_sound,
        plane_wave_angles=plane_wave_angles,
        compound_sampling_frequency=compound_sampling_frequency,
        pulse_repetition_frequency=pulse_repetition_frequency,
        beamforming_method=beamforming_method,
    )


def load_echoframe_dat(
    dat_path: str | Path,
    meta_path: str | Path,
    dat_dtype: npt.DTypeLike = np.complex64,
    header_dtype: npt.DTypeLike = np.uint64,
    n_header_items: int = 5,
) -> np.memmap:
    """Load an EchoFrame DAT file containing beamformed IQ data.

    !!! warning "EchoFrame spatial dimensions"
        EchoFrame stores data with `(x, y, z)` spatial ordering corresponding to
        (lateral, elevation, axial) dimensions, which is different from the `(z, y,
        x)` ordering used in ConfUSIus that corresponds to (elevation, axial, lateral).
        The returned array maintains the EchoFrame ordering to avoid confusion, but
        users should be aware of this when processing the data.

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
    (blocks, volumes, x, y, z) numpy.memmap
        Memory-mapped array containing the beamformed IQ data, where `blocks` is the
        number of acquisitions blocks, `volumes` is the number of volumes per block,
        `x` is the lateral dimension, `y` is the elevation dimension, and `z` is
        the axial dimension.
    """
    dat_path = check_path(dat_path, label="dat_path", type="file")
    meta_path = check_path(meta_path, label="meta_path", type="file")

    with h5.File(meta_path, "r") as f:
        recon_spec = f["ReconSpec"]
        receive_spec = f["ReceiveSpec"]
        crop = (
            bool(np.array(recon_spec["cropBF"][()]))
            if "cropBF" in recon_spec
            else False
        )

        if crop:
            z = int(
                np.array(recon_spec["croppingROI"][0, 1])
                - np.array(recon_spec["croppingROI"][0, 0])
                + 1
            )
            x = int(
                np.array(recon_spec["croppingROI"][0, 3])
                - np.array(recon_spec["croppingROI"][0, 2])
                + 1
            )
        else:
            z = int(recon_spec["nz"][0, 0])
            x = int(recon_spec["nx"][0, 0])
        n_volumes_per_block = np.array(receive_spec["nRepeats"][:]).item(0)

    header = np.fromfile(dat_path, dtype=header_dtype, count=n_header_items)
    _, header_size, n_blocks, data_size, padding_bytes = header

    if padding_bytes > 0:
        # File has padding between blocks - use structured dtype to skip it.
        block_dtype = np.dtype(
            [
                ("data", dat_dtype, (int(n_volumes_per_block), x, 1, z)),
                ("padding", np.uint8, (int(padding_bytes),)),
            ]
        )
        memmap = np.memmap(
            dat_path,
            dtype=block_dtype,
            mode="r",
            offset=header_size,
            shape=(int(n_blocks),),
        )
        return memmap["data"]  # type: ignore
    else:
        # No padding - direct memmap.
        return np.memmap(
            dat_path,
            dtype=dat_dtype,
            mode="r",
            offset=header_size,
            shape=(int(n_blocks), int(n_volumes_per_block), x, 1, z),
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

    Metadata attributes (e.g., `transmit_frequency`, `sound_velocity`) are stored
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

    meta = load_echoframe_metadata(meta_path)
    data = load_echoframe_dat(
        dat_path=dat_path,
        meta_path=meta_path,
        dat_dtype=dat_dtype,
        header_dtype=header_dtype,
        n_header_items=n_header_items,
    )

    n_blocks, n_volumes, n_x, n_y, n_z = data.shape

    total_skip = skip_first_blocks + skip_last_blocks
    if total_skip >= n_blocks:
        raise ValueError(
            f"Cannot skip {total_skip} blocks (skip_first_blocks={skip_first_blocks}, "
            f"skip_last_blocks={skip_last_blocks}) from {n_blocks} total blocks."
        )

    n_blocks_after_skip = n_blocks - total_skip
    n_total_volumes = n_blocks_after_skip * n_volumes

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
    create_array_kwargs["dtype"] = data.dtype
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
                block_start + np.arange(n_volumes) / meta["compound_sampling_frequency"]
                for block_start in block_times_array
            ]
        )
    else:
        skipped_volumes = skip_first_blocks * n_volumes
        time_values = (
            np.arange(n_total_volumes, dtype=np.float64)
            / meta["compound_sampling_frequency"]
            + skipped_volumes / meta["compound_sampling_frequency"]
        )

    zarr_group = zarr.open_group(output_path, mode="w" if overwrite else "w-")
    zarr_iq = zarr_group.create_array("iq", **create_array_kwargs)

    zarr_group.create_array("time", data=time_values, dimension_names=["time"])
    zarr_group["time"].attrs["units"] = "s"
    zarr_group["time"].attrs["long_name"] = "Time"

    # z coordinate is the stacking dimension (size 1 for 2D data).
    z_values = np.array([0.0])
    zarr_group.create_array("z", data=z_values, dimension_names=["z"])
    zarr_group["z"].attrs["units"] = "mm"
    zarr_group["z"].attrs["long_name"] = "Elevation"

    zarr_group.create_array("y", data=meta["axial_coords"], dimension_names=["y"])
    zarr_group["y"].attrs["units"] = "mm"
    zarr_group["y"].attrs["long_name"] = "Depth"

    zarr_group.create_array("x", data=meta["lateral_coords"], dimension_names=["x"])
    zarr_group["x"].attrs["units"] = "mm"
    zarr_group["x"].attrs["long_name"] = "Lateral"

    # TODO: we should compute the actual z-axis voxdim from the elevation beam width,
    # but we're currently missing some information for that, such as the elevation
    # aperture and elevation focus.
    zarr_group["z"].attrs["voxdim"] = 0.4
    zarr_group["y"].attrs["voxdim"] = float(np.diff(meta["axial_coords"]).mean())
    zarr_group["x"].attrs["voxdim"] = float(np.diff(meta["lateral_coords"]).mean())
    zarr_iq.attrs["transmit_frequency"] = meta["transmit_frequency"]
    zarr_iq.attrs["probe_n_elements"] = meta["probe_n_elements"]
    zarr_iq.attrs["probe_pitch"] = meta["probe_pitch"]
    zarr_iq.attrs["sound_velocity"] = meta["speed_of_sound"]
    zarr_iq.attrs["plane_wave_angles"] = meta["plane_wave_angles"].tolist()
    zarr_iq.attrs["compound_sampling_frequency"] = meta["compound_sampling_frequency"]
    zarr_iq.attrs["pulse_repetition_frequency"] = meta["pulse_repetition_frequency"]
    zarr_iq.attrs["beamforming_method"] = meta["beamforming_method"]

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

            batch_data = np.transpose(
                data[start_block:end_block], axes=(0, 1, 4, 3, 2)
            ).reshape(-1, 1, n_z, n_x)

            output_start = (start_block - skip_first_blocks) * n_volumes
            output_end = (end_block - skip_first_blocks) * n_volumes
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
