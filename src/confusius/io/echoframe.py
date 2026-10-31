"""Utilities for loading EchoFrame DAT files."""

from pathlib import Path
from typing import TypedDict

import dask.array as da
import h5py as h5
import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius._dims import TIME_DIM, VOXEL_DIMS
from confusius.io._utils import map_binary_array
from confusius.io.utils import check_path
from confusius.xarray.create import create_voxeldata


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
        # recon_spec["xAxis"] is lateral (x dimension in ConfUSIus).
        # recon_spec["zAxis"] is depth/axial (y dimension in ConfUSIus).
        x_axis_name = "xAxis" if "xAxis" in recon_spec else "x_axis"
        z_axis_name = "zAxis" if "zAxis" in recon_spec else "z_axis"
        x_axis_full = np.array(recon_spec[x_axis_name][:]).flatten()
        z_axis_full = np.array(recon_spec[z_axis_name][:]).flatten()

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
    _, header_size, n_blocks, _data_size, padding_bytes = (int(item) for item in header)

    block_shape = (n_volumes_per_block, x, 1, z)
    if padding_bytes:
        record_dtype = np.dtype(
            [("data", dat_dtype, block_shape), ("padding", np.uint8, (padding_bytes,))]
        )
        return map_binary_array(dat_path, (n_blocks,), record_dtype, header_size, 1)[
            "data"
        ]
    shape = (n_blocks, *block_shape)
    return map_binary_array(dat_path, shape, dat_dtype, header_size, (1, *block_shape))


def load_echoframe_dat(
    dat_path: str | Path,
    meta_path: str | Path | None = None,
    dat_dtype: npt.DTypeLike = np.complex64,
    header_dtype: npt.DTypeLike = np.uint64,
    n_header_items: int = 5,
) -> xr.DataArray:
    """Load an EchoFrame DAT file as a lazy, ConfUSIus-ordered DataArray.

    Beamformed IQ data is loaded as a VoxelData array with voxel
    dimensions `(time, k, j, i)` and a voxel-to-world index deriving world coordinates
    `z`, `y`, `x`. Coordinates and acquisition metadata (e.g. `transmit_frequency`,
    `beamforming_sound_velocity`) are attached from the EchoFrame sequence parameter
    file. The `time` coordinate is computed from frame indices and the compound
    sampling frequency; callers with acquisition timestamps for each block should
    override it (e.g. via `data.assign_coords(time=...)`). Each chunk reopens a
    read-only memory map on its worker; in-place processing requires a copy.

    Parameters
    ----------
    dat_path : str or pathlib.Path
        Path to the EchoFrame DAT file containing beamformed IQ data.
    meta_path : str or pathlib.Path, optional
        Path to the EchoFrame sequence parameter file (MAT v7.3 / HDF5 format). If not
        provided, looks for `ScanParameters.mat` next to `dat_path`, matching the
        filename EchoFrame's storage writer uses by convention.
    dat_dtype : dtype_like, default: numpy.complex64
        Data type of the beamformed IQ data in the DAT file.
    header_dtype : dtype_like, default: numpy.uint64
        Data type of the DAT file header.
    n_header_items : int, default: 5
        Number of items in the DAT file header.

    Returns
    -------
    xarray.DataArray
        Lazy VoxelData array with dimensions `(time, k, j, i)`, where
        `k` is a singleton
        elevation dimension. Data is wrapped in a Dask array, chunked so that each
        chunk corresponds to one acquisition block's volumes; individual blocks
        remain accessible via `data.isel(time=slice(i * n, (i + 1) * n))`, where `n`
        is `data.attrs["n_volumes_per_block"]`.

    Raises
    ------
    ValueError
        If `meta_path` is not provided and no `ScanParameters.mat` file exists next to
        `dat_path`.

    Notes
    -----
    Acquisition metadata is stored in `da.attrs`, including `n_volumes_per_block`
    (number of volumes per acquisition block).
    """
    dat_path = check_path(dat_path, label="dat_path", type="file")
    if meta_path is None:
        meta_path = dat_path.parent / "ScanParameters.mat"

    meta = load_echoframe_metadata(meta_path)
    x, z = len(meta["lateral_coords"]), len(meta["axial_coords"])
    n_volumes_per_block = meta["n_volumes_per_block"]

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

    # TODO: we should compute the actual z-axis spacing from the elevation beam width,
    # but we're currently missing some information for that, such as the elevation
    # aperture and elevation focus.
    z_spacing = 0.4
    y_spacing = float(np.diff(meta["axial_coords"]).mean())
    x_spacing = float(np.diff(meta["lateral_coords"]).mean())
    voxel_to_world = np.eye(4)
    voxel_to_world[:3, :3] = np.diag([z_spacing, y_spacing, x_spacing])
    voxel_to_world[:3, 3] = [0.0, meta["axial_coords"][0], meta["lateral_coords"][0]]

    attrs = {
        "probe_number_of_elements": meta["probe_number_of_elements"],
        "probe_pitch": meta["probe_pitch"],
        "plane_wave_angles": meta["plane_wave_angles"],
        "compound_sampling_frequency": meta["compound_sampling_frequency"],
        "pulse_repetition_frequency": meta["pulse_repetition_frequency"],
        "beamforming_method": meta["beamforming_method"],
        "n_volumes_per_block": n_volumes_per_block,
        "transmit_frequency": meta["transmit_frequency"],
        "beamforming_sound_velocity": meta["beamforming_sound_velocity"],
    }
    return create_voxeldata(
        iq,
        dims=(TIME_DIM, *VOXEL_DIMS),
        time=xr.DataArray(time_values, dims=["time"], attrs={"long_name": "Time"}),
        volume_acquisition_reference="start",
        volume_acquisition_duration=volume_acquisition_duration,
        voxel_to_world=voxel_to_world,
        attrs=attrs,
    )
