"""Utilities for loading Iconeus RAW beamformed IQ files.

Iconeus stores beamformed IQ data as a binary `.raw` payload paired with an HDF5
`.hraw` metadata sidecar. The payload contains interleaved little-endian `float32`
real and imaginary samples in Fortran order, with axes
`(real/imag, Z, Y, X, compound, frames, blocks)`. The binary layout follows
PyIconeus, which is licensed under the BSD-3-Clause License; see the `NOTICE` file
for details.
"""

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

_LEGACY_PAYLOAD_OFFSET = 117
"""Number of header bytes preceding the payload in legacy Iconeus RAW files."""


class IconeusRawMetadata(TypedDict):
    """Metadata extracted from an Iconeus HRAW file.

    Attributes
    ----------
    size_x : int
        Number of lateral samples.
    size_y : int
        Number of elevation samples.
    size_z : int
        Number of axial (depth) samples.
    n_compound : int
        Number of compounded images per frame.
    n_frames_per_block : int
        Number of frames per acquisition block.
    n_blocks : int
        Number of acquisition blocks.
    lateral_spacing : float
        Lateral voxel spacing in millimeters.
    elevation_spacing : float
        Elevation voxel spacing in millimeters.
    axial_spacing : float
        Axial (depth) voxel spacing in millimeters.
    depth_near : float
        Near imaging depth in millimeters.
    depth_far : float
        Far imaging depth in millimeters.
    transmit_frequency : float
        Central transmit frequency in hertz.
    pulse_repetition_frequency : float
        Pulse repetition frequency in hertz.
    beamforming_sound_velocity : float
        Speed of sound in meters per second used during beamforming.
    frame_rate : float
        Frame rate of the stored (compounded) frames in hertz.
    plane_wave_angles : numpy.ndarray
        Angles at which tilted plane waves are emitted in degrees.
    receive_aperture : numpy.ndarray
        Receive aperture samples.
    compound : bool
        Whether the stored frames are compounded images.
    is_legacy_format : bool
        Whether the payload is preceded by the legacy 117-byte header.
    acquisition_mode : str
        Iconeus acquisition mode (e.g. `"2Dscan"`).
    """

    size_x: int
    size_y: int
    size_z: int
    n_compound: int
    n_frames_per_block: int
    n_blocks: int
    lateral_spacing: float
    elevation_spacing: float
    axial_spacing: float
    depth_near: float
    depth_far: float
    transmit_frequency: float
    pulse_repetition_frequency: float
    beamforming_sound_velocity: float
    frame_rate: float
    plane_wave_angles: npt.NDArray[np.float64]
    receive_aperture: npt.NDArray[np.float64]
    compound: bool
    is_legacy_format: bool
    acquisition_mode: str


def _decrypt_hraw_values(dataset: h5.Dataset, index: int) -> npt.NDArray[np.float64]:
    """Decrypt one HRAW metadata field.

    Early Iconeus RAW headers obfuscate scalar values with a fixed affine transform.
    The transform is only applied to low-dimensional arrays, matching PyIconeus.

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset holding the encoded field.
    index : int
        One-based field index used by the encoding.

    Returns
    -------
    numpy.ndarray
        Decrypted field values.
    """
    values = np.asarray(dataset[()], dtype=np.float64)
    if values.ndim < 3:
        values = (values - 72) / (1005 * index)
    return values


def _first_hraw_scalar(dataset: h5.Dataset, index: int) -> float:
    """Decrypt a scalar HRAW field and return its first value.

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset holding the encoded field.
    index : int
        One-based field index used by the encoding.

    Returns
    -------
    float
        First decrypted value.

    Raises
    ------
    ValueError
        If the field is empty.
    """
    values = _decrypt_hraw_values(dataset, index).reshape(-1)
    if values.size == 0:
        raise ValueError(f"HRAW field F{index} is empty.")
    return float(values[0])


def _read_hraw_string(dataset: h5.Dataset) -> str:
    """Read and decode a string HRAW field.

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset holding the encoded string.

    Returns
    -------
    str
        Decoded, whitespace-trimmed string.
    """
    value = dataset.asstr()[()]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else ""
    return value.rstrip("\x00").strip()


def load_iconeus_raw_metadata(meta_path: str | Path) -> IconeusRawMetadata:
    """Load acquisition metadata from an Iconeus HRAW file.

    Parameters
    ----------
    meta_path : str or pathlib.Path
        Path to the Iconeus HRAW metadata sidecar.

    Returns
    -------
    IconeusRawMetadata
        Dictionary containing the extracted metadata fields.

    Raises
    ------
    ValueError
        If `meta_path` is not a file, a required field is missing, or the stored
        dimensions are invalid, or the depth bounds are non-finite or reversed.
    """
    meta_path = check_path(meta_path, label="meta_path", type="file")

    with h5.File(meta_path, "r") as f:
        missing = [name for name in (f"F{i}" for i in range(1, 14)) if name not in f]
        if missing:
            raise ValueError(
                f"Invalid Iconeus HRAW file {meta_path}: missing fields {missing}."
            )

        transmit_frequency_mhz = _first_hraw_scalar(f["F1"], 1)
        pulse_repetition_frequency = _first_hraw_scalar(f["F2"], 2)
        beamforming_sound_velocity = _first_hraw_scalar(f["F3"], 3)
        frame_rate = _first_hraw_scalar(f["F4"], 4)
        receive_aperture = _decrypt_hraw_values(f["F5"], 5).reshape(-1)
        depth = _decrypt_hraw_values(f["F6"], 6).reshape(-1)
        plane_wave_angles = _decrypt_hraw_values(f["F7"], 7).reshape(-1)
        voxel_spacing = _decrypt_hraw_values(f["F8"], 8).reshape(-1)
        block_dimensions = _decrypt_hraw_values(f["F9"], 9).reshape(-1)
        compound = bool(_first_hraw_scalar(f["F10"], 10))
        n_blocks_value = _first_hraw_scalar(f["F11"], 11)
        is_legacy_format = bool(_first_hraw_scalar(f["F12"], 12))
        acquisition_mode = _read_hraw_string(f["F13"])

    if depth.size < 2:
        raise ValueError("HRAW depth must contain near and far values.")
    if voxel_spacing.size < 3:
        raise ValueError("HRAW voxel spacing must contain three values.")
    if block_dimensions.size < 5:
        raise ValueError("HRAW block dimensions must contain five values.")

    # PyIconeus stores block dimensions as (sizeX, sizeY, sizeZ, compound, frames).
    dimension_values = np.append(block_dimensions[:5], n_blocks_value)
    rounded_dimensions = np.rint(dimension_values)
    if (
        not np.isfinite(dimension_values).all()
        or not np.allclose(dimension_values, rounded_dimensions, rtol=0, atol=1e-8)
        or (rounded_dimensions <= 0).any()
    ):
        raise ValueError(
            f"HRAW dimensions must be finite positive integers, got {dimension_values}."
        )
    size_x, size_y, size_z, n_compound, n_frames_per_block, n_blocks = (
        int(value) for value in rounded_dimensions
    )

    lateral_spacing, elevation_spacing, axial_spacing = (
        float(value) for value in voxel_spacing[:3]
    )
    depth_near, depth_far = (float(value) for value in depth[:2])
    transmit_frequency = transmit_frequency_mhz * 1e6
    positive_fields = {
        "lateral spacing": lateral_spacing,
        "elevation spacing": elevation_spacing,
        "axial spacing": axial_spacing,
        "transmit frequency": transmit_frequency,
        "pulse repetition frequency": pulse_repetition_frequency,
        "beamforming sound velocity": beamforming_sound_velocity,
        "frame rate": frame_rate,
    }
    for name, value in positive_fields.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"HRAW {name} must be positive, got {value}.")
    # NaN bounds bypass the ordering check below.
    if not np.isfinite([depth_near, depth_far]).all():
        raise ValueError(
            f"HRAW depth bounds must be finite, got near={depth_near}, far={depth_far}."
        )
    if depth_far < depth_near:
        raise ValueError(
            "HRAW far depth must be greater than or equal to near depth, got "
            f"{depth_far} < {depth_near}."
        )

    return IconeusRawMetadata(
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        n_compound=n_compound,
        n_frames_per_block=n_frames_per_block,
        n_blocks=n_blocks,
        lateral_spacing=lateral_spacing,
        elevation_spacing=elevation_spacing,
        axial_spacing=axial_spacing,
        depth_near=depth_near,
        depth_far=depth_far,
        transmit_frequency=transmit_frequency,
        pulse_repetition_frequency=pulse_repetition_frequency,
        beamforming_sound_velocity=beamforming_sound_velocity,
        frame_rate=frame_rate,
        plane_wave_angles=plane_wave_angles,
        receive_aperture=receive_aperture,
        compound=compound,
        is_legacy_format=is_legacy_format,
        acquisition_mode=acquisition_mode,
    )


def _load_iconeus_raw_blocks(
    raw_path: str | Path, meta: IconeusRawMetadata
) -> da.Array:
    """Map an Iconeus RAW payload as a lazy, block-chunked complex array.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Path to the Iconeus RAW payload.
    meta : IconeusRawMetadata
        Decoded HRAW metadata.

    Returns
    -------
    dask.array.Array
        Lazy array with dimensions `(time, k, j, i)`, chunked so each chunk holds one
        acquisition block's frames.
    """
    size_x = meta["size_x"]
    size_y = meta["size_y"]
    size_z = meta["size_z"]
    n_compound = meta["n_compound"]
    n_frames = meta["n_frames_per_block"]
    n_blocks = meta["n_blocks"]

    offset = _LEGACY_PAYLOAD_OFFSET if meta["is_legacy_format"] else 0
    # Complex64 is laid out as interleaved real/imaginary float32 pairs, so a
    # Fortran-ordered complex array over the remaining axes maps the documented
    # float payload without an explicit real/imaginary combination step.
    payload = map_binary_array(
        raw_path,
        (size_z, size_y, size_x, n_compound, n_frames, n_blocks),
        "<c8",
        offset,
        (size_z, size_y, size_x, n_compound, n_frames, 1),
        order="F",
    )
    payload = da.squeeze(payload, axis=3)
    # (blocks, frames, z, y, x) -> (time, elevation, depth, lateral).
    blocks_first = da.transpose(payload, (4, 3, 0, 1, 2))
    merged = blocks_first.reshape(n_blocks * n_frames, size_z, size_y, size_x)
    return da.transpose(merged, (0, 2, 1, 3))


def _build_iconeus_raw_voxel_to_world(
    meta: IconeusRawMetadata,
) -> npt.NDArray[np.float64]:
    """Build a voxel-to-world affine for an Iconeus RAW acquisition.

    The HRAW metadata describes the probe frame: lateral and elevation voxels are
    centered on zero, and depth increases away from the probe starting at
    `depth_near`.

    Parameters
    ----------
    meta : IconeusRawMetadata
        Decoded HRAW metadata.

    Returns
    -------
    (4, 4) numpy.ndarray
        Affine mapping `(k, j, i)` voxel coordinates to `(z, y, x)` world
        coordinates in millimeters.
    """
    lateral_origin = -((meta["size_x"] - 1) / 2) * meta["lateral_spacing"]
    elevation_origin = -((meta["size_y"] - 1) / 2) * meta["elevation_spacing"]
    voxel_to_world = np.eye(4, dtype=np.float64)
    voxel_to_world[:3, :3] = np.diag(
        [meta["elevation_spacing"], meta["axial_spacing"], meta["lateral_spacing"]]
    )
    voxel_to_world[:3, 3] = [
        elevation_origin,
        meta["depth_near"],
        lateral_origin,
    ]
    return voxel_to_world


def load_iconeus_raw(
    raw_path: str | Path,
    meta_path: str | Path | None = None,
) -> xr.DataArray:
    """Load an Iconeus RAW file as a lazy, ConfUSIus-ordered DataArray.

    Beamformed IQ data is loaded as a VoxelData array with voxel dimensions
    `(time, k, j, i)` and a voxel-to-world index deriving world coordinates
    `z`, `y`, `x`. Coordinates and acquisition metadata are attached from the HRAW
    sidecar. Acquisition blocks are preserved as Dask chunks: each chunk holds one
    block's frames, given by the per-block frame counts in `data.chunks[0]`. The
    `time` coordinate is theoretical: it is computed from frame indices and the HRAW
    frame rate, because RAW/HRAW files store no measured per-frame timestamps. Callers
    with acquisition timestamps should override it (e.g. via
    `data.assign_coords(time=...)`). Each chunk reopens a read-only memory map on its
    worker; in-place processing requires a copy.

    Only compounded acquisitions stored as a single compound image per frame are
    supported. This is the layout used by standard Iconeus fUSI acquisitions.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Path to the Iconeus RAW file containing beamformed IQ data.
    meta_path : str or pathlib.Path, optional
        Path to the Iconeus HRAW metadata sidecar. If not provided, looks for a file
        with the same stem and the `.hraw` suffix next to `raw_path`.

    Returns
    -------
    xarray.DataArray
        Lazy VoxelData array with dimensions `(time, k, j, i)`, where `k` is
        elevation, `j` is depth, and `i` is lateral.

    Raises
    ------
    ValueError
        If the acquisition is not compounded, if the compound axis is not singleton,
        if the HRAW dimensions are invalid, or if the payload is truncated.

    Notes
    -----
    Acquisition metadata is stored in `da.attrs`, including the BIDS-compatible
    `transmit_frequency` (hertz), `pulse_repetition_frequency` (hertz),
    `beamforming_sound_velocity` (meters per second), `compound_sampling_frequency`
    (hertz), `plane_wave_angles` (degrees), `receive_aperture`, and `depth` (near and
    far imaging depth in millimeters), plus the Iconeus-specific
    `iconeus_acquisition_mode` and `iconeus_legacy_format`.

    Blocks are assumed to be contiguous, so the uniform `time` coordinate does not
    represent acquisition gaps between them. Pass explicit acquisition timestamps when
    block timing is known to differ. The theoretical timing is also exposed as
    `volume_acquisition_duration = 1 / frame_rate`.
    """
    raw_path = check_path(raw_path, label="raw_path", type="file")
    if meta_path is None:
        meta_path = raw_path.with_suffix(".hraw")

    meta = load_iconeus_raw_metadata(meta_path)
    if not meta["compound"]:
        raise ValueError(
            "Iconeus RAW file is not compounded. Only compounded acquisitions are "
            "currently supported."
        )
    if meta["n_compound"] != 1:
        raise ValueError(
            "Iconeus RAW file stores multiple compound images per frame "
            f"({meta['n_compound']}); only single-image compounded acquisitions are "
            "currently supported."
        )

    iq = _load_iconeus_raw_blocks(raw_path, meta)
    n_time = meta["n_blocks"] * meta["n_frames_per_block"]
    time_values = np.arange(n_time, dtype=np.float64) / meta["frame_rate"]

    attrs = {
        "transmit_frequency": meta["transmit_frequency"],
        "beamforming_sound_velocity": meta["beamforming_sound_velocity"],
        "pulse_repetition_frequency": meta["pulse_repetition_frequency"],
        "compound_sampling_frequency": meta["frame_rate"],
        "plane_wave_angles": meta["plane_wave_angles"].tolist(),
        "receive_aperture": [
            int(value) for value in np.round(meta["receive_aperture"])
        ],
        "depth": (meta["depth_near"], meta["depth_far"]),
        "iconeus_acquisition_mode": meta["acquisition_mode"],
        "iconeus_legacy_format": meta["is_legacy_format"],
    }
    return create_voxeldata(
        iq,
        dims=(TIME_DIM, *VOXEL_DIMS),
        time=xr.DataArray(time_values, dims=[TIME_DIM], attrs={"long_name": "Time"}),
        volume_acquisition_duration=1 / meta["frame_rate"],
        voxel_to_world=_build_iconeus_raw_voxel_to_world(meta),
        attrs=attrs,
        name=raw_path.stem,
    )
