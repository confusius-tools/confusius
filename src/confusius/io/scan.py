"""Utilities for loading Iconeus SCAN files.

Iconeus ships two on-disk SCAN formats, both using the `.scan` extension:

- **v1**: an HDF5 container (`acqMetaData`, `scanMetaData`, `/Data`). Loaded lazily with
  h5py and Dask.
- **v2**: a flat binary file with a variable-length header followed by a little-endian
  `float64` power-Doppler payload. Loaded lazily with a NumPy memmap wrapped in Dask.

`load_scan` sniffs the format and dispatches to the matching loader. The v2 loader
follows the binary layout published by PyIconeus and keeps the payload lazy.
"""

import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dask.array as da
import h5py
import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.io.utils import check_path
from confusius.multipose.timing import build_consolidated_time_coordinate
from confusius.xarray.create import create_voxeldata

SCAN_V2_MAGIC = b"scan"
"""Magic bytes at offset 0 identifying a binary SCAN v2 file."""

_SCAN_V2_OFFSETS: dict[str, int] = {
    "total_header_bytes": 0x20,
    "payload_bytes": 0x28,
    "size_x": 0x5C,
    "size_y": 0x64,
    "size_z": 0x6C,
    "n_time": 0x74,
    "npose": 0x7C,
    "dim6_count": 0x84,
}
"""Byte offsets of fixed-position fields before SCAN v2 variable dim6 records."""


def _read_u16(buffer: bytes, offset: int) -> int:
    """Read a little-endian `uint16` from `buffer` at `offset`.

    Parameters
    ----------
    buffer : bytes
        Byte buffer to read from.
    offset : int
        Byte offset of the value.

    Returns
    -------
    int
        The decoded value.
    """
    return int.from_bytes(buffer[offset : offset + 2], "little")


def _read_u32(buffer: bytes, offset: int) -> int:
    """Read a little-endian `uint32` from `buffer` at `offset`.

    Parameters
    ----------
    buffer : bytes
        Byte buffer to read from.
    offset : int
        Byte offset of the value.

    Returns
    -------
    int
        The decoded value.
    """
    return int.from_bytes(buffer[offset : offset + 4], "little")


def _read_u64(buffer: bytes, offset: int) -> int:
    """Read a little-endian `uint64` from `buffer` at `offset`.

    Parameters
    ----------
    buffer : bytes
        Byte buffer to read from.
    offset : int
        Byte offset of the value.

    Returns
    -------
    int
        The decoded value.
    """
    return int.from_bytes(buffer[offset : offset + 8], "little")


def _read_f64(buffer: bytes, offset: int) -> float:
    """Read a little-endian `float64` from `buffer` at `offset`.

    Parameters
    ----------
    buffer : bytes
        Byte buffer to read from.
    offset : int
        Byte offset of the value.

    Returns
    -------
    float
        The decoded value.
    """
    return float(struct.unpack_from("<d", buffer, offset)[0])


def _read_scan_v2_struct(buffer: bytes, offset: int, fmt: str) -> tuple[Any, int]:
    """Read a little-endian SCAN v2 field and return the advanced offset.

    Parameters
    ----------
    buffer : bytes
        Byte buffer to read from.
    offset : int
        Byte offset of the value.
    fmt : str
        `struct` format string, including byte order.

    Returns
    -------
    value : object
        Decoded scalar value.
    offset : int
        Byte offset immediately after the decoded value.
    """
    size = struct.calcsize(fmt)
    return struct.unpack_from(fmt, buffer, offset)[0], offset + size


def _read_scan_v2_binary_string(
    buffer: bytes, offset: int, length_fmt: str = "<L"
) -> tuple[str, int]:
    """Read a SCAN v2 length-prefixed UTF-8 string.

    Parameters
    ----------
    buffer : bytes
        Byte buffer to read from.
    offset : int
        Byte offset of the length prefix.
    length_fmt : str, default: "<L"
        `struct` format of the length prefix.

    Returns
    -------
    text : str
        Decoded string value.
    offset : int
        Byte offset immediately after the string bytes.

    Raises
    ------
    ValueError
        If the string length points beyond the buffer.
    """
    length, offset = _read_scan_v2_struct(buffer, offset, length_fmt)
    end = offset + int(length)
    if end > len(buffer):
        raise ValueError("SCAN v2 string field is truncated.")
    return buffer[offset:end].decode("utf-8", errors="replace"), end


WORLD_TO_PROBE_PERMUTATION: npt.NDArray[np.float64] = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)
"""Permutation matrix that maps ConfUSIus world to probe world.

ConfUSIus input (z_conf, y_conf, x_conf, 1) is mapped to the probe world (x_probe,
y_probe, z_probe, 1):

  x_probe =  x_conf      (lateral, same direction)
  y_probe =  z_conf      (elevation, same direction)
  z_probe = -y_conf      (axial depth, sign flip: y_conf = -z_probe > 0)

Its transpose maps probe world (x_probe, y_probe, z_probe, 1) back to ConfUSIus
world (z_conf, y_conf, x_conf, 1):

  z_conf =  y_probe      (elevation)
  y_conf = -z_probe      (depth, sign flip)
  x_conf =  x_probe      (lateral)

"""


def _read_scan_str(h5: h5py.File, path: str) -> str:
    """Read a scalar string dataset from a SCAN HDF5 file.

    SCAN files store string fields as MATLAB-written object-dtype datasets with shape
    `(1, 1)`. This helper flattens the dataset and decodes bytes if necessary.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        HDF5 dataset path.

    Returns
    -------
    str
        Decoded string value.
    """
    val = h5[path][()].flat[0]
    if isinstance(val, bytes):
        val = val.decode()
    return str(val)


def _read_scan_scalar(h5: h5py.File, path: str) -> float:
    """Read a scalar float dataset from a SCAN HDF5 file.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        HDF5 dataset path.

    Returns
    -------
    float
        Scalar float value.
    """
    return float(h5[path][()].flat[0])


def _build_probe_to_lab(
    probe_to_lab: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert `probeToLab` to a ConfUSIus `probe_to_lab` affine in mm.

    `probeToLab` maps probe coordinates `(x_probe, y_probe, z_probe, 1)` to Iconeus lab
    space `(x_lab, y_lab, z_lab, 1)` in meters. The Iconeus lab frame is a fixed scanner
    frame; `probeToLab` carries any translation and rotation of the probe within it.

    `probe_to_lab` maps ConfUSIus-ordered probe coordinates `(z_probe, y_probe, x_probe,
    1)` (elevation, depth, lateral) to **ConfUSIus-ordered** lab coordinates `(z_lab,
    y_lab, x_lab)` in millimeters, using the permutation `P`:

    ```python
    probe_to_lab = WORLD_TO_PROBE_PERMUTATION^T @ probeToLab @ WORLD_TO_PROBE_PERMUTATION
    ```

    Parameters
    ----------
    probe_to_lab : (4, 4) or (npose, 4, 4) numpy.ndarray
        `probeToLab` affine(s) from a SCAN file (units meters).

    Returns
    -------
    numpy.ndarray
        `probe_to_lab` affine(s) in millimeters. Shape matches input: `(4, 4)` for
        `2Dscan` or `(npose, 4, 4)` for `3Dscan`/`4Dscan`.
    """
    probe_to_lab = (
        WORLD_TO_PROBE_PERMUTATION.T @ probe_to_lab @ WORLD_TO_PROBE_PERMUTATION
    )
    probe_to_lab[..., :3, 3] *= 1e3
    return probe_to_lab


def _build_voxel_to_probe(
    voxels_to_probe: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert `voxelsToProbe` into a zero-based ConfUSIus `voxel_to_probe` affine.

    Parameters
    ----------
    voxels_to_probe : (4, 4) numpy.ndarray
        SCAN `voxelsToProbe` affine mapping one-based probe voxel coordinates to probe
        coordinates in metres.

    Returns
    -------
    (4, 4) numpy.ndarray
        Affine mapping zero-based ConfUSIus voxel coordinates `(k, j, i)` to
        ConfUSIus-ordered probe coordinates `(z, y, x)` in millimetres.
    """
    conf_voxel_to_probe_voxel = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    metres_to_mm = np.diag([1e3, 1e3, 1e3, 1.0])
    return (
        metres_to_mm
        @ WORLD_TO_PROBE_PERMUTATION.T
        @ np.asarray(voxels_to_probe, dtype=np.float64)
        @ conf_voxel_to_probe_voxel
    )


def _swap_depth_elevation_axes(arr):
    """Swap the last two spatial axes to reach ConfUSIus `(..., k, j, i)` order.

    Scan payloads store their last three axes as (depth, elevation, lateral);
    ConfUSIus order is (elevation=`k`, depth=`j`, lateral=`i`). Only the depth/
    elevation pair needs swapping, regardless of how many leading (`time`/`pose`)
    axes precede them.

    Parameters
    ----------
    arr : dask.array.Array
        Array whose last 3 axes are `(depth, elevation, lateral)`.

    Returns
    -------
    dask.array.Array
        `arr` with its last two axes swapped.
    """
    axes = list(range(arr.ndim))
    axes[-3], axes[-2] = axes[-2], axes[-3]
    return da.transpose(arr, axes)


def _fold_block_repeat_into_time(raw_lazy, npose: int, nblock_repeat: int):
    """Fold a v1 `nblock_repeat` axis into `time`.

    HDF5 SCAN v1 stores repeated sub-blocks along axis 2 rather than as part of
    `time` directly; folding them in gives a single, longer `time` axis.

    Parameters
    ----------
    raw_lazy : dask.array.Array
        Raw array with dims `(n_time, npose, nblock_repeat, z, y, x)`.
    npose : int
        Number of robot positions.
    nblock_repeat : int
        Number of repeated sub-blocks per time point.

    Returns
    -------
    dask.array.Array
        Array with dims `(n_time * nblock_repeat, npose, z, y, x)`.
    """
    if nblock_repeat == 1:
        return da.squeeze(raw_lazy, axis=2)
    n_time_total = raw_lazy.shape[0] * nblock_repeat
    transposed = da.transpose(raw_lazy, [0, 2, 1, 3, 4, 5])
    return transposed.reshape(n_time_total, npose, *raw_lazy.shape[3:])


def _scan_time_attrs(duration: float) -> dict[str, Any]:
    """Return the standard `time` coordinate attrs for a scan volume.

    Parameters
    ----------
    duration : float
        Volume acquisition duration in seconds.

    Returns
    -------
    dict[str, Any]
        `units`/`volume_acquisition_reference`/`volume_acquisition_duration` attrs.
    """
    return {
        "units": "s",
        "volume_acquisition_reference": "end",
        "volume_acquisition_duration": duration,
    }


def load_bps(bps_path: str | Path) -> npt.NDArray[np.float64]:
    """Load a BPS file and return an affine from Iconeus' brain space to ConfUSIus lab space.

    BPS files are HDF5 sidecars produced by Iconeus' brain positioning system. They
    store a `BrainToLab` affine that maps Iconeus brain coordinates `(x_brain, y_brain,
    z_brain, 1)` to Iconeus lab coordinates `(x_lab, y_lab, z_lab, 1)` in meters.
    The Iconeus lab frame is a fixed scanner frame; `probeToLab` carries any rotation
    of the probe within it.

    To compose this affine with the rest of the ConfUSIus pipeline we re-express the lab
    side as **ConfUSIus-ordered** lab space `(z_lab, y_lab, x_lab)` in millimeters. The
    brain side is left in its original axis order (the brain coordinate units are not
    declared by the BPS format and are therefore not converted).

    The change of basis from ConfUSIus-ordered millimeter lab coordinates to
    Iconeus-ordered meter lab coordinates is

    ```
    confusius_lab_to_iconeus_lab = mm_to_m @ WORLD_TO_PROBE_PERMUTATION
    ```

    `WORLD_TO_PROBE_PERMUTATION` permutes the axes from ConfUSIus order `(z, y, x)` to
    probe / Iconeus-lab order `(x, y, z)`, and `mm_to_m = diag(1e-3, 1e-3, 1e-3, 1)`
    rescales the translation column. The returned affine is then

    ```
    brain_to_lab = inv(confusius_lab_to_iconeus_lab) @ BrainToLab
    ```

    Parameters
    ----------
    bps_path : str or pathlib.Path
        Path to the BPS file (`.bps`).

    Returns
    -------
    (4, 4) numpy.ndarray
        Affine mapping Iconeus brain coordinates to ConfUSIus-ordered Iconeus lab
        coordinates `(z_lab, y_lab, x_lab, 1)` in millimeters.
    """
    bps_path = check_path(bps_path, label="bps_path", type="file")

    with h5py.File(bps_path, "r") as f:
        brain_to_lab = f["BrainToLab"][:]

    mm_to_m = np.diag([1e-3, 1e-3, 1e-3, 1.0])
    confusius_lab_to_iconeus_lab = mm_to_m @ WORLD_TO_PROBE_PERMUTATION

    brain_to_lab = np.linalg.inv(confusius_lab_to_iconeus_lab) @ brain_to_lab
    return brain_to_lab


def _add_world_to_brain(affines: dict[str, Any], bps_path: str | Path) -> None:
    """Compose a `world_to_brain` affine from a BPS sidecar and store it in `affines`.

    ConfUSIus world coordinates for SCAN data are already lab space, so
    `world_to_brain = inv(load_bps(bps_path))`.

    Parameters
    ----------
    affines : dict
        The DataArray's `affines` attribute; mutated in place to add `world_to_brain`.
    bps_path : str or pathlib.Path
        Path to the BPS file (`.bps`).

    Returns
    -------
    None
        `affines` is updated in place.
    """
    brain_to_lab = load_bps(bps_path)
    affines["world_to_brain"] = np.linalg.inv(brain_to_lab)


def load_scan(
    path: str | Path,
    bps_path: str | Path | None = None,
    chunks: int | tuple[int, ...] | str | None = "auto",
) -> xr.DataArray:
    """Load an Iconeus SCAN file as a lazy VoxelData array.

    SCAN files (`.scan`) come in two on-disk formats, both handled here:

    - **v1**: an HDF5 container produced by IcoScan/NeuroScan, holding power Doppler
      data and spatial/temporal metadata for 2D, 3D, or 3D+t fUSI volumes. The returned
      DataArray wraps an open `h5py` handle via a Dask array; keep it in scope (or call
      `.compute()`) before the handle is garbage-collected.
    - **v2**: a flat binary file (variable-length header + little-endian `float64`
      payload). The returned DataArray wraps a NumPy memmap via a Dask array and, when
      a `bps_path` is given, adds `world_to_brain`.

    `load_scan` sniffs the format automatically and dispatches accordingly.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the SCAN file (`.scan`).
    bps_path : str or pathlib.Path, optional
        Path to the corresponding BPS file (`.bps`). If provided, a `world_to_brain`
        affine is added to `da.attrs["affines"]`.
    chunks : int or tuple[int, ...] or str or None, default: "auto"
        Dask chunk specification passed to `dask.array.from_array`. Accepted forms:

        - A blocksize like `1000`.
        - A blockshape like `(1000, 1000)`.
        - Explicit sizes of all blocks like `((1000, 1000, 500), (400, 400))`.
        - A size in bytes like `"100 MiB"`.
        - `"auto"` to let Dask choose based on heuristics.
        - `-1` or `None` for the full dimension size (no chunking).

    Returns
    -------
    xarray.DataArray
        Lazy VoxelData array with dimensions and coordinates:

        - v1 `2Dscan` → `(time, k, j, i)`.
        - v1 `3Dscan` → `(pose, k, j, i)`.
        - v1 `4Dscan`/`4DscanCustom` → `(time, pose, k, j, i)`.
        - v2 single-pose → `(time, k, j, i)`.
        - v2 multi-pose → `(time, pose, k, j, i)`.
        - v2 dim6 payloads prepend a `dim6` dimension.

        World coordinates `z`, `y`, `x` are in millimeters. The `time` coordinate is in
        seconds. For v1/v2 multi-pose time series, `time` is pose-dependent
        (`(time, pose)`-shaped), holding each pose's own acquisition timestamps directly.

    Raises
    ------
    ValueError
        If `path` does not exist or is not a file, if the file is neither an
        HDF5-based SCAN (v1) nor a binary SCAN v2 file, if a v2 file's
        `probe_to_lab` affine cannot be built, or if a v1 `acquisitionMode` is not one
        of `"2Dscan"`, `"3Dscan"`, `"4Dscan"`, or `"4DscanCustom"`.

    Notes
    -----
    **v2.** The binary layout follows PyIconeus: header fields, measured times, one
    6DOF probe pose per `pose`, acquisition/provenance metadata, then a Fortran-ordered
    `(size_x, size_y, size_z, time, pose, dim6)` payload. The loader exposes this as
    VoxelData order `(..., time, pose, k, j, i)` and keeps dim6 as an extra leading
    dimension when present.

    Acquisition settings that correspond to fUSI-BIDS fields are also surfaced as
    attributes, in native header units: `probe_model`, `probe_center_frequency` (MHz),
    `probe_pitch` (mm), `imaging_depth` (mm start/end), `transmit_frequency` (MHz),
    `pulse_repetition_frequency` (Hz), `plane_wave_angles` (deg), `svd_low_cutoff`, and
    `power_doppler_integration_duration` (s).

    ConfUSIus world coordinates `(z, y, x)` for SCAN data are **ConfUSIus-ordered
    Iconeus lab coordinates** (mm): a fixed scanner frame shared by every pose, used
    as the canonical world frame because it is the one frame that is physically
    meaningful across poses. For multi-pose files, `da`'s voxel-to-world geometry is
    itself pose-dependent (a `(npose, 4, 4)` affine stack, one per
    `da.coords["pose"]` label — see
    [VoxelToWorldIndex][confusius._utils.geometry.VoxelToWorldIndex]); world
    selection therefore requires reducing `pose` to a scalar first, e.g.
    `da.isel(pose=0).sel(z=..., y=..., x=...)` or `da.isel(pose=0)`.

    If `bps_path` is provided, a `world_to_brain` affine is stored in
    `da.attrs["affines"]["world_to_brain"]` that maps ConfUSIus world coordinates
    (already lab space, as above) to Iconeus' brain coordinates. Apply as
    `da.attrs["affines"]["world_to_brain"] @ np.array([z, y, x, 1.0])`.

    Provenance attributes are stored in `da.attrs`: BIDS-compatible fields
    (`device_serial_number`, `software_version`) and Iconeus-specific fields
    (`iconeus_scan_mode`, `iconeus_subject`, `iconeus_session`, `iconeus_scan`,
    `iconeus_project`, `iconeus_date`).
    """
    path = check_path(path, type="file")

    if h5py.is_hdf5(path):
        return _load_scan_v1(path, bps_path, chunks)

    with path.open("rb") as f:
        magic = f.read(len(SCAN_V2_MAGIC))

    if magic == SCAN_V2_MAGIC:
        # Re-raise parse failures with a pointer to the issue tracker and the original
        # error appended.
        try:
            data_array = _load_scan_v2(path, chunks)
        except Exception as error:
            raise ValueError(
                "This Iconeus SCAN v2 file could not be parsed. Please open an issue at "
                "https://github.com/confusius-tools/confusius/issues (attaching an "
                f"example file if possible).\n\nOriginal error: "
                f"{type(error).__name__}: {error}"
            ) from error

        if bps_path is not None:
            _add_world_to_brain(data_array.attrs["affines"], bps_path)
        return data_array

    raise ValueError(
        f"{path.name!r} is not a SCAN file recognized by ConfUSIus. Expected an "
        f"HDF5-based SCAN (v1) or a binary SCAN v2 file starting with the magic bytes "
        f"{SCAN_V2_MAGIC!r}."
    )


def _load_scan_v1(
    path: Path,
    bps_path: str | Path | None,
    chunks: int | tuple[int, ...] | str | None,
) -> xr.DataArray:
    """Load an HDF5-based Iconeus SCAN (v1) file as a lazy VoxelData array.

    Parameters
    ----------
    path : pathlib.Path
        Path to the v1 SCAN file, already validated as HDF5.
    bps_path : str or pathlib.Path, optional
        Path to the corresponding BPS file (`.bps`). If provided, a `world_to_brain`
        affine is added to `da.attrs["affines"]`.
    chunks : int or tuple[int, ...] or str or None
        Dask chunk specification passed to `dask.array.from_array`.

    Returns
    -------
    xarray.DataArray
        Lazy VoxelData array whose dims depend on the file's
        `acquisitionMode`. See
        `load_scan` for the full contract.

    Raises
    ------
    ValueError
        If the `acquisitionMode` stored in the file is not one of `"2Dscan"`,
        `"3Dscan"`, `"4Dscan"`, or `"4DscanCustom"`.
    """
    h5 = h5py.File(path, "r")

    try:
        mode = _read_scan_str(h5, "/acqMetaData/acquisitionMode")

        npose = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/npose"))
        nblock_repeat = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/nblockRepeat"))

        voxels_to_probe: npt.NDArray[np.float64] = np.array(
            h5["/acqMetaData/voxelsToProbe"][()], dtype=np.float64
        )
        probe_to_lab: npt.NDArray[np.float64] = np.array(
            h5["/acqMetaData/probeToLab"][()], dtype=np.float64
        )

        voxel_to_probe = _build_voxel_to_probe(voxels_to_probe)
        probe_to_lab = _build_probe_to_lab(probe_to_lab)
        # Lab is the canonical VoxelData world frame for SCAN data, so composing these
        # physical transforms produces the VoxelData geometry directly.
        voxel_to_world = probe_to_lab @ voxel_to_probe

        attrs: dict[str, Any] = {
            "affines": {},
            "device_serial_number": _read_scan_str(h5, "/scanMetaData/Machine_SN"),
            "software_version": _read_scan_str(h5, "/scanMetaData/Neuroscan_version"),
            "iconeus_scan_mode": mode,
            "iconeus_subject": _read_scan_str(h5, "/scanMetaData/Subject_tag"),
            "iconeus_session": _read_scan_str(h5, "/scanMetaData/Session_tag"),
            "iconeus_scan": _read_scan_str(h5, "/scanMetaData/Scan_tag"),
            "iconeus_project": _read_scan_str(h5, "/scanMetaData/Project_tag"),
            "iconeus_date": _read_scan_str(h5, "/scanMetaData/Date"),
        }

        raw_lazy = da.from_array(h5["/Data"], chunks=chunks, asarray=False)

        if mode == "2Dscan":
            data_array = _load_2dscan(h5, raw_lazy, attrs, voxel_to_world)
        elif mode == "3Dscan":
            data_array = _load_3dscan(raw_lazy, attrs, npose, voxel_to_world)
        elif mode in {"4Dscan", "4DscanCustom"}:
            data_array = _load_4dscan(
                h5,
                raw_lazy,
                attrs,
                npose,
                nblock_repeat,
                voxel_to_world,
            )
        else:
            raise ValueError(
                f"Unknown acquisitionMode: {mode!r}. Expected one of '2Dscan',"
                " '3Dscan', '4Dscan', '4DscanCustom'."
            )

        data_array.name = attrs["iconeus_scan"] or path.stem
        if bps_path is not None:
            _add_world_to_brain(data_array.attrs["affines"], bps_path)
    except Exception:
        h5.close()
        raise

    return data_array


def _load_2dscan(
    h5: h5py.File,
    raw_lazy: da.Array,
    attrs: dict[str, Any],
    voxel_to_world: npt.NDArray[np.float64],
) -> xr.DataArray:
    """Build a VoxelData array for `2Dscan` mode."""
    data_lazy = _swap_depth_elevation_axes(raw_lazy)
    time: npt.NDArray[np.float64] = np.array(
        h5["/acqMetaData/time"][()], dtype=np.float64
    ).squeeze()
    time_attrs = _scan_time_attrs(float(time.min()))
    return create_voxeldata(
        data_lazy,
        dims=("time", "k", "j", "i"),
        time=xr.DataArray(time, dims=["time"], attrs=time_attrs),
        voxel_to_world=voxel_to_world,
        attrs=attrs,
    )


def _load_3dscan(
    raw_lazy: da.Array,
    attrs: dict[str, Any],
    npose: int,
    voxel_to_world: npt.NDArray[np.float64],
) -> xr.DataArray:
    """Build a VoxelData array for `3Dscan` mode."""
    sq = da.squeeze(raw_lazy, axis=1)
    data_lazy = _swap_depth_elevation_axes(sq)
    return create_voxeldata(
        data_lazy,
        dims=("pose", "k", "j", "i"),
        pose=np.arange(npose),
        voxel_to_world=voxel_to_world,
        attrs=attrs,
    )


def _load_4dscan(
    h5: h5py.File,
    raw_lazy: da.Array,
    attrs: dict[str, Any],
    npose: int,
    nblock_repeat: int,
    voxel_to_world: npt.NDArray[np.float64],
) -> xr.DataArray:
    """Build a VoxelData array for `4Dscan` mode."""
    n_time = raw_lazy.shape[0] * nblock_repeat
    sq = _fold_block_repeat_into_time(raw_lazy, npose, nblock_repeat)
    data_lazy = _swap_depth_elevation_axes(sq)
    time_raw: npt.NDArray[np.float64] = (
        np.array(h5["/acqMetaData/time"][()], dtype=np.float64)
        .squeeze()
        .reshape(n_time, npose)
    )
    time_attrs = _scan_time_attrs(float(time_raw.min()))
    return create_voxeldata(
        data_lazy,
        dims=("time", "pose", "k", "j", "i"),
        time=xr.DataArray(time_raw, dims=["time", "pose"], attrs=time_attrs),
        pose=np.arange(npose),
        voxel_to_world=voxel_to_world,
        attrs=attrs,
    )


def _read_scan_v2_header(header: bytes) -> dict[str, Any]:
    """Parse a SCAN v2 header using the binary layout published by PyIconeus.

    Portions of this function are derived from PyIconeus, which is licensed under the
    BSD-3-Clause License. See `NOTICE` file for details.

    Parameters
    ----------
    header : bytes
        The full header bytes (`total_header_bytes` long).

    Returns
    -------
    dict
        Parsed SCAN v2 metadata needed to expose the payload as VoxelData.

    Raises
    ------
    ValueError
        If the header is truncated, reports non-positive dimensions, or contains an
        unsupported dim6 intent.
    """
    if len(header) < 132:
        raise ValueError("SCAN v2 header is truncated before the dimension block.")

    fields: dict[str, Any] = {
        "total_header_bytes": _read_u64(header, _SCAN_V2_OFFSETS["total_header_bytes"]),
        "payload_bytes": _read_u64(header, _SCAN_V2_OFFSETS["payload_bytes"]),
        "size_x": _read_u64(header, _SCAN_V2_OFFSETS["size_x"]),
        "size_y": _read_u64(header, _SCAN_V2_OFFSETS["size_y"]),
        "size_z": _read_u64(header, _SCAN_V2_OFFSETS["size_z"]),
        "n_time": _read_u64(header, _SCAN_V2_OFFSETS["n_time"]),
        "npose": _read_u64(header, _SCAN_V2_OFFSETS["npose"]),
    }
    for key in ("size_x", "size_y", "size_z", "n_time", "npose"):
        if fields[key] < 1:
            raise ValueError(
                f"SCAN v2 header reports a non-positive {key}={fields[key]}; the file "
                "may be corrupt or use an unsupported layout."
            )

    offset = _SCAN_V2_OFFSETS["dim6_count"]
    dim6_count, offset = _read_scan_v2_struct(header, offset, "<Q")
    if dim6_count < 1:
        raise ValueError("SCAN v2 header reports a non-positive dim6 count.")
    fields["dim6_count"] = int(dim6_count)

    dim6_intents: list[int] = []
    for _ in range(int(dim6_count)):
        intent, offset = _read_scan_v2_struct(header, offset, "<L")
        dim6_intents.append(int(intent))
    fields["dim6_intents"] = dim6_intents

    dim6_attrs: dict[str, Any] = {}
    for intent in dim6_intents:
        if intent in {1, 3}:  # EnhancedDoppler or BrainMaskedDoppler.
            offset += 20
        elif intent == 0:  # ClutterFiltering.
            filter_type, offset = _read_scan_v2_struct(header, offset, "<L")
            window, offset = _read_scan_v2_struct(header, offset, "<d")
            cutoff_fmt = "<L" if int(filter_type) in {0, 1} else "<f"
            low, offset = _read_scan_v2_struct(header, offset, cutoff_fmt)
            high, offset = _read_scan_v2_struct(header, offset, cutoff_fmt)
            dim6_attrs.update(
                {
                    "clutter_filter_type": int(filter_type),
                    "clutter_filter_window_duration": float(window),
                    "svd_low_cutoff": low,
                    "svd_high_cutoff": high,
                }
            )
        elif intent == 2:  # VelocityBandFiltering.
            vmin, offset = _read_scan_v2_struct(header, offset, "<f")
            vmax, offset = _read_scan_v2_struct(header, offset, "<f")
            offset += 12
            dim6_attrs.update(
                {"velocity_min": float(vmin), "velocity_max": float(vmax)}
            )
        else:
            raise ValueError(f"Unsupported SCAN v2 dim6 intent: {intent}.")
    fields["dim6_attrs"] = dim6_attrs

    vox = np.frombuffer(header, dtype="<f8", count=6, offset=offset).copy()
    offset += 48
    fields.update(
        {
            "x_voxel_m": float(vox[0]),
            "y_voxel_m": float(vox[1]),
            "z_voxel_m": float(vox[2]),
            "dt": float(vox[3]),
            "dr": float(vox[4]),
            "dtheta": float(vox[5]),
        }
    )

    time_count = fields["size_y"] * fields["n_time"] * fields["npose"]
    time_end = offset + 12 * time_count
    pose_end = time_end + 48 * fields["npose"]
    if pose_end > len(header):
        raise ValueError(
            "SCAN v2 header is truncated or reports an implausible timing/pose block "
            f"(time_count={time_count}, header={len(header)} bytes)."
        )
    fields["measured_times"] = np.frombuffer(
        header, dtype="<f8", count=time_count, offset=offset
    ).copy()
    offset += 8 * time_count
    fields["theoretical_time_indices"] = np.frombuffer(
        header, dtype="<u4", count=time_count, offset=offset
    ).copy()
    offset += 4 * time_count
    fields["probe_to_lab_translations"] = (
        np.frombuffer(header, dtype="<f8", count=3 * fields["npose"], offset=offset)
        .copy()
        .reshape(fields["npose"], 3)
    )
    offset += 24 * fields["npose"]
    fields["probe_to_lab_rotations"] = (
        np.frombuffer(header, dtype="<f8", count=3 * fields["npose"], offset=offset)
        .copy()
        .reshape(fields["npose"], 3)
    )
    offset += 24 * fields["npose"]

    fields.update(_read_scan_v2_tail(header, offset))
    return fields


def _read_scan_v2_tail(header: bytes, offset: int) -> dict[str, Any]:
    """Parse SCAN v2 acquisition and provenance fields after the pose block.

    Portions of this function are derived from PyIconeus, which is licensed under the
    BSD-3-Clause License. See `NOTICE` file for details.

    Parameters
    ----------
    header : bytes
        The full header bytes.
    offset : int
        Byte offset immediately after the pose rotation block.

    Returns
    -------
    dict
        Parsed metadata fields.
    """
    result: dict[str, Any] = {}
    offset += 4
    acquisition_mode, offset = _read_scan_v2_struct(header, offset, "<L")
    result["acquisition_mode_code"] = int(acquisition_mode)
    offset += 4

    probe_type, offset = _read_scan_v2_struct(header, offset, "<L")
    center_frequency, offset = _read_scan_v2_struct(header, offset, "<d")
    pitch, offset = _read_scan_v2_struct(header, offset, "<d")
    elevation_aperture, offset = _read_scan_v2_struct(header, offset, "<d")
    offset += 8
    radius, offset = _read_scan_v2_struct(header, offset, "<d")
    n_elements, offset = _read_scan_v2_struct(header, offset, "<H")
    probe_model, offset = _read_scan_v2_binary_string(header, offset, "<H")
    probe_name, offset = _read_scan_v2_binary_string(header, offset, "<H")
    depth_near, offset = _read_scan_v2_struct(header, offset, "<d")
    depth_far, offset = _read_scan_v2_struct(header, offset, "<d")
    transmit_frequency, offset = _read_scan_v2_struct(header, offset, "<d")
    prf, offset = _read_scan_v2_struct(header, offset, "<d")
    sampling_frequency, offset = _read_scan_v2_struct(header, offset, "<d")
    offset += 8
    n_angles, offset = _read_scan_v2_struct(header, offset, "<L")
    if n_angles > (len(header) - offset) // 8:
        raise ValueError("SCAN v2 plane-wave angle block is truncated.")
    plane_wave_angles = np.frombuffer(
        header, dtype="<f8", count=int(n_angles), offset=offset
    ).copy()
    offset += 8 * int(n_angles)

    skip_count, offset = _read_scan_v2_struct(header, offset, "<L")
    offset += int(skip_count) * 24 + 8
    transmit_voltage, offset = _read_scan_v2_struct(header, offset, "<d")
    offset += 4
    delay_after_trigger, offset = _read_scan_v2_struct(header, offset, "<d")
    skip_count, offset = _read_scan_v2_struct(header, offset, "<L")
    offset += int(skip_count) * 8
    is_multiplane, offset = _read_scan_v2_struct(header, offset, "<?")
    offset += 1
    integration_window, offset = _read_scan_v2_struct(header, offset, "<d")

    sequence_name, offset = _read_scan_v2_binary_string(header, offset)
    project, offset = _read_scan_v2_binary_string(header, offset)
    project_description, offset = _read_scan_v2_binary_string(header, offset)
    subject, offset = _read_scan_v2_binary_string(header, offset)
    session, offset = _read_scan_v2_binary_string(header, offset)
    species, offset = _read_scan_v2_binary_string(header, offset)
    gender, offset = _read_scan_v2_struct(header, offset, "<L")
    transfer_ts, offset = _read_scan_v2_struct(header, offset, "<q")
    age, offset = _read_scan_v2_struct(header, offset, "<Q")
    subject_description, offset = _read_scan_v2_binary_string(header, offset)
    weight_unit, offset = _read_scan_v2_struct(header, offset, "<L")
    weight, offset = _read_scan_v2_struct(header, offset, "<f")
    treatment, offset = _read_scan_v2_binary_string(header, offset)
    scan, offset = _read_scan_v2_binary_string(header, offset)
    study_type, offset = _read_scan_v2_binary_string(header, offset)
    task_name, offset = _read_scan_v2_binary_string(header, offset)
    task_description, offset = _read_scan_v2_binary_string(header, offset)
    username, offset = _read_scan_v2_binary_string(header, offset)
    for _ in range(2):
        skip_count, offset = _read_scan_v2_struct(header, offset, "<L")
        offset += int(skip_count)
    acquisition_ts, offset = _read_scan_v2_struct(header, offset, "<q")
    scan_type, offset = _read_scan_v2_struct(header, offset, "<L")
    n_toggle, offset = _read_scan_v2_struct(header, offset, "<L")
    if n_toggle > (len(header) - offset) // 4:
        raise ValueError("SCAN v2 stimulation-toggle block is truncated.")
    stimulation_toggle_times = np.frombuffer(
        header, dtype="<f4", count=int(n_toggle), offset=offset
    ).copy()
    offset += 4 * int(n_toggle)
    major, offset = _read_scan_v2_struct(header, offset, "<L")
    minor, offset = _read_scan_v2_struct(header, offset, "<L")
    patch, _ = _read_scan_v2_struct(header, offset, "<L")

    result.update(
        {
            "probe_type": int(probe_type),
            "probe_model": probe_name,
            "probe_model_number": probe_model,
            "probe_center_frequency": float(center_frequency),
            "probe_pitch": float(pitch),
            "probe_elevation_aperture": float(elevation_aperture),
            "probe_radius_of_curvature": float(radius),
            "probe_number_of_elements": int(n_elements),
            "depth_start_mm": float(depth_near),
            "imaging_depth": (float(depth_near), float(depth_far)),
            "transmit_frequency": float(transmit_frequency),
            "pulse_repetition_frequency": float(prf),
            "ultrafast_sampling_frequency": float(sampling_frequency),
            "plane_wave_angles": plane_wave_angles.tolist(),
            "transmit_voltage": float(transmit_voltage),
            "delay_after_trigger": float(delay_after_trigger),
            "is_multiplane": bool(is_multiplane),
            "power_doppler_integration_duration": float(integration_window),
            "iconeus_sequence": sequence_name,
            "iconeus_project": project,
            "iconeus_project_description": project_description,
            "iconeus_subject": subject,
            "iconeus_session": session,
            "iconeus_species": species,
            "iconeus_gender": int(gender),
            "iconeus_transfer_datetime": datetime.fromtimestamp(
                int(transfer_ts), UTC
            ).isoformat(),
            "iconeus_age_at_transfer": int(age),
            "iconeus_subject_description": subject_description,
            "iconeus_weight_unit": int(weight_unit),
            "iconeus_weight": float(weight),
            "iconeus_treatment": treatment,
            "iconeus_scan": scan,
            "iconeus_study_type": study_type,
            "iconeus_task_name": task_name,
            "iconeus_task_description": task_description,
            "iconeus_experimenter": username,
            "iconeus_datetime": datetime.fromtimestamp(
                int(acquisition_ts), UTC
            ).isoformat(),
            "iconeus_scan_type": int(scan_type),
            "stimulation_toggle_times": stimulation_toggle_times.tolist(),
            "software_version": f"{int(major)}.{int(minor)}.{int(patch)}",
        }
    )
    return result


def _load_scan_v2(
    path: Path,
    chunks: int | tuple[int, ...] | str | None,
) -> xr.DataArray:
    """Load a binary Iconeus SCAN v2 file as a lazy VoxelData array.

    The v2 format is a flat binary file: a variable-length header followed by a
    little-endian `float64` power-Doppler payload. The payload is wrapped in a NumPy
    memmap (never fully read here) and exposed lazily through Dask.

    The header stores Iconeus-ordered dimensions `(size_x=lateral, size_y=elevation,
    size_z=depth, n_time, npose, dim6)`. The payload is Fortran-ordered with the same
    axis order. Output dims are VoxelData-ordered: optional `dim6`, optional `time`,
    optional `pose`, then `(k, j, i)`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the v2 SCAN file, already validated to start with `SCAN_V2_MAGIC`.
    chunks : int or tuple[int, ...] or str or None
        Dask chunk specification passed to `dask.array.from_array`.

    Returns
    -------
    xarray.DataArray
        Lazy VoxelData array with dims `(time, k, j, i)` or
        `(time, pose, k, j, i)` and
        world coordinates derived from a `VoxelToWorldIndex` affine (depth origin
        from the header when found; lateral/elevation centered on zero — see
        `load_scan` Notes).

    Raises
    ------
    ValueError
        If the header is truncated, reports implausible dimensions, or if the reported
        payload size does not match the product of the dimensions.
    """
    offset = _SCAN_V2_OFFSETS["total_header_bytes"]
    with path.open("rb") as f:
        total_header_bytes = _read_u64(f.read(offset + 8), offset)
        f.seek(0)
        header = f.read(total_header_bytes)

    meta = _read_scan_v2_header(header)
    size_x = meta["size_x"]
    size_y = meta["size_y"]
    size_z = meta["size_z"]
    n_time = meta["n_time"]
    npose = meta["npose"]
    dim6_count = meta["dim6_count"]
    n_elements = size_x * size_y * size_z * n_time * npose * dim6_count
    payload_bytes = path.stat().st_size - total_header_bytes
    if meta["payload_bytes"] and meta["payload_bytes"] != payload_bytes:
        payload_bytes = meta["payload_bytes"]
    if payload_bytes != n_elements * 8:
        raise ValueError(
            f"SCAN v2 payload size ({payload_bytes} bytes) does not match the product "
            f"of the header dimensions ({n_elements} float64 = {n_elements * 8} "
            "bytes). The file may be corrupt or use an unsupported layout."
        )

    # PyIconeus documents the binary payload as Fortran-ordered
    # `(size_x, size_y, size_z, n_time, npose, dim6)`.
    memmap = np.memmap(
        path,
        dtype="<f8",
        mode="r",
        offset=total_header_bytes,
        shape=(size_x, size_y, size_z, n_time, npose, dim6_count),
        order="F",
    )
    raw_lazy = da.from_array(memmap, chunks=chunks, asarray=False)
    data_lazy = da.transpose(raw_lazy, [5, 3, 4, 1, 2, 0])

    scan_mode = _scan_v2_mode(meta)
    include_time = n_time > 1 or scan_mode == "2Dscan"
    include_pose = npose > 1
    dims_list = ["dim6", "time", "pose", "k", "j", "i"]
    squeeze_axes = []
    if dim6_count == 1:
        squeeze_axes.append(0)
        dims_list.remove("dim6")
    if not include_time:
        squeeze_axes.append(1)
        dims_list.remove("time")
    if not include_pose:
        squeeze_axes.append(2)
        dims_list.remove("pose")
    if squeeze_axes:
        data_lazy = da.squeeze(data_lazy, axis=tuple(squeeze_axes))

    attrs = _scan_v2_public_attrs(meta, scan_mode)
    attrs.update(meta["dim6_attrs"])

    voxel_to_probe = _build_scan_v2_voxel_to_probe(meta)
    voxel_to_world = _build_scan_v2_probe_to_lab(meta) @ voxel_to_probe

    data_array = create_voxeldata(
        data_lazy,
        dims=tuple(dims_list),
        time=_build_scan_v2_time_coord(meta) if include_time else None,
        pose=np.arange(npose) if include_pose else None,
        voxel_to_world=voxel_to_world,
        attrs=attrs,
        name=attrs.get("iconeus_scan") or path.stem,
    )
    if dim6_count > 1:
        data_array = data_array.assign_coords(dim6=np.arange(dim6_count))
    slice_time = _build_scan_v2_slice_time_coord(meta, include_time)
    if slice_time is not None:
        data_array = data_array.assign_coords(slice_time=slice_time)
        if include_time:
            data_array = data_array.assign_coords(
                time=build_consolidated_time_coordinate(
                    data_array.time, slice_time.values, dict(slice_time.attrs)
                )
            )
    return data_array


def _build_rotation_matrix(rx: float, ry: float, rz: float) -> npt.NDArray[np.float64]:
    """Build a 3x3 rotation matrix from intrinsic Z-Y-X Euler angles (radians).

    The Euler convention (order Z-Y-X) is assumed, not confirmed: the only non-trivial
    example so far has a single non-zero angle (`rz`), for which the order is irrelevant.

    Parameters
    ----------
    rx, ry, rz : float
        Rotation angles about the x, y, and z axes, in radians.

    Returns
    -------
    (3, 3) numpy.ndarray
        The composed rotation matrix `Rz @ Ry @ Rx`.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _build_scan_v2_probe_to_lab(meta: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Build `probe_to_lab` affine(s) from SCAN v2 6DOF pose arrays.

    Parameters
    ----------
    meta : dict
        Parsed SCAN v2 header fields.

    Returns
    -------
    numpy.ndarray
        `(4, 4)` affine for single-pose scans, or `(pose, 4, 4)` affine stack for
        multi-pose scans.

    Raises
    ------
    ValueError
        If a pose value is implausible.
    """
    translations = meta["probe_to_lab_translations"]
    rotations = meta["probe_to_lab_rotations"]
    if np.any(np.abs(translations) >= 1.0) or np.any(np.abs(rotations) > 2 * np.pi):
        raise ValueError("SCAN v2 probe-pose block is missing or implausible.")

    probe_to_lab = np.repeat(np.eye(4)[None, :, :], meta["npose"], axis=0)
    for pose, (translation, rotation) in enumerate(zip(translations, rotations)):
        probe_to_lab[pose, :3, :3] = _build_rotation_matrix(*rotation)
        probe_to_lab[pose, :3, 3] = translation
    converted = _build_probe_to_lab(probe_to_lab)
    return converted[0] if meta["npose"] == 1 else converted


def _scan_v2_mode(meta: dict[str, Any]) -> str:
    """Return a ConfUSIus scan-mode label for parsed SCAN v2 metadata.

    Parameters
    ----------
    meta : dict
        Parsed SCAN v2 header fields.

    Returns
    -------
    str
        One of `2Dscan`, `3Dscan`, `4Dscan`, or `4DscanCustom`.
    """
    code = meta.get("acquisition_mode_code")
    if code == 0:
        return "2Dscan"
    if meta["n_time"] <= 1:
        return "3Dscan"
    return "4DscanCustom" if code == 3 else "4Dscan"


def _scan_v2_public_attrs(meta: dict[str, Any], scan_mode: str) -> dict[str, Any]:
    """Return public DataArray attrs from parsed SCAN v2 metadata.

    Parameters
    ----------
    meta : dict
        Parsed SCAN v2 header fields.
    scan_mode : str
        ConfUSIus scan-mode label.

    Returns
    -------
    dict
        Attributes copied to the loaded DataArray.
    """
    keys = (
        "probe_model",
        "probe_model_number",
        "probe_center_frequency",
        "probe_pitch",
        "probe_elevation_aperture",
        "probe_radius_of_curvature",
        "probe_number_of_elements",
        "imaging_depth",
        "transmit_frequency",
        "pulse_repetition_frequency",
        "ultrafast_sampling_frequency",
        "plane_wave_angles",
        "transmit_voltage",
        "delay_after_trigger",
        "is_multiplane",
        "power_doppler_integration_duration",
        "iconeus_sequence",
        "iconeus_project",
        "iconeus_project_description",
        "iconeus_subject",
        "iconeus_session",
        "iconeus_species",
        "iconeus_gender",
        "iconeus_transfer_datetime",
        "iconeus_age_at_transfer",
        "iconeus_subject_description",
        "iconeus_weight_unit",
        "iconeus_weight",
        "iconeus_treatment",
        "iconeus_scan",
        "iconeus_study_type",
        "iconeus_task_name",
        "iconeus_task_description",
        "iconeus_experimenter",
        "iconeus_datetime",
        "iconeus_scan_type",
        "stimulation_toggle_times",
        "software_version",
    )
    attrs = {key: meta[key] for key in keys if key in meta}
    attrs.update(
        {
            "affines": {},
            "iconeus_scan_format": "v2",
            "iconeus_scan_mode": scan_mode,
        }
    )
    return attrs


def _build_scan_v2_voxel_to_probe(meta: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Build a v2 affine from ConfUSIus voxel coordinates to probe space.

    Parameters
    ----------
    meta : dict
        Parsed SCAN v2 header fields.

    Returns
    -------
    (4, 4) numpy.ndarray
        Affine mapping `(k, j, i)` voxel coordinates to `(z, y, x)` probe
        coordinates in millimeters.
    """
    dx_mm = meta["x_voxel_m"] * 1e3
    dz_mm = meta["y_voxel_m"] * 1e3
    dy_mm = meta["z_voxel_m"] * 1e3
    size_x = meta["size_x"]
    size_y = meta["size_y"]
    x0 = -((size_x - 1) / 2) * dx_mm
    z0 = -((size_y - 1) / 2) * dz_mm
    y0 = meta.get("depth_start_mm", 0.0)

    voxel_to_probe = np.eye(4, dtype=np.float64)
    voxel_to_probe[:3, :3] = np.diag([dz_mm, dy_mm, dx_mm])
    voxel_to_probe[:3, 3] = [z0, y0, x0]
    return voxel_to_probe


def _build_scan_v2_slice_time_coord(
    meta: dict[str, Any], include_time: bool
) -> xr.DataArray | None:
    """Build the `slice_time` coordinate for single-pose stacked SCAN v2 data.

    Parameters
    ----------
    meta : dict
        Parsed header fields from `_read_scan_v2_header`.
    include_time : bool
        Whether the loaded DataArray keeps a `time` dimension.

    Returns
    -------
    xarray.DataArray or None
        Absolute slice acquisition times with dims `(time, k)` or `(k,)`, or `None`
        when the file is not a single-pose stack.
    """
    if meta["npose"] != 1 or meta["size_y"] <= 1:
        return None

    values = meta["measured_times"].reshape(
        meta["n_time"], meta["npose"], meta["size_y"]
    )[:, 0, :]
    # Acquisition spacing includes idle time; only integration belongs in the window.
    attrs = _scan_time_attrs(float(meta["power_doppler_integration_duration"]))
    if include_time:
        return xr.DataArray(values, dims=["time", "k"], attrs=attrs)
    return xr.DataArray(values[0], dims=["k"], attrs=attrs)


def _build_scan_v2_time_coord(meta: dict[str, Any]) -> xr.DataArray:
    """Build the `time` coordinate DataArray for a SCAN v2 volume.

    Parameters
    ----------
    meta : dict
        Parsed header fields from `_read_scan_v2_header`.

    Returns
    -------
    xarray.DataArray
        `time` coordinate. Multi-pose scans use a pose-dependent `(time, pose)` array.
    """
    times = meta["measured_times"].reshape(
        meta["n_time"], meta["npose"], meta["size_y"]
    )
    time_vals = times.max(axis=2)
    time_attrs = _scan_time_attrs(float(time_vals.min()) if time_vals.size else 0.0)
    if meta["npose"] == 1:
        return xr.DataArray(time_vals[:, 0], dims=["time"], attrs=time_attrs)
    return xr.DataArray(time_vals, dims=["time", "pose"], attrs=time_attrs)
