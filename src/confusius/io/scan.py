"""Utilities for loading Iconeus SCAN files.

This module provides functions to load Iconeus SCAN (`.scan`) HDF5 files as lazy Xarray
DataArrays using h5py and Dask for out-of-core processing.
"""

from pathlib import Path
from typing import Any

import dask.array as da
import h5py
import numpy as np
import numpy.typing as npt
import xarray as xr

from confusius.io.utils import check_path

# Permutation matrix P that maps ConfUSIus physical input (z_conf, y_conf, x_conf, 1)
# to probe physical (x_probe, y_probe, z_probe, 1):
#
#   x_probe =  x_conf      (lateral, same direction)
#   y_probe =  z_conf      (elevation, same direction)
#   z_probe = -y_conf      (axial depth, sign flip: y_conf = -z_probe > 0)
#
# Its transpose P^T maps probe physical (x_probe, y_probe, z_probe, 1) back to
# ConfUSIus physical (z_conf, y_conf, x_conf, 1):
#
#   z_conf =  y_probe      (elevation)
#   y_conf = -z_probe      (depth, sign flip)
#   x_conf =  x_probe      (lateral)
_PHYSICAL_TO_PROBE_PERMUTATION: npt.NDArray[np.float64] = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)


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


def _infer_scan_volume_acquisition_duration(
    time_values: npt.ArrayLike,
) -> float | None:
    """Infer SCAN end-reference acquisition duration from the first timestamp.

    Iconeus SCAN stores timestamps using an `end` reference. When acquisition is
    triggered at `t=0`, the first timestamp therefore equals the duration between the
    start and end reference points for the exported `time` coordinate.

    Parameters
    ----------
    time_values : array_like
        One-dimensional sequence of SCAN timestamps for one acquisition stream.

    Returns
    -------
    float or None
        Inferred acquisition duration in seconds. Returns `None` when no timestamps are
        available or when the first timestamp is non-positive.
    """
    flattened = np.asarray(time_values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return None

    first_timestamp = float(flattened[0])
    if first_timestamp <= 0:
        return None

    return first_timestamp


def _coords_from_voxels_to_probe(
    voxels_to_probe: npt.NDArray[np.float64],
    size_x: int,
    size_y: int,
    size_z: int,
) -> dict[str, xr.DataArray]:
    """Build spatial coordinate arrays in millimeters from `voxelsToProbe`.

    The `voxelsToProbe` affine maps one-indexed (probably because MATLAB-based) voxel
    integer indices `(ix, iy, iz)` to physical probe coordinates in meters. Coordinates
    are multiplied by `1e3` to convert to millimeters, consistent with all other
    ConfUSIus loaders.

    Dimension mapping (probe -> ConfUSIus):

    - Probe `x_probe` (lateral, row 0) -> ConfUSIus `x`.
    - Probe `y_probe` (elevation, row 1) -> ConfUSIus `z`.
    - Probe `z_probe` (axial depth, row 2, negative diagonal) -> ConfUSIus `y`
      with a sign flip so that `y` is always positive and increases with depth.

    Parameters
    ----------
    voxels_to_probe : (4, 4) numpy.ndarray
        `voxelsToProbe` affine from a SCAN files (units meters).
    size_x : int
        Number of lateral voxels.
    size_y : int
        Number of elevation voxels per position (`sizeY`).
    size_z : int
        Number of axial voxels (`sizeZ`).

    Returns
    -------
    dict[str, xarray.DataArray]
        Coordinate DataArrays keyed by `"x"`, `"y"`, and `"z"`.
    """
    # MATLAB-based voxelsToProbe uses one-indexed voxels; Python uses zero-indexed. Add
    # 1 to voxel indices before applying the affine.
    x_vals = 1e3 * (
        voxels_to_probe[0, 0] * (np.arange(size_x) + 1) + voxels_to_probe[0, 3]
    )
    z_vals = 1e3 * (
        voxels_to_probe[1, 1] * (np.arange(size_y) + 1) + voxels_to_probe[1, 3]
    )
    # v2p[2,2] is negative (-dz), so negating gives positive depth values.
    y_vals = 1e3 * (
        -(voxels_to_probe[2, 2] * (np.arange(size_z) + 1) + voxels_to_probe[2, 3])
    )

    x_voxdim = float(1e3 * abs(voxels_to_probe[0, 0]))
    z_voxdim = float(1e3 * abs(voxels_to_probe[1, 1]))
    y_voxdim = float(1e3 * abs(voxels_to_probe[2, 2]))

    coords: dict[str, xr.DataArray] = {
        "x": xr.DataArray(
            x_vals, dims=["x"], attrs={"units": "mm", "voxdim": x_voxdim}
        ),
        "z": xr.DataArray(
            z_vals, dims=["z"], attrs={"units": "mm", "voxdim": z_voxdim}
        ),
        "y": xr.DataArray(
            y_vals, dims=["y"], attrs={"units": "mm", "voxdim": y_voxdim}
        ),
    }
    return coords


def _build_physical_to_lab(
    probe_to_lab: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert `probeToLab` to a ConfUSIus `physical_to_lab` affine in mm.

    `probeToLab` maps probe physical `(x_probe, y_probe, z_probe, 1)` to Iconeus lab
    space in metres. The Iconeus lab frame is aligned with the probe axes:
    `x_lab = lateral`, `y_lab = elevation`, `z_lab = axial`.

    We want `physical_to_lab` to map ConfUSIus physical `(z_conf, y_conf, x_conf, 1)`
    to **ConfUSIus-ordered** lab space `(z_lab, y_lab, x_lab)` (elevation, depth,
    lateral) in millimetres, using the same permutation `P` that maps between the two
    physical spaces:

    ```
    physical_to_lab = P^T @ probeToLab @ P
    ```

    where `P = _PHYSICAL_TO_PROBE_PERMUTATION`. This produces a ConfUSIus-ordered affine
    whose rotation block is identity for a non-rotated probe, making it directly usable
    in napari and other tools that expect `(z, y, x)` axis order.

    Parameters
    ----------
    probe_to_lab : (4, 4) or (npose, 4, 4) numpy.ndarray
        `probeToLab` affine(s) from a SCAN file (units metres).

    Returns
    -------
    numpy.ndarray
        `physical_to_lab` affine(s) in millimetres. Shape matches input: `(4, 4)` for
        `2Dscan` or `(npose, 4, 4)` for `3Dscan`/`4Dscan`.
    """
    P = _PHYSICAL_TO_PROBE_PERMUTATION
    physical_to_lab = P.T @ probe_to_lab @ P
    physical_to_lab[..., :3, 3] *= 1e3
    return physical_to_lab


def load_scan(
    path: str | Path,
    chunks: int | tuple[int, ...] | str | None = "auto",
) -> xr.DataArray:
    """Load an Iconeus SCAN file as a lazy Xarray DataArray.

    SCAN files (`.scan`) are HDF5 files produced by IcoScan/NeuroScan. They contain
    power Doppler data and spatial/temporal metadata for 2D, 3D, or 3D+t fUSI volumes.

    The returned DataArray wraps an open `h5py` file handle via a Dask array. The
    file remains open while the Dask graph is un-computed. Call `.compute()` before
    closing the file, or keep the returned DataArray in scope to prevent the handle from
    being garbage-collected.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the SCAN file (`.scan`).
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
        Lazy DataArray with dimensions and coordinates:

        - `2Dscan` → `(time, z, y, x)`.
        - `3Dscan` → `(pose, z, y, x)`.
        - `4Dscan` → `(time, pose, z, y, x)`.

        All spatial coordinates are in millimeters. The `time` coordinate is in
        seconds. For `4Dscan`, a `pose_time` non-dimension coordinate of shape
        `(time, pose)` stores the actual per-pose acquisition timestamps.

    Raises
    ------
    ValueError
        If `path` does not exist or is not a file, or if the `acquisitionMode` stored in
        the file is not one of `"2Dscan"`, `"3Dscan"`, or `"4Dscan"`.

    Notes
    -----
    The `physical_to_lab` affine stored in `da.attrs["affines"]` maps ConfUSIus physical
    coordinates `(z, y, x)` to Iconeus lab coordinates (mm). Apply as
    `da.attrs["affines"]["physical_to_lab"] @ np.array([z, y, x, 1.0])`. For multi-pose
    files the shape is `(npose, 4, 4)`; index with `da.coords["pose"].values` after
    `isel`.

    Provenance attributes are stored in `da.attrs`: BIDS-compatible fields
    (`device_serial_number`, `software_version`) and Iconeus-specific fields
    (`iconeus_scan_mode`, `iconeus_subject`, `iconeus_session`, `iconeus_scan`,
    `iconeus_project`, `iconeus_date`).
    """
    path = check_path(path, type="file")

    h5 = h5py.File(path, "r")

    try:
        mode = _read_scan_str(h5, "/acqMetaData/acquisitionMode")

        size_x = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/sizeX"))
        size_y = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/sizeY"))
        size_z = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/sizeZ"))
        npose = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/npose"))
        nblock_repeat = int(_read_scan_scalar(h5, "/acqMetaData/imgDim/nblockRepeat"))

        voxels_to_probe: npt.NDArray[np.float64] = np.array(
            h5["/acqMetaData/voxelsToProbe"][()], dtype=np.float64
        )
        probe_to_lab: npt.NDArray[np.float64] = np.array(
            h5["/acqMetaData/probeToLab"][()], dtype=np.float64
        )

        spatial_coords = _coords_from_voxels_to_probe(
            voxels_to_probe, size_x, size_y, size_z
        )
        physical_to_lab = _build_physical_to_lab(probe_to_lab)

        attrs: dict[str, Any] = {
            "affines": {"physical_to_lab": physical_to_lab},
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
            data_array = _load_2dscan(h5, raw_lazy, spatial_coords, attrs)
        elif mode == "3Dscan":
            data_array = _load_3dscan(raw_lazy, spatial_coords, attrs, npose)
        elif mode == "4Dscan":
            data_array = _load_4dscan(
                h5, raw_lazy, spatial_coords, attrs, npose, nblock_repeat
            )
        else:
            raise ValueError(
                f"Unknown acquisitionMode: {mode!r}. Expected one of '2Dscan',"
                " '3Dscan', '4Dscan'."
            )

        name_parts = [
            attrs["iconeus_subject"],
            attrs["iconeus_session"],
            attrs["iconeus_scan"],
        ]
        data_array.name = "_".join(p for p in name_parts if p) or path.stem
    except Exception:
        h5.close()
        raise

    return data_array


def _load_2dscan(
    h5: h5py.File,
    raw_lazy: da.Array,
    spatial_coords: dict[str, xr.DataArray],
    attrs: dict[str, Any],
) -> xr.DataArray:
    """Build a DataArray for `2Dscan` mode.

    Raw shape (h5py, C-order): `(nblockRepeat, sizeZ, sizeY, sizeX)`. Output dims:
    `(time, z, y, x)`.

    `nblockRepeat` is the number of time frames; the HDF5 file stores depth (`sizeZ`)
    before elevation (`sizeY`), while ConfUSIus uses `(z=elevation, y=depth)`, so axes 1
    and 2 are swapped.

    The `h5` handle is kept open because `raw_lazy` wraps an h5py dataset; it must
    remain open until the Dask graph is computed.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    raw_lazy : dask.array.Array
        Lazy Dask array wrapping `/Data`.
    spatial_coords : dict[str, xarray.DataArray]
        Spatial coordinate arrays for `x`, `z`, `y`.
    attrs : dict
        Provenance and affine attributes.

    Returns
    -------
    xarray.DataArray
        DataArray with dims `(time, z, y, x)`.
    """
    # Swap depth (axis 1) and elevation (axis 2): HDF5 order is (sizeZ, sizeY),
    # ConfUSIus order is (z=elevation, y=depth).
    data_lazy = da.transpose(raw_lazy, [0, 2, 1, 3])

    # The orientation of the time array is inconsistent across file versions; squeeze
    # handles both (1, T) and (T, 1).
    time_raw: npt.NDArray[np.float64] = np.array(
        h5["/acqMetaData/time"][()], dtype=np.float64
    )
    time_vals = time_raw.squeeze()
    time_attrs: dict[str, Any] = {
        "units": "s",
        "volume_acquisition_reference": "end",
    }
    volume_acquisition_duration = _infer_scan_volume_acquisition_duration(time_vals)
    if volume_acquisition_duration is not None:
        time_attrs["volume_acquisition_duration"] = volume_acquisition_duration

    coords: dict[str, Any] = {
        "time": xr.DataArray(
            time_vals,
            dims=["time"],
            # Iconeus SCAN files store end-referenced timestamps.
            attrs=time_attrs,
        ),
        **spatial_coords,
    }

    return xr.DataArray(
        data_lazy, dims=["time", "z", "y", "x"], coords=coords, attrs=attrs
    )


def _load_3dscan(
    raw_lazy: da.Array,
    spatial_coords: dict[str, xr.DataArray],
    attrs: dict[str, Any],
    npose: int,
) -> xr.DataArray:
    """Build a DataArray for `3Dscan` mode.

    Raw shape (h5py, C-order): `(npose, nblockRepeat, sizeZ, sizeY, sizeX)`. Output
    dims: `(pose, z, y, x)`.

    `nblockRepeat` is always 1 in practice for anatomical 3D scans, so it is squeezed
    away. The HDF5 file stores depth (`sizeZ`) before elevation (`sizeY`), while
    ConfUSIus uses `(z=elevation, y=depth)`, so axes 1 and 2 (after the squeeze) are
    swapped.

    The `3Dscan` `acqMetaData/time` array (shape `(npose, 1)`) holds one timestamp per
    robot position. For anatomical scans these are often all zero and carry no
    physiological meaning; they are intentionally not exposed as a coordinate. Use
    `4Dscan` mode when per-block timing is required.

    `raw_lazy` wraps an open h5py dataset; the caller must keep the file handle open
    until the Dask graph is computed.

    Parameters
    ----------
    raw_lazy : dask.array.Array
        Lazy Dask array wrapping `/Data`.
    spatial_coords : dict[str, xarray.DataArray]
        Spatial coordinate arrays for `x`, `z`, `y`.
    attrs : dict
        Provenance and affine attributes.
    npose : int
        Number of robot positions.

    Returns
    -------
    xarray.DataArray
        DataArray with dims `(pose, z, y, x)`.
    """
    # axis=1 is nblockRepeat=1; squeeze it away before transposing.
    sq = da.squeeze(raw_lazy, axis=1)
    # Swap depth (axis 1) and elevation (axis 2): HDF5 order is (sizeZ, sizeY),
    # ConfUSIus order is (z=elevation, y=depth).
    data_lazy = da.transpose(sq, [0, 2, 1, 3])

    pose_vals = np.arange(npose)
    coords: dict[str, Any] = {
        "pose": xr.DataArray(pose_vals, dims=["pose"]),
        **spatial_coords,
    }

    return xr.DataArray(
        data_lazy, dims=["pose", "z", "y", "x"], coords=coords, attrs=attrs
    )


def _load_4dscan(
    h5: h5py.File,
    raw_lazy: da.Array,
    spatial_coords: dict[str, xr.DataArray],
    attrs: dict[str, Any],
    npose: int,
    nblock_repeat: int,
) -> xr.DataArray:
    """Build a DataArray for `4Dscan` mode.

    Raw shape (h5py, C-order): `(nscanRepeat, npose, nblockRepeat, sizeZ, sizeY,
    sizeX)`. Output dims: `(time, pose, z, y, x)`.

    `nblockRepeat` is a per-pose repetition count. When `nblockRepeat > 1`, the
    `nscanRepeat` and `nblockRepeat` axes are combined into a single `time` axis of
    length `nscanRepeat * nblockRepeat` by transposing to
    `(nscanRepeat, nblockRepeat, npose, sizeZ, sizeY, sizeX)` and reshaping. When
    `nblockRepeat == 1` the axis is simply squeezed away.

    The HDF5 file stores depth (`sizeZ`) before elevation (`sizeY`), while ConfUSIus
    uses `(z=elevation, y=depth)`, so those two spatial axes are swapped.

    The `h5` handle is kept open because `raw_lazy` wraps an h5py dataset; it must
    remain open until the Dask graph is computed.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    raw_lazy : dask.array.Array
        Lazy Dask array wrapping `/Data`.
    spatial_coords : dict[str, xarray.DataArray]
        Spatial coordinate arrays for `x`, `z`, `y`.
    attrs : dict
        Provenance and affine attributes.
    npose : int
        Number of robot positions.
    nblock_repeat : int
        Number of block repeats per scan repeat (`nblockRepeat` from the file).

    Returns
    -------
    xarray.DataArray
        DataArray with dims `(time, pose, z, y, x)`.
    """
    nscan_repeat = raw_lazy.shape[0]
    n_time = nscan_repeat * nblock_repeat

    if nblock_repeat == 1:
        # axis=2 is the nblockRepeat=1 singleton; squeeze it away.
        sq = da.squeeze(raw_lazy, axis=2)
    else:
        # Transpose to (nscanRepeat, nblockRepeat, npose, sizeZ, sizeY, sizeX),
        # then reshape to (n_time, npose, sizeZ, sizeY, sizeX).
        transposed = da.transpose(raw_lazy, [0, 2, 1, 3, 4, 5])
        sq = transposed.reshape(n_time, npose, *raw_lazy.shape[3:])

    # Swap depth (axis 2) and elevation (axis 3): HDF5 order is (sizeZ, sizeY),
    # ConfUSIus order is (z=elevation, y=depth).
    data_lazy = da.transpose(sq, [0, 1, 3, 2, 4])

    # Raw time shape is (npose * nscanRepeat * nblockRepeat, 1) or its transpose;
    # squeeze normalises both orientations before reshaping.
    time_raw: npt.NDArray[np.float64] = (
        np.array(h5["/acqMetaData/time"][()], dtype=np.float64)
        .squeeze()
        .reshape(n_time, npose)
    )

    # Iconeus SCAN files store end-referenced timestamps.
    block_time = time_raw.max(axis=1)
    time_attrs: dict[str, Any] = {
        "units": "s",
        "volume_acquisition_reference": "end",
    }
    volume_acquisition_duration = _infer_scan_volume_acquisition_duration(block_time)
    if volume_acquisition_duration is not None:
        time_attrs["volume_acquisition_duration"] = volume_acquisition_duration

    coords: dict[str, Any] = {
        "time": xr.DataArray(block_time, dims=["time"], attrs=time_attrs),
        "pose": xr.DataArray(np.arange(npose), dims=["pose"]),
        "pose_time": xr.DataArray(time_raw, dims=["time", "pose"], attrs=time_attrs),
        **spatial_coords,
    }

    return xr.DataArray(
        data_lazy, dims=["time", "pose", "z", "y", "x"], coords=coords, attrs=attrs
    )
