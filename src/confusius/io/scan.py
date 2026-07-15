"""Utilities for loading Iconeus SCAN files.

Iconeus ships two on-disk SCAN formats, both using the `.scan` extension:

- **v1**: an HDF5 container (`acqMetaData`, `scanMetaData`, `/Data`). Loaded lazily with
  h5py and Dask.
- **v2**: a flat binary file with a variable-length header followed by a little-endian
  `float64` power-Doppler payload. Loaded lazily with a NumPy memmap wrapped in Dask.

`load_scan` sniffs the format and dispatches to the matching loader. The v2 loader is
**experimental**: its field offsets were reverse-engineered from a small number of
example files and may not cover every acquisition mode. See `_load_scan_v2` for details.
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
    "nblock_repeat": 0x84,
    "dt": 0x94,
    "svd_clutter_cutoff": 0x9C,
    "power_doppler_integration_window": 0xA0,
    "x_voxel_m": 0xA4,
    "y_voxel_m": 0xAC,
    "z_voxel_m": 0xB4,
    "time_coords": 0xD4,
}
"""Byte offsets of fixed-position fields in the SCAN v2 header.

All fields listed here live before the first variable-length string in the header, so
their absolute offsets are stable across files. Integer fields are little-endian
`uint64`; `dt` and the voxel sizes are little-endian `float64`; `time_coords` is the
start of an array of `n_time` little-endian `float64` values.
"""

PHYSICAL_TO_PROBE_PERMUTATION: npt.NDArray[np.float64] = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)
"""Permutation matrix that maps ConfUSIus physical to probe physical.

ConfUSIus input (z_conf, y_conf, x_conf, 1) is mapped to the probe physical (x_probe,
y_probe, z_probe, 1):

  x_probe =  x_conf      (lateral, same direction)
  y_probe =  z_conf      (elevation, same direction)
  z_probe = -y_conf      (axial depth, sign flip: y_conf = -z_probe > 0)

Its transpose maps probe physical (x_probe, y_probe, z_probe, 1) back to ConfUSIus
physical (z_conf, y_conf, x_conf, 1):

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
    space `(x_lab, y_lab, z_lab, 1)` in metres. The Iconeus lab frame is a fixed scanner
    frame; `probeToLab` carries any rotation of the probe within it.

    We want `physical_to_lab` to map ConfUSIus physical `(z_conf, y_conf, x_conf, 1)`
    (elevation, depth, lateral) to **ConfUSIus-ordered** lab space `(z_lab, y_lab,
    x_lab)` in millimetres, using the same permutation `P` that maps between the two
    physical spaces:

    ```python
    physical_to_lab = PHYSICAL_TO_PROBE_PERMUTATION^T @ probeToLab @ PHYSICAL_TO_PROBE_PERMUTATION
    ```

    This produces a ConfUSIus-ordered affine whose rotation block is identity for a
    non-rotated probe, making it directly usable in napari and other tools that expect
    `(z, y, x)` axis order.

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
    physical_to_lab = (
        PHYSICAL_TO_PROBE_PERMUTATION.T @ probe_to_lab @ PHYSICAL_TO_PROBE_PERMUTATION
    )
    physical_to_lab[..., :3, 3] *= 1e3
    return physical_to_lab


def load_bps(bps_path: str | Path) -> npt.NDArray[np.float64]:
    """Load a BPS file and return an affine from Iconeus' brain space to ConfUSIus lab space.

    BPS files are HDF5 sidecars produced by Iconeus' brain positioning system. They
    store a `BrainToLab` affine that maps Iconeus brain coordinates `(x_brain, y_brain,
    z_brain, 1)` to Iconeus lab coordinates `(x_lab, y_lab, z_lab, 1)` in meters.
    The Iconeus lab frame is a fixed scanner frame; `probeToLab` carries any rotation
    of the probe within it.

    To compose this affine with the rest of the ConfUSIus pipeline we re-express
    the lab side as **ConfUSIus-ordered** lab space `(z_lab, y_lab, x_lab)` in
    millimeters, matching the convention used by `physical_to_lab` (see
    `_build_physical_to_lab`). The brain side is left in its original axis order
    (the brain coordinate units are not declared by the BPS format and are
    therefore not converted).

    The change of basis from ConfUSIus-ordered millimetre lab coordinates to
    Iconeus-ordered metre lab coordinates is

    ```
    confusius_lab_to_iconeus_lab = mm_to_m @ PHYSICAL_TO_PROBE_PERMUTATION
    ```

    `PHYSICAL_TO_PROBE_PERMUTATION` permutes the axes from ConfUSIus order `(z, y, x)`
    to probe / Iconeus-lab order `(x, y, z)`, and `mm_to_m = diag(1e-3, 1e-3, 1e-3, 1)`
    rescales the translation column. The returned affine is then

    ```
    brain_to_confusius_lab = inv(confusius_lab_to_iconeus_lab) @ BrainToLab
    ```

    Parameters
    ----------
    bps_path : str or pathlib.Path
        Path to the BPS file (`.bps`).

    Returns
    -------
    (4, 4) numpy.ndarray
        Affine mapping Iconeus brain coordinates to ConfUSIus-ordered Iconeus lab
        coordinates `(z_lab, y_lab, x_lab, 1)` in millimetres.
    """
    bps_path = check_path(bps_path, label="bps_path", type="file")

    with h5py.File(bps_path, "r") as f:
        brain_to_lab = f["BrainToLab"][:]

    mm_to_m = np.diag([1e-3, 1e-3, 1e-3, 1.0])
    confusius_lab_to_iconeus_lab = mm_to_m @ PHYSICAL_TO_PROBE_PERMUTATION

    brain_to_confusius_lab = np.linalg.inv(confusius_lab_to_iconeus_lab) @ brain_to_lab
    return brain_to_confusius_lab


def load_scan(
    path: str | Path,
    bps_path: str | Path | None = None,
    chunks: int | tuple[int, ...] | str | None = "auto",
) -> xr.DataArray:
    """Load an Iconeus SCAN file as a lazy Xarray DataArray.

    SCAN files (`.scan`) come in two on-disk formats, both handled here:

    - **v1**: an HDF5 container produced by IcoScan/NeuroScan, holding power Doppler
      data and spatial/temporal metadata for 2D, 3D, or 3D+t fUSI volumes. The returned
      DataArray wraps an open `h5py` handle via a Dask array; keep it in scope (or call
      `.compute()`) before the handle is garbage-collected.
    - **v2**: a flat binary file (variable-length header + little-endian `float64`
      payload). The returned DataArray wraps a NumPy memmap via a Dask array. Support is
      **experimental** (see Notes); `bps_path` is not yet supported for v2.

    `load_scan` sniffs the format automatically and dispatches accordingly.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the SCAN file (`.scan`).
    bps_path : str or pathlib.Path, optional
        Path to the corresponding BPS file (`.bps`). If provided, the BPS transformation
        matrix will be added as an affine attribute to the returned DataArray. Only
        supported for v1 (HDF5) files.
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

        - v1 `2Dscan` → `(time, z, y, x)`.
        - v1 `3Dscan` → `(pose, z, y, x)`.
        - v1 `4Dscan` → `(time, pose, z, y, x)`.
        - v2 single-pose → `(time, z, y, x)`.
        - v2 multi-pose → `(time, pose, z, y, x)`.

        All spatial coordinates are in millimeters. The `time` coordinate is in
        seconds. For v1 `4Dscan`, a `pose_time` non-dimension coordinate of shape
        `(time, pose)` stores the actual per-pose acquisition timestamps.

    Raises
    ------
    ValueError
        If `path` does not exist or is not a file, if the file is neither an
        HDF5-based SCAN (v1) nor a binary SCAN v2 file, if `bps_path` is passed for a
        v2 file, or if a v1 `acquisitionMode` is not one of `"2Dscan"`, `"3Dscan"`, or
        `"4Dscan"`.

    Notes
    -----
    **v2 (experimental).** The v2 field offsets were reverse-engineered from a small
    number of example files. Data, temporal geometry, and voxel spacing are recovered.
    The depth (`y`) origin is read from the header when the depth range can be located;
    the lateral (`x`) and elevation (`z`) origins are not encoded, so those axes are
    centred on zero (correct spacing, arbitrary origin). No `probeToLab`-equivalent
    affine has been located, so v2 DataArrays carry **no** `physical_to_lab` affine.
    Multi-pose / multi-block v2 layouts are inferred by analogy with v1 and have not
    been validated against real files. Provenance strings are mapped to v1-style
    `iconeus_*` fields heuristically (by position, with the hex-encoded serial/hardware
    strings as anchors); the raw decoded strings are always kept in
    `iconeus_header_strings` for manual re-mapping.

    Acquisition settings that correspond to fUSI-BIDS fields are also surfaced as
    attributes, in native header units: `probe_model`, `probe_center_frequency` (MHz),
    `probe_pitch` (mm), `probe_focal_depth` (mm), `imaging_depth` (mm start/end),
    `transmit_frequency` (MHz), `pulse_repetition_frequency` (Hz), `plane_wave_angles`
    (deg), `svd_low_cutoff`, and `power_doppler_integration_window`. The probe/sequence
    subset is parsed from a structured block whose layout is anchor-validated against the
    depth origin; if that check fails, those fields are omitted rather than guessed.

    The `physical_to_lab` affine stored in `da.attrs["affines"]` maps ConfUSIus physical
    coordinates `(z, y, x)` to **ConfUSIus-ordered** Iconeus lab coordinates (mm). Apply
    as `da.attrs["affines"]["physical_to_lab"] @ np.array([z, y, x, 1.0])`. For
    multi-pose files the shape is `(npose, 4, 4)`; index with `da.coords["pose"].values`
    after `isel`.

    If `bps_path` is provided, a `physical_to_brain` affine is stored in
    `da.attrs["affines"]["physical_to_brain"]` that maps ConfUSIus physical coordinates
    `(z, y, x)` to Iconeus' brain coordinates. Apply as
    `da.attrs["affines"]["physical_to_brain"] @ np.array([z, y, x, 1.0])`.

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
        if bps_path is not None:
            raise ValueError(
                "bps_path is not supported for SCAN v2 files: the physical_to_lab "
                "affine needed to compose the BPS transform has not yet been located "
                "in the v2 header."
            )
        # v2 support is reverse-engineered and incomplete: re-raise any parse failure
        # with a pointer to the issue tracker and the original error appended.
        try:
            return _load_scan_v2(path, chunks)
        except Exception as error:
            raise ValueError(
                "Loading Iconeus SCAN v2 files is experimental and this file could "
                "not be parsed. Please open an issue at "
                "https://github.com/confusius-tools/confusius/issues (attaching an "
                f"example file if possible).\n\nOriginal error: "
                f"{type(error).__name__}: {error}"
            ) from error

    raise ValueError(
        f"{path.name!r} is not a SCAN file recognised by ConfUSIus. Expected an "
        f"HDF5-based SCAN (v1) or a binary SCAN v2 file starting with the magic bytes "
        f"{SCAN_V2_MAGIC!r}."
    )


def _load_scan_v1(
    path: Path,
    bps_path: str | Path | None,
    chunks: int | tuple[int, ...] | str | None,
) -> xr.DataArray:
    """Load an HDF5-based Iconeus SCAN (v1) file as a lazy DataArray.

    Parameters
    ----------
    path : pathlib.Path
        Path to the v1 SCAN file, already validated as HDF5.
    bps_path : str or pathlib.Path, optional
        Path to the corresponding BPS file (`.bps`). If provided, a `physical_to_brain`
        affine is added to `da.attrs["affines"]`.
    chunks : int or tuple[int, ...] or str or None
        Dask chunk specification passed to `dask.array.from_array`.

    Returns
    -------
    xarray.DataArray
        Lazy DataArray whose dims depend on the file's `acquisitionMode`. See
        `load_scan` for the full contract.

    Raises
    ------
    ValueError
        If the `acquisitionMode` stored in the file is not one of `"2Dscan"`,
        `"3Dscan"`, or `"4Dscan"`.
    """
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

        data_array.name = attrs["iconeus_scan"] or path.stem
        if bps_path is not None:
            brain_to_lab = load_bps(bps_path)
            physical_to_brain = np.linalg.inv(brain_to_lab) @ physical_to_lab
            data_array.attrs["affines"]["physical_to_brain"] = physical_to_brain
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
    time: npt.NDArray[np.float64] = np.array(
        h5["/acqMetaData/time"][()], dtype=np.float64
    ).squeeze()

    # Iconeus SCAN files store end-referenced timestamps.
    time_attrs: dict[str, Any] = {
        "units": "s",
        "volume_acquisition_reference": "end",
        # Infer per-pose duration from the earliest time point recorded. Since
        # timestamps are end-referenced, the minimum timestamp corresponds to the
        # duration of the first acquired volume.
        "volume_acquisition_duration": time.min(),
    }

    coords: dict[str, Any] = {
        "time": xr.DataArray(time, dims=["time"], attrs=time_attrs),
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
        # Infer per-pose duration from the earliest time point recorded. Since
        # timestamps are end-referenced, the minimum timestamp corresponds to the
        # duration of the first acquired volume.
        "volume_acquisition_duration": time_raw.min(),
    }

    coords: dict[str, Any] = {
        "time": xr.DataArray(block_time, dims=["time"], attrs=time_attrs),
        "pose": xr.DataArray(np.arange(npose), dims=["pose"]),
        "pose_time": xr.DataArray(time_raw, dims=["time", "pose"], attrs=time_attrs),
        **spatial_coords,
    }

    return xr.DataArray(
        data_lazy, dims=["time", "pose", "z", "y", "x"], coords=coords, attrs=attrs
    )


def _read_scan_v2_header(header: bytes) -> dict[str, Any]:
    """Parse the fixed-position numeric fields of a SCAN v2 header.

    Only the fields listed in `_SCAN_V2_OFFSETS` are read. These all live before the
    first variable-length string, so their absolute offsets are stable across files.
    Provenance strings, which follow the variable-length region, are handled separately
    by `_scan_v2_strings`.

    Parameters
    ----------
    header : bytes
        The full header bytes (`total_header_bytes` long).

    Returns
    -------
    dict
        Parsed fields: `size_x`, `size_y`, `size_z`, `n_time`, `npose`,
        `nblock_repeat`, `payload_bytes`, `total_header_bytes`, `svd_clutter_cutoff`,
        `power_doppler_integration_window` (int); `dt`, `x_voxel_m`, `y_voxel_m`,
        `z_voxel_m` (float); and `time_coords` (`(n_time,) numpy.ndarray`).

    Raises
    ------
    ValueError
        If the header is too short to contain the fixed fields, or if any dimension is
        not a positive integer.
    """
    o = _SCAN_V2_OFFSETS

    def u32(offset: int) -> int:
        return int.from_bytes(header[offset : offset + 4], "little")

    def u64(offset: int) -> int:
        return int.from_bytes(header[offset : offset + 8], "little")

    def f64(offset: int) -> float:
        return float(struct.unpack_from("<d", header, offset)[0])

    n_time = u64(o["n_time"])
    time_end = o["time_coords"] + 8 * n_time
    if n_time < 1 or time_end > len(header):
        raise ValueError(
            "SCAN v2 header is truncated or reports an implausible time-point count "
            f"(n_time={n_time}, header={len(header)} bytes)."
        )

    fields: dict[str, Any] = {
        "size_x": u64(o["size_x"]),
        "size_y": u64(o["size_y"]),
        "size_z": u64(o["size_z"]),
        "n_time": n_time,
        "npose": u64(o["npose"]),
        "nblock_repeat": u64(o["nblock_repeat"]),
        "payload_bytes": u64(o["payload_bytes"]),
        "total_header_bytes": u64(o["total_header_bytes"]),
        "dt": f64(o["dt"]),
        "svd_clutter_cutoff": u32(o["svd_clutter_cutoff"]),
        "power_doppler_integration_window": u32(o["power_doppler_integration_window"]),
        "x_voxel_m": f64(o["x_voxel_m"]),
        "y_voxel_m": f64(o["y_voxel_m"]),
        "z_voxel_m": f64(o["z_voxel_m"]),
        "time_coords": np.frombuffer(
            header, dtype="<f8", count=n_time, offset=o["time_coords"]
        ).copy(),
    }

    for key in ("size_x", "size_y", "size_z", "npose", "nblock_repeat"):
        if fields[key] < 1:
            raise ValueError(
                f"SCAN v2 header reports a non-positive {key}={fields[key]}; the file "
                "may be corrupt or use an unsupported layout."
            )

    return fields


def _scan_v2_strings(header: bytes) -> list[tuple[str, bool, int]]:
    """Extract length-prefixed strings from a SCAN v2 header, best-effort.

    The provenance region stores strings as a `uint32` byte-length prefix followed by
    that many ASCII bytes. This walker greedily collects every position whose prefix is
    followed by a non-empty printable run of exactly the stated length. Some Iconeus
    fields store the ASCII **as hex** (e.g. serial number, hardware label); those are
    decoded once more so the readable text is returned, and flagged as hex so callers
    can use them as structural anchors.

    Because the strings are variable-length and interleaved with numeric fields, this is
    a heuristic recovery, not a schema parse: it may miss fields or pick up spurious
    matches. See `_scan_v2_provenance` for how the result is mapped to named fields.

    Parameters
    ----------
    header : bytes
        The full header bytes.

    Returns
    -------
    list[tuple[str, bool, int]]
        `(text, is_hex, end)` triples in the order they appear in the header, where
        `is_hex` marks values that were hex-decoded and `end` is the byte offset just
        past the record (used to locate the trailing timestamp).
    """
    records: list[tuple[str, bool, int]] = []
    i = 0
    n = len(header)
    while i + 4 <= n:
        length = int.from_bytes(header[i : i + 4], "little")
        if 1 <= length <= 256 and i + 4 + length <= n:
            body = header[i + 4 : i + 4 + length]
            if all(32 <= c < 127 for c in body):
                text = body.decode("ascii")
                is_hex = False
                # Some fields (serial number, hardware label) are stored as hex-encoded
                # ASCII; decode a second time when the run is valid hex.
                if length % 2 == 0:
                    try:
                        decoded = bytes.fromhex(text)
                        if all(32 <= c < 127 for c in decoded):
                            text = decoded.decode("ascii")
                            is_hex = True
                    except ValueError:
                        pass
                records.append((text, is_hex, i + 4 + length))
                i += 4 + length
                continue
        i += 1
    return records


def _scan_v2_datetime(header: bytes, records: list[tuple[str, bool, int]]) -> str:
    """Recover the acquisition timestamp from the SCAN v2 header, best-effort.

    The header stores the acquisition time as a `uint64` Unix timestamp (full seconds
    precision), but its offset shifts between files and several unrelated fields fall in
    a plausible date range, so a blind scan is unreliable. Instead this anchors on
    structure: the acquisition timestamp is the first `uint64` in a plausible range
    (years ~2015–2035) that follows the leading provenance strings, so the scan starts
    at the end of the fifth plain string (species). A later, second timestamp — the file
    save time — is deliberately skipped by starting after the acquisition one.

    Parameters
    ----------
    header : bytes
        The full header bytes.
    records : list[tuple[str, bool, int]]
        `(text, is_hex, end)` triples from `_scan_v2_strings`.

    Returns
    -------
    str
        The acquisition timestamp as an ISO 8601 UTC datetime (e.g.
        `2026-07-14T05:00:00+00:00`), or an empty string if it cannot be located.
    """
    plain_ends = [end for text, is_hex, end in records if not is_hex]
    start = plain_ends[4] if len(plain_ends) >= 5 else 0
    for offset in range(start, len(header) - 7):
        value = int.from_bytes(header[offset : offset + 8], "little")
        if 1_420_000_000 <= value <= 2_050_000_000:
            return datetime.fromtimestamp(value, UTC).isoformat()
    return ""


def _scan_v2_provenance(records: list[tuple[str, bool, int]]) -> dict[str, str]:
    """Map decoded header strings to v1-style provenance fields, best-effort.

    The provenance strings appear in a stable order:

    `sequence, project, subject, session, species, <type>, scan, <unknown>, <unknown>,
    experimenter, serial (hex), hardware (hex)`

    The two trailing hex-decoded strings (serial, hardware) are used as structural
    anchors; the leading fields are mapped positionally. This is heuristic and matches
    a small number of example files — a field that is empty or absent in another file
    would shift the positional mapping, so the raw strings are always kept alongside the
    mapped fields (see `_scan_v2_attrs`).

    Parameters
    ----------
    records : list[tuple[str, bool, int]]
        `(text, is_hex, end)` triples from `_scan_v2_strings`.

    Returns
    -------
    dict[str, str]
        Provenance fields (`iconeus_subject`, `iconeus_session`, ...) that could be
        recovered; missing fields are simply absent.
    """
    provenance: dict[str, str] = {}

    hex_positions = [i for i, (_, is_hex, _) in enumerate(records) if is_hex]
    if len(hex_positions) >= 2:
        provenance["device_serial_number"] = records[hex_positions[-2]][0]
        provenance["iconeus_hardware"] = records[hex_positions[-1]][0]
        # The experimenter is the last plain string before the trailing hex pair.
        plain_before = [
            t for t, is_hex, _ in records[: hex_positions[-2]] if not is_hex
        ]
        if plain_before:
            provenance["iconeus_experimenter"] = plain_before[-1]

    plain = [text for text, is_hex, _ in records if not is_hex]
    leading_fields = (
        "iconeus_sequence",
        "iconeus_project",
        "iconeus_subject",
        "iconeus_session",
        "iconeus_species",
    )
    for field, value in zip(leading_fields, plain):
        provenance[field] = value
    if len(plain) > 6:
        provenance["iconeus_scan"] = plain[6]

    return provenance


def _load_scan_v2(
    path: Path,
    chunks: int | tuple[int, ...] | str | None,
) -> xr.DataArray:
    """Load a binary Iconeus SCAN v2 file as a lazy DataArray (experimental).

    The v2 format is a flat binary file: a variable-length header followed by a
    little-endian `float64` power-Doppler payload. Field offsets were reverse-engineered
    from example files; see the module docstring and `load_scan` Notes for caveats.

    The payload is wrapped in a NumPy memmap (never fully read here) and exposed lazily
    through Dask, mirroring the laziness of the v1 loader.

    Dimension handling follows the v1 convention. The header stores Iconeus-ordered
    dimensions `(size_x=lateral, size_y=elevation, size_z=depth, n_time, npose,
    nblock_repeat)`; the payload is C-ordered as
    `(n_time, npose, nblock_repeat, size_z, size_y, size_x)`. Output dims:

    - single-pose → `(time, z, y, x)`.
    - multi-pose → `(time, pose, z, y, x)`.

    where ConfUSIus `z` is elevation (`size_y`) and `y` is depth (`size_z`), so those two
    payload axes are swapped. `nblock_repeat > 1` is folded into `time`, as in the v1
    `4Dscan` loader.

    Parameters
    ----------
    path : pathlib.Path
        Path to the v2 SCAN file, already validated to start with `SCAN_V2_MAGIC`.
    chunks : int or tuple[int, ...] or str or None
        Dask chunk specification passed to `dask.array.from_array`.

    Returns
    -------
    xarray.DataArray
        Lazy DataArray with dims `(time, z, y, x)` or `(time, pose, z, y, x)` and
        millimetre spatial coordinates (depth origin from the header when found;
        lateral/elevation centred on zero — see `load_scan` Notes).

    Raises
    ------
    ValueError
        If the header is truncated, reports implausible dimensions, or if the reported
        payload size does not match the product of the dimensions.
    """
    with path.open("rb") as f:
        total_header_bytes = int.from_bytes(
            f.read(_SCAN_V2_OFFSETS["total_header_bytes"] + 8)[-8:], "little"
        )
        f.seek(0)
        header = f.read(total_header_bytes)

    meta = _read_scan_v2_header(header)
    size_x = meta["size_x"]
    size_y = meta["size_y"]
    size_z = meta["size_z"]
    n_time = meta["n_time"]
    npose = meta["npose"]
    nblock_repeat = meta["nblock_repeat"]

    n_elements = size_x * size_y * size_z * n_time * npose * nblock_repeat
    if meta["payload_bytes"] != n_elements * 8:
        raise ValueError(
            f"SCAN v2 payload size ({meta['payload_bytes']} bytes) does not match the "
            f"product of the header dimensions ({n_elements} float64 = "
            f"{n_elements * 8} bytes). The file may be corrupt or use an unsupported "
            "layout."
        )

    meta["depth_start_mm"] = _scan_v2_depth_start(
        header, size_z, meta["z_voxel_m"] * 1e3
    )

    # The payload is memory-mapped (not read here); Dask keeps every downstream reshape
    # and transpose lazy, so the array is only materialised on compute.
    memmap = np.memmap(
        path,
        dtype="<f8",
        mode="r",
        offset=total_header_bytes,
        shape=(n_time, npose, nblock_repeat, size_z, size_y, size_x),
    )
    raw_lazy = da.from_array(memmap, chunks=chunks)

    n_time_total = n_time * nblock_repeat
    if nblock_repeat == 1:
        sq = da.squeeze(raw_lazy, axis=2)
    else:
        # Fold nblock_repeat into time: (n_time, nblock, npose, z, y, x) -> (T, npose, ...).
        transposed = da.transpose(raw_lazy, [0, 2, 1, 3, 4, 5])
        sq = transposed.reshape(n_time_total, npose, size_z, size_y, size_x)

    # Swap depth (size_z) and elevation (size_y): payload order is (depth, elevation),
    # ConfUSIus order is (z=elevation, y=depth).
    data_lazy = da.transpose(sq, [0, 1, 3, 2, 4])

    if npose == 1:
        data_lazy = da.squeeze(data_lazy, axis=1)
        dims = ["time", "z", "y", "x"]
    else:
        dims = ["time", "pose", "z", "y", "x"]

    coords = _scan_v2_coords(meta, n_time_total, npose)
    attrs = _scan_v2_attrs(header, npose, n_time_total)
    attrs.update(_scan_v2_acquisition(header, n_time, meta["depth_start_mm"]))

    data_array = xr.DataArray(data_lazy, dims=dims, coords=coords, attrs=attrs)
    data_array.name = attrs.get("iconeus_scan") or path.stem
    return data_array


def _scan_v2_depth_start(header: bytes, size_z: int, dz_mm: float) -> float:
    """Recover the depth-axis origin (mm) from the SCAN v2 header, best-effort.

    The header stores the imaging depth range as an adjacent `(start, end)` pair of
    `float64` values in millimetres (grouped with the probe geometry). Its absolute
    offset shifts between files because it follows variable-length fields, so instead of
    seeking a fixed offset this scans the header for the first adjacent `float64` pair
    `(a, b)` whose extent `b - a` matches the depth span implied by the voxel count and
    spacing. That span is distinctive enough (metres apart from the plane-wave angles,
    time steps, and TGC gains) to identify the pair unambiguously.

    Parameters
    ----------
    header : bytes
        The full header bytes.
    size_z : int
        Number of depth voxels.
    dz_mm : float
        Depth voxel spacing in millimetres.

    Returns
    -------
    float
        The depth-axis origin in millimetres, or `0.0` if no matching pair is found
        (in which case the depth axis is left relative, starting at zero).
    """
    if size_z < 2 or dz_mm <= 0:
        return 0.0

    expected_span = (size_z - 1) * dz_mm
    tolerance = max(0.1, dz_mm)
    for offset in range(0, len(header) - 16):
        start = struct.unpack_from("<d", header, offset)[0]
        end = struct.unpack_from("<d", header, offset + 8)[0]
        if 0.0 < start < end < 100.0 and abs((end - start) - expected_span) < tolerance:
            return float(start)
    return 0.0


def _scan_v2_acquisition(
    header: bytes, n_time: int, depth_start_mm: float
) -> dict[str, Any]:
    """Parse the acquisition-settings block of a SCAN v2 header, best-effort.

    The probe- and sequence-related fields sit in a structured block whose position is
    computable from `n_time` (the preceding time and frame-index arrays have `n_time`
    elements each). The probe-name string is variable-length, so the fields after it are
    located relative to its end.

    Because the layout is inferred from a small number of example files, the walk is
    **anchor-validated**: the depth range stored right after the probe name must match
    the value found independently by `_scan_v2_depth_start`. If it does not, the block is
    assumed misaligned and nothing is returned, so a mislaid offset never emits wrong
    metadata. The two SVD/power-Doppler filter fields live at fixed offsets and are read
    unconditionally.

    Fields map to fUSI-BIDS as follows (values kept in native header units): `probe_model`
    → `ProbeModel`; `probe_center_frequency` (MHz) → `ProbeCenterFrequency`; `probe_pitch`
    (mm) → `ProbePitch`; `probe_focal_depth` (mm) → `ProbeFocalDepth`; `imaging_depth`
    (mm start/end) → `Depth`; `transmit_frequency` (MHz) → `UltrasoundTransmitFrequency`;
    `pulse_repetition_frequency` (Hz) → `UltrasoundPulseRepetitionFrequency`;
    `plane_wave_angles` (deg) → `PlaneWaveAngles`. `svd_low_cutoff` is the low cutoff of
    the SVD clutter filter (see `confusius.iq.clutter_filter_svd_from_indices`) and
    `power_doppler_integration_window` relates to `PowerDopplerIntegrationDuration`.

    Parameters
    ----------
    header : bytes
        The full header bytes.
    n_time : int
        Number of stored time points (before folding `nblock_repeat`).
    depth_start_mm : float
        Depth origin found by `_scan_v2_depth_start`, used as the alignment anchor.

    Returns
    -------
    dict
        Acquisition attributes that could be recovered; fields whose block could not be
        validated are simply absent.
    """
    o = _SCAN_V2_OFFSETS

    def u16(offset: int) -> int:
        return int.from_bytes(header[offset : offset + 2], "little")

    def u32(offset: int) -> int:
        return int.from_bytes(header[offset : offset + 4], "little")

    def f64(offset: int) -> float:
        return float(struct.unpack_from("<d", header, offset)[0])

    # Filter fields at fixed absolute offsets, always available.
    result: dict[str, Any] = {
        "svd_low_cutoff": u32(o["svd_clutter_cutoff"]),
        "power_doppler_integration_window": u32(o["power_doppler_integration_window"]),
    }

    # Structured block: after time_coords (n_time f64) and frame indices (n_time u32)
    # come a 64-byte orientation block, then the frequency block, a 16-byte gap, and the
    # probe-name string.
    freq_off = o["time_coords"] + 12 * n_time + 64
    name_off = freq_off + 48
    if name_off + 2 > len(header):
        return result

    name_len = u16(name_off)
    name_end = name_off + 2 + name_len
    if not (1 <= name_len <= 64) or name_end + 52 > len(header):
        return result
    name_bytes = header[name_off + 2 : name_end]
    if not all(32 <= c < 127 for c in name_bytes):
        return result

    # Anchor check: the depth range starts right after the probe name and must agree
    # with the independently located depth origin.
    depth_start = f64(name_end)
    if not depth_start_mm or abs(depth_start - depth_start_mm) > 0.5:
        return result

    result["probe_model"] = name_bytes.decode("ascii")
    result["probe_center_frequency"] = f64(freq_off)
    result["probe_pitch"] = f64(freq_off + 8)
    result["probe_focal_depth"] = f64(freq_off + 24)
    result["imaging_depth"] = (depth_start, f64(name_end + 8))
    result["transmit_frequency"] = f64(name_end + 16)
    result["pulse_repetition_frequency"] = f64(name_end + 24)

    n_angles = u32(name_end + 48)
    if 1 <= n_angles <= 512 and name_end + 52 + 8 * n_angles <= len(header):
        angles = np.frombuffer(
            header, dtype="<f8", count=n_angles, offset=name_end + 52
        )
        result["plane_wave_angles"] = angles.tolist()

    return result


def _scan_v2_coords(
    meta: dict[str, Any], n_time_total: int, npose: int
) -> dict[str, xr.DataArray]:
    """Build coordinate DataArrays for a SCAN v2 volume.

    Spatial coordinates use the header voxel spacings (converted to millimetres). The
    v2 header does not encode a lateral/elevation origin, so those axes (`x`, `z`) are
    centred on zero. The depth (`y`) origin is recovered from the header when possible
    (see `_scan_v2_depth_start`) and otherwise defaults to zero. Spacing is exact;
    lateral/elevation absolute position is not (see `load_scan` Notes).

    Parameters
    ----------
    meta : dict
        Parsed header fields from `_read_scan_v2_header`.
    n_time_total : int
        Number of time points after folding `nblock_repeat` into time.
    npose : int
        Number of robot positions.

    Returns
    -------
    dict[str, xarray.DataArray]
        Coordinate DataArrays keyed by `"time"`, `"x"`, `"y"`, `"z"`, and `"pose"` when
        `npose > 1`.
    """
    # Iconeus voxel axes map to ConfUSIus as: x_voxel=lateral->x, y_voxel=elevation->z,
    # z_voxel=depth->y.
    dx_mm = meta["x_voxel_m"] * 1e3
    dz_mm = meta["y_voxel_m"] * 1e3
    dy_mm = meta["z_voxel_m"] * 1e3

    size_x = meta["size_x"]
    size_y = meta["size_y"]
    size_z = meta["size_z"]

    x_vals = (np.arange(size_x) - (size_x - 1) / 2) * dx_mm
    z_vals = (np.arange(size_y) - (size_y - 1) / 2) * dz_mm
    y_vals = meta.get("depth_start_mm", 0.0) + np.arange(size_z) * dy_mm

    time_vals = meta["time_coords"]
    if time_vals.size != n_time_total:
        # nblock_repeat > 1 (unobserved): the header only stores n_time timestamps, so
        # fall back to a regular grid spaced by dt. Keep it end-referenced (first volume
        # ends at dt) to stay consistent with volume_acquisition_reference below.
        time_vals = meta["dt"] * (np.arange(n_time_total) + 1)

    time_attrs: dict[str, Any] = {
        "units": "s",
        "volume_acquisition_reference": "end",
        "volume_acquisition_duration": float(time_vals.min())
        if time_vals.size
        else 0.0,
    }

    coords: dict[str, xr.DataArray] = {
        "time": xr.DataArray(time_vals, dims=["time"], attrs=time_attrs),
        "x": xr.DataArray(x_vals, dims=["x"], attrs={"units": "mm", "voxdim": dx_mm}),
        "z": xr.DataArray(z_vals, dims=["z"], attrs={"units": "mm", "voxdim": dz_mm}),
        "y": xr.DataArray(y_vals, dims=["y"], attrs={"units": "mm", "voxdim": dy_mm}),
    }
    if npose > 1:
        coords["pose"] = xr.DataArray(np.arange(npose), dims=["pose"])
    return coords


def _scan_v2_attrs(header: bytes, npose: int, n_time_total: int) -> dict[str, Any]:
    """Assemble provenance attributes for a SCAN v2 volume.

    The v2 header carries no `probeToLab`-equivalent affine, so `affines` is left empty.
    Header strings are mapped to v1-style provenance fields (`iconeus_subject`,
    `iconeus_session`, ...) on a best-effort basis (see `_scan_v2_provenance`); the raw
    decoded strings are always kept in `iconeus_header_strings` so users can re-map them
    if the heuristic mislabels a field.

    Parameters
    ----------
    header : bytes
        The full header bytes.
    npose : int
        Number of robot positions.
    n_time_total : int
        Number of time points after folding `nblock_repeat` into time.

    Returns
    -------
    dict
        Attributes for the DataArray: `affines` (empty), `iconeus_scan_format`,
        `iconeus_scan_mode`, `iconeus_datetime`, the mapped provenance fields, and the raw
        `iconeus_header_strings`.
    """
    if npose > 1:
        scan_mode = "4Dscan" if n_time_total > 1 else "3Dscan"
    else:
        scan_mode = "2Dscan"

    records = _scan_v2_strings(header)
    attrs: dict[str, Any] = {
        # No physical_to_lab affine has been located in the v2 header yet.
        "affines": {},
        "iconeus_scan_format": "v2",
        "iconeus_scan_mode": scan_mode,
        "iconeus_datetime": _scan_v2_datetime(header, records),
        "iconeus_header_strings": [text for text, _, _ in records],
    }
    attrs.update(_scan_v2_provenance(records))
    return attrs
