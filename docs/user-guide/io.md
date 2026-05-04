---
icon: lucide/file-input
---

# Input/Output

## Overview

ConfUSIus is designed to handle large-scale fUSI datasets efficiently. This guide
explains how to load supported formats directly in ConfUSIus, choose appropriate storage
formats for your workflow, and convert beamformed IQ data when needed.

## Working with Xarray

ConfUSIus uses [Xarray](https://docs.xarray.dev/) as its core data structure for
representing multi-dimensional fUSI data. Xarray provides several advantages over raw
NumPy arrays:

- **Named dimensions**: Access data using meaningful names (e.g., `time`, `z`, `y`, `x`)
  instead of remembering axis indices.
- **Coordinates**: Associate physical coordinates with each dimension (e.g., time
  in seconds, depth in millimeters).
- **Metadata storage**: Keep acquisition parameters, units, and other metadata alongside
  your data.
- **Unified API**: Use the same operations regardless of the underlying storage format.

### Xarray-Compatible Formats

Xarray can read and write data from multiple storage formats, including:

- **[Zarr](https://zarr.dev/)**: Chunked, compressed, cloud-native format (recommended for large datasets).
- **[HDF5](https://www.hdfgroup.org/solutions/hdf5/)**: Hierarchical format widely used in research.
- **[netCDF](https://www.unidata.ucar.edu/software/netcdf)**: Self-describing format
  common in scientific computing (e.g., basis of the
  [MINC](https://mcin.ca/technology/minc/) 1.0 format).

Additionally, ConfUSIus provides utilities to read and write
[**NIfTI**](https://nifti.nimh.nih.gov/) files (the standard neuroimaging format for
BIDS) as Xarray DataArrays, automatically reading and writing matching fUSI-BIDS JSON
sidecars when present.

### Recommended Formats for fUSI

ConfUSIus works with all Xarray-compatible formats, but two formats are particularly
well-suited for fUSI workflows:

=== "NIfTI for Sharing and Interoperability"

    [NIfTI](https://nifti.nimh.nih.gov/) is the standard format for:

    - **fUSI-BIDS compliance**: Required for sharing datasets following the BIDS
      specification.
    - **Neuroimaging pipelines**: Compatible with tools like
      [Nilearn](https://nilearn.github.io/),
      [ANTsPy](https://antspy.readthedocs.io/en/stable/), and
      [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/), which have been used in some fUSI
      studies.
    - **Derived acquisitions**: Power Doppler, velocity, and other processed signals.

    Use NIfTI when you need to share data, ensure BIDS compliance, or integrate with
    existing neuroimaging analysis tools.

=== "Zarr for Large-Scale Processing"

    [Zarr](https://zarr.readthedocs.io/) excels at handling massive datasets through:

    - **Out-of-core processing**: Work with datasets larger than memory by loading only
      needed chunks.
    - **Compression**: Reduce storage footprint without sacrificing access speed.
    - **Concurrent chunk access**: Multiple workers can read or write different
      chunks efficiently.
    - **Cloud compatibility**: Store and access data on remote object storage (S3, GCS,
      etc.).

    Use Zarr for beamformed IQ data, large-scale analyses, and cloud-based processing
    workflows. ConfUSIus converts beamformed IQ data to Zarr by default for optimal
    performance with large-scale datasets.

## fUSI Data Types

fUSI workflows involve two main categories of data:

=== "Beamformed IQ Data"

    Complex-valued signals resulting from ultrasound beamforming of RF signals. These
    signals are typically further processed to extract derived signals such as power Doppler
    or velocity. Beamformed IQ datasets are typically very large (10s to 100s of GB per
    acquisition session), and stored in system-specific layouts depending on the
    acquisition system (e.g., Iconeus, AUTC, EchoFrame).

=== "Derived Acquisitions"

    Processed data products such as power Doppler, velocity, or other signals derived from
    beamformed IQ data. These datasets are generally much smaller than the original IQ data
    (often 10-100x smaller) and are frequently stored in standardized formats like NIfTI for 
    interoperability and BIDS compliance. 

    !!! tip "Zarr for large derived datasets"
        Large-scale derived acquisitions (e.g., long power Doppler recordings) can also
        benefit from storage in Zarr for efficient processing.

## Converting Beamformed IQ Data

Beamformed IQ exports are often stored as large binary files together with acquisition
metadata. The file structure from AUTC and EchoFrame systems is documented, allowing
ConfUSIus to provide built-in conversion utilities that reorganize these datasets into
Zarr for efficient processing.

!!! question "Why not Iconeus RAW files?"
    ConfUSIus cannot currently read Iconeus RAW beamformed IQ files because this
    format is not documented publicly. If you need to process RAW files with
    ConfUSIus, please contact Iconeus to request an export tool or format
    documentation.

=== "AUTC DATs"

    This format consists of a series of binary `.dat` files (often split into parts),
    where each file contains multiple acquisition blocks.

    To convert a folder of AUTC DAT files to Zarr, use the
    [`convert_autc_dats_to_zarr`][confusius.io.convert_autc_dats_to_zarr] function.

    ```python
    from confusius.io import convert_autc_dats_to_zarr

    convert_autc_dats_to_zarr(
        dats_root="path/to/data_folder",
        output_path="sub-01_task-awake_iq.zarr",
        # Optional: specify block start times, transmit frequency, axis coordinates, and
        # other metadata via keyword arguments (see API for details).
        block_times=block_times,
        compound_sampling_frequency=500.0,
        transmit_frequency=15.625e6,
        beamforming_sound_velocity=1510.0,
    )
    ```

    This will create a Zarr group containing:

    - `iq`: Beamformed IQ data with dimensions `(time, z, y, x)`.
    - `time`, `z`, `y`, `x`: Coordinate arrays.
    - Voxel sizes (`voxdim`) as per-coordinate attributes on `z`, `y`, and `x`.
    - Metadata attributes (e.g., `transmit_frequency`, `plane_wave_angles`)
      as provided via keyword arguments.

=== "EchoFrame DAT"

    This format consists of a binary `.dat` file containing the beamformed data and a
    `.mat` file containing metadata (sequence parameters).

    To convert EchoFrame data to Zarr, use the
    [`convert_echoframe_dat_to_zarr`][confusius.io.convert_echoframe_dat_to_zarr]
    function.

    ```python
    from confusius.io import convert_echoframe_dat_to_zarr

    convert_echoframe_dat_to_zarr(
        dat_path="path/to/data.dat",
        meta_path="path/to/metadata.mat",
        output_path="sub-01_task-awake_iq.zarr",
        # Optional: specify block start times. Other metadata (e.g., transmit frequency,
        # axis coordinates) will be automatically extracted from the metadata file.
        block_times=block_times,
    )
    ```

    This will create a Zarr group containing:

    - `iq`: Beamformed IQ data with dimensions `(time, z, y, x)`.
    - `time`, `z`, `y`, `x`: Coordinate arrays.
    - Voxel sizes (`voxdim`) as per-coordinate attributes on `z`, `y`, and `x`.
    - Metadata attributes (e.g., `transmit_frequency`, `plane_wave_angles`)
      as extracted from the metadata file.

### Other Systems

If you're working with IQ data from a system other than AUTC or EchoFrame, load it
using your own loader and use [`validate_iq`][confusius.validation.validate_iq] to
ensure it meets ConfUSIus requirements before processing:

```python
import xarray as xr
from confusius.validation import validate_iq

# Load IQ data from an unsupported format (example using your own loading function).
iq = your_loader_function("path/to/iq_data")

# Validate the IQ data structure and required attributes.
try:
    iq_validated = validate_iq(iq)
    print("✓ IQ data is valid and ready for processing")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
```

The [`validate_iq`][confusius.validation.validate_iq] function checks that your data:

- Has the correct dimensions: `(time, z, y, x)`.
- Is complex-valued ([`numpy.complex64`][numpy.complex64] or
  [`numpy.complex128`][numpy.complex128]).
- Contains required attributes:
    - `compound_sampling_frequency`: Effective IQ sampling frequency in Hz.
    - `transmit_frequency`: Ultrasound transmit frequency in Hz.
    - `beamforming_sound_velocity`: Speed of sound in m/s.

!!! question "Finding attribute values"
    If you're unsure about the correct values for these attributes:

    - `compound_sampling_frequency`: Check your acquisition software settings. The
      compound sampling frequency is generally around 500-1000 Hz for fUSI acquisitions,
      but can vary based on the system and settings used.
    - `transmit_frequency`: Found in your probe specifications or acquisition
      settings. Generally around 5-10 MHz for clinical probes, and 12-20 MHz for
      high-frequency probes used in small animal imaging.
    - `beamforming_sound_velocity`: Typically 1540 m/s for brain tissues, but may vary
      with temperature and tissue type.

## Loading Data

All ConfUSIus loaders return **lazy** DataArrays backed by Dask—data stays on disk until
an operation requires it.

!!! tip "Load into memory when it fits"
    Lazy loading is essential for datasets larger than available RAM, but it introduces
    Dask scheduling overhead on every operation. If your data fits comfortably in memory
    (leaving enough headroom for intermediate results), load it eagerly with
    [`.compute()`][xarray.DataArray.compute] for better performance:

    ```python
    da = cf.load("sub-01_task-awake_pwd.nii.gz").compute()
    ```

### Loading Zarr Files

Once your data is in Zarr format, load it with [`confusius.load`][confusius.load]:

```python
import confusius as cf

# Load beamformed IQ data (returns the first variable as a DataArray by default).
iq_data = cf.load("sub-01_task-awake_iq.zarr")

# Or specify the variable name if there are multiple variables in the Zarr store.
iq_data = cf.load("sub-01_task-awake_iq.zarr", variable="iq")

print(iq_data)
```

!!! question "Loading a full Dataset"
    [`confusius.load`][confusius.load] always returns a single DataArray. To load all
    variables in a Zarr store as a Dataset, use [`xarray.open_zarr`][xarray.open_zarr]
    directly:

    ```python
    import xarray as xr

    ds = xr.open_zarr("sub-01_task-awake_iq.zarr")
    ```

This returns an DataArray lazily loaded from the Zarr store, ready for processing:

```text
<xarray.DataArray 'iq' (time: 1168500, z: 1, y: 118, x: 52)> Size: 57GB
dask.array<open_dataset-iq, shape=(1168500, 1, 118, 52), dtype=complex64, chunksize=(300, 1, 118, 52), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) float64 9MB 5.551 5.553 5.555 ... 2.355e+03 2.355e+03
  * z        (z) float64 8B 0.0
  * y        (y) float64 944B 4.656 4.705 4.753 4.802 ... 10.23 10.28 10.33
  * x        (x) float64 416B -2.671 -2.57 -2.469 -2.369 ... 2.268 2.369 2.469
Attributes:
    transmit_frequency:             15625000.0
    probe_number_of_elements:       128
    probe_pitch:                    0.0001
    beamforming_sound_velocity:     1510.0
    plane_wave_angles:              [-10.0, -9.310344696044922, -8.620689392089...
    compound_sampling_frequency:    500.0
    pulse_repetition_frequency:     15000.0
    beamforming_method:             Fourier
```

Notice that the data remains on disk (shown by `dask.array<...>`) until you explicitly
compute operations on it.

### Loading Iconeus SCAN Files

Use [`load_scan`][confusius.io.load_scan] to load Iconeus `.scan` files (HDF5 files
produced by IcoScan) as lazy Xarray DataArrays. Three acquisition modes are supported,
each yielding a DataArray with different dimensions:

| Mode | Dimensions | Description |
|------|------------|-------------|
| `2Dscan` | `(time, z, y, x)` | 2D+t fUSI |
| `3Dscan` | `(pose, z, y, x)` | Multi-pose anatomical volume |
| `4Dscan` | `(time, pose, z, y, x)` | Multi-pose time-series (3D+t fUSI) |

All spatial coordinates are in millimeters; the `time` coordinate is in seconds.

=== "2Dscan"

    ```python
    import confusius as cf

    da = cf.load("sub-01_task-awake_pwd.source.scan")

    print(da.dims)
    # Output: ('time', 'z', 'y', 'x')
    ```

=== "3Dscan"

    ```python
    import confusius as cf

    da = cf.load("sub-01_acq-anat_pwd.source.scan")

    print(da.dims)
    # Output: ('pose', 'z', 'y', 'x')
    ```

    The `pose` dimension indexes each probe position in the multi-pose acquisition.
    Each pose has its own `physical_to_lab` affine (shape `(npose, 4, 4)`) stored in
    `da.attrs["affines"]["physical_to_lab"]`.

=== "4Dscan"

    ```python
    import confusius as cf

    da = cf.load("sub-01_task-awake_pwd.source.scan")

    print(da.dims)
    # Output: ('time', 'pose', 'z', 'y', 'x')
    ```

    In addition to the `time` coordinate (earliest timestamp per block), a
    `pose_time` non-dimension coordinate of shape `(time, pose)` stores the exact
    per-pose acquisition timestamps.

The DataArray is loaded **lazily**: data remains on disk until explicitly computed.
SCAN files stay open while the Dask graph is un-computed, so keep the DataArray in
scope or call [`.compute()`][xarray.DataArray.compute] before discarding it.

!!! warning "SCAN files and parallel processing"
    SCAN files are HDF5 files, and h5py datasets **cannot be pickled**. This means
    lazy SCAN DataArrays cannot be passed to functions that use parallel workers
    (e.g., [`register_volumewise`][confusius.registration.register_volumewise] with
    `n_jobs != 1`). Call `.compute()` to load the data into memory before running any
    parallel operation:

    ```python
    import confusius as cf

    fusi = cf.load("recording.scan").compute()  # materialize first
    fusi = cf.registration.register_volumewise(fusi)
    ```

    Alternatively, use `n_jobs=1` for serial processing (slower but works with lazy
    SCAN data).

Provenance metadata from the file is stored in `da.attrs`: `scan_mode`, `subject`,
`session`, `scan`, `project`, `date`, `neuroscan_version`, and `machine_sn`.

#### Loading a BPS File

Iconeus' BPS files are HDF5 containers produced by Iconeus' Brain Positioning System.
They contain an affine matrix that maps Iconeus brain coordinates `(x_brain, y_brain,
z_brain, 1)` to Iconeus lab coordinates `(x_lab, y_lab, z_lab, 1)` in meters. The
Iconeus lab frame is a fixed scanner frame; `probeToLab` carries any rotation of the
probe within it. ConfUSIus re-expresses this space as **ConfUSIus-ordered** lab space
`(z_lab, y_lab, x_lab)` in millimeters, matching the convention used throughout the
rest of the package.

The affine matrix in the BPS file can be loaded with [`confusius.io.load_bps`][confusius.io.load_bps].

```python
import confusius as cf

bps = cf.io.load_bps("sub-01_task-awake_pwd.bps")
```

Compute `physical_to_brain` (Iconeus brain space from ConfUSIus physical space) from
the SCAN file `physical_to_lab` affine and store it in the DataArray attributes:

```python
import confusius as cf
import numpy as np

da = cf.load("sub-01_task-awake_pwd.source.scan")
bps = cf.io.load_bps("sub-01_task-awake_pwd.bps")

physical_to_lab = da.attrs["affines"]["physical_to_lab"]
physical_to_brain = np.linalg.inv(bps) @ physical_to_lab
da.attrs["affines"]["physical_to_brain"] = physical_to_brain
```

In fact, if you pass the BPS file using the `bps_path` argument when loading a
SCAN file with [`confusius.load`][confusius.load], the `physical_to_brain` affine
will be computed automatically and stored in the resulting DataArray's attributes
`affines` alongside the `physical_to_lab` affines:

```python
import confusius as cf

da = cf.load(
    "sub-01_task-awake_pwd.source.scan",
    bps_path="sub-01_task-awake_pwd.bps",
)

physical_to_brain = da.attrs["affines"]["physical_to_brain"]
```

Compose it with brain-side affines (e.g. brain-to-CCFv3 from a brain atlas) to register
fUSI data into atlas space directly. For multi-pose files (`3Dscan`, `4Dscan`) the
affine has shape `(npose, 4, 4)` and is indexed by pose, the same way `physical_to_lab`
is.

#### Converting SCAN Data to NIfTI

Since [`load_scan`][confusius.io.load_scan] returns a standard Xarray DataArray with
ConfUSIus-compatible dimensions and coordinates, you can save it directly to NIfTI using
[`save_nifti`][confusius.io.save_nifti] or the Xarray accessor.

For **2Dscan** data, save it directly:

```python
import confusius as cf

da = cf.load("sub-01_task-awake_pwd.scan")
cf.save(da, "sub-01_task-awake_pwd.nii.gz")
# Or equivalently:
da.fusi.save("sub-01_task-awake_pwd.nii.gz")
```

For **3Dscan** and **4Dscan** data, consolidate the poses into a single volume before
saving, or save each pose separately if you want to retain the multi-pose structure:

=== "Consolidation"

    ```python
    import confusius as cf

    anat = cf.load("sub-01_acq-anat_pwd.scan")
    volume = cf.multipose.consolidate_poses(anat)
    cf.save(volume, "sub-01_acq-anat_pwd.nii.gz")
    ```

=== "Separate pose files"

    ```python
    import confusius as cf

    anat = cf.load("sub-01_acq-anat_pwd.scan")
    for pose in anat.pose:
        pose_da = anat.sel(pose=pose)
        cf.save(pose_da, f"sub-01_acq-anat_pose-{pose.values}.nii.gz")
    ```

### Loading NIfTI Files

Use [`confusius.load`][confusius.load] to load NIfTI files as lazy Xarray DataArrays:

```python
import confusius as cf

# Load with automatic fUSI-BIDS sidecar metadata.
da = cf.load("sub-01_task-awake_pwd.nii.gz")
print(da.dims)
# Output: ('time', 'z', 'y', 'x')
```

ConfUSIus automatically loads a JSON sidecar file with the same basename (e.g.,
`sub-01_task-awake_pwd.json`) if present. Metadata fields are interpreted using the
fUSI-BIDS naming conventions and converted back to the usual ConfUSIus attribute names
on the loaded DataArray. Timing metadata in the sidecar takes precedence over the NIfTI
header when both are available.

## Saving Data

You can save DataArrays to NIfTI and Zarr using [`confusius.save`][confusius.save] or
the Xarray accessor:

=== "`confusius.save`"

    ```python
    import confusius as cf

    # Save to NIfTI with automatic fUSI-BIDS JSON sidecar creation.
    cf.save(data_array, "output.nii.gz")

    # Save to Zarr.
    cf.save(data_array, "output.zarr")
    ```

=== "Xarray accessor"

    ```python
    import confusius

    # Save to NIfTI with automatic fUSI-BIDS JSON sidecar creation.
    data_array.fusi.save("output.nii.gz")

    # Save to Zarr.
    data_array.fusi.save("output.zarr")
    ```

When saving to NIfTI, a JSON sidecar file will be automatically created in fUSI-BIDS
style. Spatial coordinates and units are encoded in the NIfTI header itself; the
sidecar stores converted metadata fields, custom attributes, and timing metadata such
as `RepetitionTime`, `DelayAfterTrigger`, or `VolumeTiming`. When possible,
`RepetitionTime` is inferred directly from the `time` coordinate so the sidecar stays
consistent with the data being saved.

If `data_array.attrs["affines"]` contains named physical-to-reference affines, you can
choose which ones are written into the NIfTI header:

```python
import confusius as cf

cf.save_nifti(
    data_array,
    "output.nii.gz",
    qform="physical_to_scanner",
    sform="physical_to_template",
)
```

When `qform` and/or `sform` are omitted, [`save_nifti`][confusius.io.save_nifti]
falls back to `"physical_to_qform"` and `"physical_to_sform"` if those keys exist in
`attrs["affines"]`. Any affine actually written into the NIfTI `qform` or `sform`
header is omitted from the `ConfUSIusAffines` JSON sidecar field so it is not stored
twice.

## Format Conversion Reference

Quick reference for converting between formats:

| From | To | Function |
|------|-----|----------|
| AUTC DATs | Zarr | [`confusius.io.convert_autc_dats_to_zarr`][confusius.io.convert_autc_dats_to_zarr] |
| EchoFrame DAT | Zarr | [`confusius.io.convert_echoframe_dat_to_zarr`][confusius.io.convert_echoframe_dat_to_zarr] |
| Iconeus SCAN | Xarray DataArray | [`confusius.load`][confusius.load] |
| NIfTI | Xarray DataArray | [`confusius.load`][confusius.load] |
| Zarr | Xarray DataArray | [`confusius.load`][confusius.load] / [`xarray.open_zarr`][xarray.open_zarr] (Dataset) |
| Xarray DataArray | NIfTI | [`confusius.save`][confusius.save] / [`.fusi.save`][confusius.xarray.FUSIAccessor.save] |
| Xarray DataArray | Zarr | [`confusius.save`][confusius.save] / [`.fusi.save`][confusius.xarray.FUSIAccessor.save] / [`.to_zarr`][xarray.DataArray.to_zarr] |
