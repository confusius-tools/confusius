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

- **Named dimensions**: Access data using meaningful names (e.g., `i`, `j`, `k`, `pose`,
  `time`) instead of remembering axis indices.
- **Coordinates**: Associate time and world coordinates with the native dimensions
  (e.g., `x`, `y`, and `z` in millimeters).
- **Metadata storage**: Keep acquisition parameters, units, and other metadata alongside
  your data.
- **Unified API**: Use the same operations regardless of the underlying storage format.

Every loader in this guide returns **VoxelData arrays**: DataArrays with trailing
`k`/`j`/`i` voxel dimensions backed by a `VoxelToWorldIndex`, which derives world
coordinates `z`/`y`/`x` (in millimeters) from voxel-to-world affine transformations.
World coordinates are never stored directly—they're always this index's output. See
[Spatial Conventions](spatial-conventions.md) for details on the coordinate model.

### Xarray-Compatible Formats

Xarray can read and write data from multiple storage formats, including:

- **[Zarr](https://zarr.dev/)**: Chunked, compressed, cloud-native format (recommended for large datasets).
- **[HDF5](https://www.hdfgroup.org/solutions/hdf5/)**: Hierarchical format widely used in research.
- **[netCDF](https://www.unidata.ucar.edu/software/netcdf)**: Self-describing format
  common in scientific computing (e.g., basis of the
  [MINC](https://mcin.ca/technology/minc/) 1.0 format).

Additionally, ConfUSIus provides utilities to read and write
[**NIfTI**](https://nifti.nimh.nih.gov/) files (the standard neuroimaging format for
BIDS) as VoxelData arrays, automatically reading and writing matching fUSI-BIDS JSON
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
    - **Regular recordings**: Power Doppler, velocity, and other processed signals.

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
    signals are typically further processed to extract derived signals such as power
    Doppler or velocity. Beamformed IQ datasets are typically very large (10s to 100s of
    GB per acquisition session), and stored in system-specific layouts depending on the
    acquisition system (e.g., Iconeus, AUTC, EchoFrame).

=== "Power Doppler, velocity, etc."

    Processed data such as power Doppler, velocity, or other signals derived from
    beamformed IQ data. These datasets are generally much smaller than the original IQ
    data (often 10-100x smaller) and are frequently stored in standardized formats like
    NIfTI for interoperability and BIDS compliance. 

    !!! tip "Zarr for large recordings"
        Large-scale fUSI recordings (e.g., using matrix probes) can also benefit from
        storage in Zarr for efficient processing.

## Loading Data

The universal [`confusius.load`][confusius.load] function allows loading VoxelData
arrays from multiple formats, including NIfTI, Zarr, EchoFrame DAT, and Iconeus
SCAN files. It automatically detects the file format and dispatches to the
appropriate loader ([`load_nifti`][confusius.io.load_nifti],
[`load_echoframe_dat`][confusius.io.load_echoframe_dat],
[`load_scan`][confusius.io.load_scan], etc.).

All ConfUSIus loaders return **lazy** DataArrays backed by Dask—data stays on disk until
an operation requires it.

!!! tip "Load into memory when it fits"
    Lazy loading is essential for datasets larger than available RAM, but it introduces
    Dask scheduling overhead on every operation and can make things difficult if your
    data chunking layout doesn't fit your operation. If your data fits comfortably in
    memory (leaving enough headroom for intermediate results), load it eagerly with
    [`.compute()`][xarray.DataArray.compute] for better performance:

    ```python
    da = cf.load("sub-01_task-awake_pwd.nii.gz").compute()
    ```

### Loading Zarr Files

Xarray treats Zarr stores as multi-variable datasets. ConfUSIus'
[`load`][confusius.load] returns the first variable as a DataArray by default. If the
Zarr store contains multiple variables, you can specify which one to load with the
`variable` argument.

```pycon
>>> import confusius as cf
>>>
>>> # Load beamformed IQ data (returns the first variable as a DataArray by default).
>>> iq_data = cf.load("sub-01_task-awake_iq.zarr")
>>>
>>> # Or specify the variable name if there are multiple variables in the Zarr store.
>>> iq_data = cf.load("sub-01_task-awake_iq.zarr", variable="iq")
>>>
>>> iq_data
<xarray.DataArray 'iq' (time: 1168500, k: 1, j: 118, i: 52)> Size: 57GB
dask.array<open_dataset-iq, shape=(1168500, 1, 118, 52), dtype=complex64, chunksize=(300, 1, 118, 52), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) float64 9MB 5.551 5.553 5.555 ... 2.355e+03 2.355e+03
  * k        (k) float64 8B 0.0
  * j        (j) float64 944B 0.0 1.0 2.0 3.0 ... 114.0 115.0 116.0 117.0
  * i        (i) float64 416B 0.0 1.0 2.0 3.0 ... 48.0 49.0 50.0 51.0
  * z        (k, j, i) float64 ... 0.0 0.0 0.0 ...
  * y        (k, j, i) float64 ... 4.656 4.705 4.753 ...
  * x        (k, j, i) float64 ... -2.671 -2.57 -2.469 ...
Indexes:
  ┌ z        VoxelToWorldIndex
  │ y
  └ x
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

!!! question "Loading a full Dataset"
    [`confusius.load`][confusius.load] always returns a single DataArray. To load all
    variables in a Zarr store as a Dataset, use [`xarray.open_zarr`][xarray.open_zarr]
    directly:

    ```python
    import xarray as xr

    ds = xr.open_zarr("sub-01_task-awake_iq.zarr")
    ```

### Loading EchoFrame DAT Files

For one-time processing, load EchoFrame DAT files directly with
[`confusius.load`][confusius.load]:

```python
import confusius as cf

# Uses ScanParameters.mat next to the DAT file by default.
iq = cf.load("path/to/data.dat")

# Or pass the sidecar path explicitly.
iq = cf.load("path/to/data.dat", meta_path="path/to/metadata.mat")
```

When you expect to process the same recording repeatedly, [convert it to
Zarr](#converting-beamformed-iq-data-to-zarr) instead for better performance.

### Loading Iconeus SCAN Files

Iconeus SCAN files come in two flavors, depending on your version of IcoScan: IcoScan
versions before 2.0 output SCAN files using the HDF5-based SCAN v1 format, while
versions 2.0 and after use the newer binary SCAN v2 format.
[`confusius.load`][confusius.load] automatically detects the format and dispatches to
the appropriate loader.

#### SCAN v1 (HDF5 format)

```pycon
>>> import confusius as cf
>>> da = cf.load("sub-01_task-awake_pwd.source.scan")
>>> da
<xarray.DataArray 'sub-01_task-awake_pwd' (time: 500, pose: 4, k: 4, j: 92, i: 103)> Size: 606MB
dask.array<transpose, shape=(500, 4, 4, 92, 103), dtype=float64, chunksize=(110, 4, 4, 92, 103), chunktype=numpy.ndarray>
Coordinates:
    time     (time, pose) float64 16kB 0.4 2.2 1.0 ... 1.199e+03 1.199e+03
  * pose     (pose) int64 32B 0 1 2 3
  * k        (k) int64 32B 0 1 2 3
  * j        (j) int64 736B 0 1 2 3 4 5 6 7 8 9 ... 83 84 85 86 87 88 89 90 91
  * i        (i) int64 824B 0 1 2 3 4 5 6 7 8 ... 94 95 96 97 98 99 100 101 102
  * z        (pose, k, j, i) float64 1MB -0.19 -0.3 -0.41 ... -11.3 -11.41
  * y        (pose, k, j, i) float64 1MB 20.3 20.3 20.3 ... 29.27 29.27 29.27
  * x        (pose, k, j, i) float64 1MB -3.851 -3.851 -3.851 ... 4.024 4.024
Indexes:
  ┌ z        VoxelToWorldIndex
  │ y
  └ x
Attributes:
    affines:               {}
    device_serial_number:  XXXXXXXX
    software_version:      IcoScan version 1.0.0
    iconeus_scan_mode:     4Dscan
    iconeus_subject:       Mouse01
    iconeus_session:       Session01
    iconeus_scan:          sub-01_task-awake_pwd
    iconeus_project:       Project01
    iconeus_date:          2021-10-22 14:14:12
```

Provenance metadata from the file is stored in `da.attrs`, including
`iconeus_scan_mode`, `iconeus_subject`, `iconeus_session`, `iconeus_scan`,
`iconeus_project`, `iconeus_date`, `software_version`, and
`device_serial_number`.

Note that for recordings consisting of multiple probe poses (e.g., `3Dscan`, `4Dscan`),
the `time` coordinate is **pose-dependent**, giving the acquisition time of each volume
for each pose. This is useful for volume-timing corrections and for aligning multi-pose
acquisitions with external events. 

Additionally, each pose has its own voxel-to-world affine transformation, meaning the
`z`/`y`/`x` world coordinates are **pose-dependent**. The `pose` coordinate indexes the
pose dimension, and the `z`/`y`/`x` coordinates are derived from the corresponding
pose's affine. See [Spatial Conventions](spatial-conventions.md) for details on how
world coordinates are derived from voxel indices and affines.

!!! warning "SCAN files and parallel processing"
    SCAN v1 files are HDF5 files, and h5py datasets **cannot be pickled**. This means
    lazy SCAN DataArrays cannot be passed to functions that use parallel workers (e.g.,
    [`register_volumewise`][confusius.registration.register_volumewise] with `n_jobs !=
    1`). Call `.compute()` to load the data into memory before running any parallel
    operation:

    ```python
    import confusius as cf

    fusi = cf.load("recording.scan").compute()  # materialize first
    fusi = cf.registration.register_volumewise(fusi)
    ```

    Alternatively, use `n_jobs=1` for serial processing (slower but works with lazy
    SCAN data).

#### SCAN v2 (binary format)

!!! warning "Experimental"
    SCAN v2 metadata were reverse-engineered from a few example files kindly provided by
    Iconeus users. The data, timing, voxel spacing, depth origin, `world_to_lab` affine,
    provenance (`iconeus_*` attrs), acquisition datetime, and BIDS-corresponding
    acquisition settings (`probe_model`, `probe_center_frequency`, `transmit_frequency`,
    `pulse_repetition_frequency`, `plane_wave_angles`, `svd_low_cutoff`, …) are
    recovered into `da.attrs`. Lateral/elevation coordinates are centered on zero. This
    and multi-pose layouts are unvalidated and may change.

```pycon
>>> import confusius as cf
>>> da = cf.load("sub-01_task-awake_pwd.source.scan")
>>> da
<xarray.DataArray 'sub-01_task-awake_pwd' (time: 800, k: 1, j: 92, i: 128)> Size: 84MB
dask.array<getitem, shape=(800, 1, 92, 128), dtype=float64, chunksize=(800, 1, 92, 128), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) float64 304B 0.4 0.8 1.2 1.6 2.0 ... 318.8, 319.2, 319.6, 320.0
  * k        (k) int64 8B 0
  * j        (j) int64 736B 0 1 2 3 4 5 6 7 8 9 ... 83 84 85 86 87 88 89 90 91
  * i        (i) int64 1kB 0 1 2 3 4 5 6 7 8 ... 120 121 122 123 124 125 126 127
  * z        (k, j, i) float64 94kB -4.385 -4.275 -4.165 ... 9.365 9.475 9.585
  * y        (k, j, i) float64 94kB 1.0 1.0 1.0 1.0 ... 9.969 9.969 9.969 9.969
  * x        (k, j, i) float64 94kB 0.7 0.7 0.7 0.7 0.7 ... 0.7 0.7 0.7 0.7 0.7
Indexes:
  ┌ z        VoxelToWorldIndex
  │ y
  └ x
Attributes: (12/26)
    affines:                           {}
    iconeus_scan_format:               v2
    iconeus_scan_mode:                 2Dscan
    iconeus_datetime:                  2026-07-14T05:00:00+00:00
    device_serial_number:              XXXXXXX
    iconeus_hardware:                  Hardware_1_0-0123
    ...                                ...
    probe_pitch:                       0.11
    probe_focal_depth:                 8.0
    imaging_depth:                     (1.0, 10.0)
    transmit_frequency:                15.625
    pulse_repetition_frequency:        5500.0
    plane_wave_angles:                 [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2...
```

#### Loading a BPS File

Iconeus' BPS files are HDF5 containers produced by Iconeus' Brain Positioning System.
They contain an affine matrix that maps Iconeus brain coordinates `(x_brain, y_brain,
z_brain, 1)` to Iconeus lab coordinates `(x_lab, y_lab, z_lab, 1)` in meters. The
Iconeus lab frame is a fixed scanner space defined by Iconeus. ConfUSIus re-expresses
this space as **ConfUSIus-ordered** lab space `(z_lab, y_lab, x_lab)` in millimeters:
that is the world space used when loading SCAN files.

The affine matrix in the BPS file can be loaded with
[`confusius.io.load_bps`][confusius.io.load_bps]. It is generally a good idea to store
it in the `.attrs["affines"]` attribute of the DataArray:

```python
import confusius as cf
import numpy as np

da = cf.load("sub-01_task-awake_pwd.source.scan")

brain_to_world = cf.io.load_bps("sub-01_task-awake_pwd.bps")
da.attrs["affines"]["world_to_brain"] = np.linalg.inv(brain_to_world)
```

Passing the BPS file path using the `bps_path` argument when loading a SCAN file offers
a convenient shortcut:

```python
import confusius as cf

da = cf.load(
    "sub-01_task-awake_pwd.source.scan",
    bps_path="sub-01_task-awake_pwd.bps",
)
```

Compose it with brain-side affines (e.g. brain-to-CCFv3 from a brain atlas) to register
fUSI data into atlas space directly. For multi-pose files (`3Dscan`, `4Dscan`) the
affine has shape `(npose, 4, 4)` and is indexed by pose, the same way `world_to_lab` is.

#### Converting SCAN Data to NIfTI

Since [`load`][confusius.load] returns a VoxelData array, you can save it directly to
NIfTI using [`save`][confusius.save] or the Xarray accessor.

For **2Dscan** data, save it directly:

```python
import confusius as cf

da = cf.load("sub-01_task-awake_pwd.scan")
cf.save(da, "sub-01_task-awake_pwd.nii.gz")
# Or equivalently:
da.fusi.save("sub-01_task-awake_pwd.nii.gz")
```

For **3Dscan** and **4Dscan** data, consolidate the poses into a single volume before
saving, or save each pose separately if you want to retain the multi-pose structure. See
the [Multi-Pose Data guide](multi-pose-data.md) for details on working with multi-pose
acquisitions.

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

When loading NIfTI files, ConfUSIus automatically loads a
[fUSI-BIDS](https://bids-specification.readthedocs.io/en/stable/) JSON sidecar file with
the same basename (e.g., `sub-01_task-awake_pwd.json`) if present. Metadata fields are
interpreted using the fUSI-BIDS naming conventions and converted back to the usual
ConfUSIus attribute names on the loaded DataArray. Timing metadata in the sidecar takes
precedence over the NIfTI header when both are available.

```pycon
>>> import confusius as cf
>>>
>>> # Load with automatic fUSI-BIDS sidecar metadata.
>>> da = cf.load("sub-01_task-awake_pwd.nii.gz")
>>> da
<xarray.DataArray 'sub-CR020_ses-20191122_task-checkerboard_acq-slice01_pwd' (
                                                                              time: 1280,
                                                                              k: 1,
                                                                              j: 125,
                                                                              i: 80)> Size: 51MB
dask.array<transpose, shape=(1280, 1, 125, 80), dtype=float32, chunksize=(1280, 1, 125, 80), chunktype=numpy.ndarray>
Coordinates:
  * time     (time) float64 10kB 10.42 10.72 11.02 11.32 ... 393.6 393.9 394.2
  * k        (k) int64 8B 0
  * j        (j) int64 1kB 0 1 2 3 4 5 6 7 8 ... 117 118 119 120 121 122 123 124
  * i        (i) int64 640B 0 1 2 3 4 5 6 7 8 9 ... 71 72 73 74 75 76 77 78 79
  * z        (k, j, i) float64 80kB 0.3 0.3 0.3 0.3 0.3 ... 0.3 0.3 0.3 0.3 0.3
  * y        (k, j, i) float64 80kB 2.996 2.996 2.996 ... 8.988 8.988 8.988
  * x        (k, j, i) float64 80kB -3.95 -3.85 -3.75 -3.65 ... 3.75 3.85 3.95
Indexes:
  ┌ z        VoxelToWorldIndex
  │ y
  └ x
Attributes: (12/24)
    qform_code:                          1
    manufacturer:                        Verasonics
    manufacturers_model_name:            Vantage 128
    software_version:                    Alan Urban Technology & Consulting (...
    probe_manufacturer:                  Vermon
    probe_type:                          linear
    ...                                  ...
    task_description:                    Visual stimulation using a flickerin...
    depth:                               [0.0, 5.991680000000001]
    transmit_frequency:                  15625000.0
    compound_sampling_frequency:         500.0
    plane_wave_angles:                   [-10.0, -7.9, -5.8, -3.6999999999999...
    probe_voltage:                       25.0
```

### Loading Other Formats

For unsupported formats, such as lab-specific MAT-files, load the array with the
appropriate Python tool and use [`create_voxeldata`][confusius.xarray.create_voxeldata]
to attach dimensions, coordinates, and metadata:

```python
import confusius as cf

# Replace this with scipy.io.loadmat, h5py, mat73, or your lab's loader.
raw_power = load_my_mat_file("path/to/power_doppler.mat")  # source array, (i, j, time)

power = cf.create_voxeldata(
    raw_power,
    dims=("i", "j", "time"),  # missing k is added as a singleton voxel dimension
    dt=1 / 2.5,  # 2.5 Hz frame rate
    spacing=(0.4, 0.05, 0.1),  # world spacing in z/y/x order, in mm.
    attrs={"description": "Power Doppler from my system"},
)
```

See the [Create a VoxelData array from a MAT
file](../examples/_built/io/create_voxeldata_from_mat.md) example for a complete
walkthrough, from a real lab-specific MAT file to motion correction and a task GLM.

## Converting Beamformed IQ Data to Zarr

Beamformed IQ exports are often stored as large binary files together with acquisition
metadata. The file structure from AUTC and EchoFrame systems is documented, allowing
ConfUSIus to provide built-in conversion utilities that reorganize these datasets into
Zarr for more efficient processing.

!!! question "Why doesn't ConfUSIus support Iconeus RAW files?"
    ConfUSIus cannot currently read Iconeus beamformed IQ files because the Iconeus RAW
    format is not documented publicly. If you need to process RAW files with ConfUSIus,
    please contact Iconeus to request an export tool or format documentation.

=== "AUTC DATs"

    This format consists of a series of binary `.dat` files (often split into parts),
    where each file contains multiple acquisition blocks.

    To convert a folder of AUTC DAT files to Zarr, use the
    [`convert_autc_dats_to_zarr`][confusius.io.convert_autc_dats_to_zarr] function.
    ConfUSIus does not provide direct lazy loading for AUTC DAT acquisitions because
    they are split across multiple binary files, which makes practical Dask parallel
    processing difficult.

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

    - `iq`: Beamformed IQ data with native dimensions `(time, k, j, i)`.
    - `time`, `k`, `j`, `i`: Native time and voxel-space dimension coordinates.
    - `voxel_to_world` and `world_coord_attrs`: Attributes on `iq` used to restore the
      `VoxelToWorldIndex` and derive `z`/`y`/`x` world coordinates.
    - Metadata attributes (e.g., `transmit_frequency`, `plane_wave_angles`)
      as provided via keyword arguments.

=== "EchoFrame DAT"

    This format consists of a binary `.dat` file containing the beamformed data and a
    `.mat` file containing metadata (sequence parameters).

    To convert EchoFrame data to Zarr, use
    [`convert_echoframe_dat_to_zarr`][confusius.io.convert_echoframe_dat_to_zarr].
    Zarr is useful for long-term storage or repeated processing of the same recording.

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

    - `iq`: Beamformed IQ data with native dimensions `(time, k, j, i)`.
    - `time`, `k`, `j`, `i`: Native time and voxel-space dimension coordinates.
    - `voxel_to_world` and `world_coord_attrs`: Attributes on `iq` used to restore the
      `VoxelToWorldIndex` and derive `z`/`y`/`x` world coordinates.
    - Metadata attributes (e.g., `transmit_frequency`, `plane_wave_angles`)
      as extracted from the metadata file.

### Other Systems

For beamformed IQ data from a system other than AUTC or EchoFrame, load the complex
array with the tool appropriate for your file format, then wrap it as a VoxelData array
with [`create_voxeldata`][confusius.xarray.create_voxeldata]. Put IQ-specific metadata
such as `transmit_frequency` and `beamforming_sound_velocity` in `attrs`:

```python
import confusius as cf
from confusius.validation import validate_voxeldata

raw_iq = load_my_iq_file("path/to/iq.mat")  # complex array, (time, k, j, i)

iq = cf.create_voxeldata(
    raw_iq,
    dims=("time", "k", "j", "i"),
    dt=1 / 500,
    spacing=(0.4, 0.05, 0.1),  # world spacing in z/y/x order, in mm.
    volume_acquisition_duration=1 / 500,
    attrs={
        "transmit_frequency": 15.625e6,
        "beamforming_sound_velocity": 1540.0,
    },
)
validate_voxeldata(iq, require_velocity_attrs=True)
```

See [Processing Beamformed IQ Data](beamformed-iq.md#expected-data-structure) for the
required dimensions, metadata fields, and processing assumptions.

## Saving Data

You can save VoxelData arrays to NIfTI and Zarr using the universal
[`confusius.save`][confusius.save] function or the Xarray accessor:

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

When saving to NIfTI, a
[fUSI-BIDS](https://bids-specification.readthedocs.io/en/stable/) JSON sidecar file will
be automatically created in fUSI-BIDS style. Spatial coordinates and units are encoded
in the NIfTI header itself; the sidecar stores converted metadata fields, custom
attributes, and timing metadata such as `RepetitionTime`, `DelayAfterTrigger`, or
`VolumeTiming`. When possible, `RepetitionTime` is inferred directly from the `time`
coordinate so the sidecar stays consistent with the data being saved. Multi-pose data
cannot be written directly to a single NIfTI file; save it to Zarr, consolidate poses
first, or save one pose per file instead.

If `data_array.attrs["affines"]` contains named world-to-reference affines, you can
choose which ones are written into the NIfTI header:

```python
import confusius as cf

cf.save(
    data_array,
    "output.nii.gz",
    qform="world_to_scanner",
    sform="world_to_template",
)
```

When `qform` and/or `sform` are omitted, [`save`][confusius.save] first looks for
`"world_to_qform"` and `"world_to_sform"` in `attrs["affines"]`. If those keys are not
present, the NIfTI header falls back to the DataArray's coordinate-defining
`voxel_to_world` geometry. Any named affine from `attrs["affines"]` that is actually
written into the NIfTI `qform` or `sform` header is omitted from the `ConfUSIusAffines`
JSON sidecar field so it is not stored twice.

## Format Conversion Reference

Quick reference for converting between formats:

| From | To | Function |
|------|-----|----------|
| AUTC DATs | Zarr | [`confusius.io.convert_autc_dats_to_zarr`][confusius.io.convert_autc_dats_to_zarr] |
| EchoFrame DAT | VoxelData array | [`confusius.io.load_echoframe_dat`][confusius.io.load_echoframe_dat] |
| EchoFrame DAT | Zarr | [`confusius.io.convert_echoframe_dat_to_zarr`][confusius.io.convert_echoframe_dat_to_zarr] |
| Iconeus SCAN | VoxelData array | [`confusius.load`][confusius.load] |
| NIfTI | VoxelData array | [`confusius.load`][confusius.load] |
| Zarr | VoxelData array or Dataset | [`confusius.load`][confusius.load] / [`xarray.open_zarr`][xarray.open_zarr] (Dataset) |
| VoxelData array | NIfTI | [`confusius.save`][confusius.save] / [`.fusi.save`][confusius.xarray.FUSIAccessor.save] |
| VoxelData array | Zarr | [`confusius.save`][confusius.save] / [`.fusi.save`][confusius.xarray.FUSIAccessor.save] / [`.to_zarr`][xarray.DataArray.to_zarr] |
