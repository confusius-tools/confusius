# Roadmap

Alpha release checklist for ConfUSIus.

## ✅ Data I/O

- [x] [`io` module](api/io.md): load and save fUSI data in common formats.
    - [x] AUTC beamformed IQ.
    - [x] EchoFrame beamformed IQ.
    - [x] NIfTI.
    - [x] Zarr.
    - [x] Iconeus SCAN (HDF5; see <https://github.com/XunMa1214/fUS-toolbox>).

## ✅ IQ Processing

- [x] [`iq` module](api/iq.md): convert raw beamformed IQ data into functional images.
    - [x] [`process_iq_blocks`][confusius.iq.process_iq_blocks]: apply a processing pipeline over sliding windows of IQ frames.
    - [x] [`process_iq_to_power_doppler`][confusius.iq.process_iq_to_power_doppler]: compute power Doppler volumes from IQ blocks.
    - [x] [`process_iq_to_axial_velocity`][confusius.iq.process_iq_to_axial_velocity]: estimate axial blood velocity from IQ blocks.

## ✅ Signal Extraction

- [x] [`extract` module](api/extract.md): extract voxel time series from spatial imaging data.
    - [x] [`with_mask`][confusius.extract.extract_with_mask]: flatten spatial dimensions into a voxel array using a boolean mask.
    - [x] [`with_labels`][confusius.extract.extract_with_labels]: extract region-averaged signals using an integer label map.
    - [x] [`unmask`][confusius.extract.unmask]: reconstruct a full spatial volume from a flat voxel array.

## ✅ Signal Preprocessing

- [x] [`signal` module](api/signal.md): denoise and preprocess voxel time series.
    - [x] [`censor_samples`][confusius.signal.censor_samples] /
      [`interpolate_samples`][confusius.signal.interpolate_samples]: mark or interpolate corrupted samples (e.g. motion outliers).
    - [x] [`regress_confounds`][confusius.signal.regress_confounds]: remove nuisance signals via least-squares regression.
    - [x] [`filter_butterworth`][confusius.signal.filter_butterworth]: apply a low-pass, high-pass, or band-pass Butterworth filter.
    - [x] [`standardize`][confusius.signal.standardize]: scale signals to unit variance or percent signal change.
    - [x] [`detrend`][confusius.signal.detrend]: remove linear or polynomial trends from time series.

## ✅ Visualization

- [x] [`plotting` module](api/plotting.md): visualize fUSI volumes and time series.
    - [x] [`plot_napari`][confusius.plotting.plot_napari]: interactive 3D/4D viewer with physical-space scaling.
    - [x] [`plot_carpet`][confusius.plotting.plot_carpet]: carpet plot (voxel × time raster) for quality control.
    - [x] [`plot_volume`][confusius.plotting.plot_volume]: display 2D slices of a volume
          using `matplotlib.pyplot.pcolormesh`.
    - [x] [`plot_contours`][confusius.plotting.plot_contours]: overlay region contours
          on imaging slices.

## ✅ Registration

- [x] [`registration` module](api/registration.md): align fUSI volumes to a reference or template.
    - [x] [`register_volumewise`][confusius.registration.register_volumewise]: register each frame in a 4D series to a reference volume.
    - [x] [`compute_framewise_displacement`][confusius.registration.compute_framewise_displacement]: quantify frame-to-frame motion as a scalar displacement.
    - [x] [`register_volume`][confusius.registration.register_volume]: rigid/affine/deformable registration of a single 3D volume.
    - [x] [`resample_volume`][confusius.registration.resample_volume]: resample a volume to a target affine and shape.

## ✅ Atlas

- [x] [`atlas` module](api/atlas.md): fetch and resample standard brain atlases.
    - [x] [BrainGlobe Atlas
      API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html) wrapper
      for loading brain parcellations, returned as Xarray Datasets.

## ✅ Functional Connectivity

- [x] [`connectivity` module](api/connectivity.md): compute functional connectivity from
  preprocessed fUSI time series (pairwise correlations, network metrics).
    - [x] [`SeedBasedMaps`][confusius.connectivity.SeedBasedMaps]: compute voxelwise
      correlation maps for a given seed region.
    - [x] [`ConnectivityMatrix`][confusius.connectivity.ConnectivityMatrix]: compute
      pairwise connectivity matrices between regions of interest.

## 🚧 General Linear Models
- [ ] `glm` module: general linear model for stimulus-evoked and task-based fUSI
  analysis.
