# API Reference

Complete reference documentation for the ConfUSIus API. Please refer to the [user
guide](../user-guide/index.md) for more detailed explanations and usage examples.

## Core Modules

<div class="grid cards" markdown>

- **[:lucide-file-input: Input/Output](io.md)**

    ---

    Loading and saving fUSI data in various formats, including those from AUTC and
    EchoFrame systems, as well as general-purpose formats like Zarr and NIfTI.

- **[:lucide-file-badge: fUSI-BIDS](bids.md)**

    ---

    fUSI-BIDS metadata validation, conversion utilities, and coordinate handling for
    BIDS-compliant fUSI data.

- **[:lucide-check: Data Validation](validation.md)**

    ---

    Validation of fUSI data compatibility with expected formats in ConfUSIus.

- **[:lucide-waypoints: Multi-Pose](multipose.md)**

    ---

    Multi-pose consolidation tools for combining fUSI data across acquisition poses.

- **[:lucide-audio-waveform: Beamformed IQ](iq.md)**

    ---

    Processing of beamformed IQ signals, clutter filtering, and computation of derived
    measures such as power Doppler and axial velocity.

- **[:lucide-chart-area: Plotting](plotting.md)**

    ---

    Rich visualization utilities for fUSI data including static plots and interactive
    napari viewers.

- **[:lucide-clipboard-check: Quality Control](qc.md)**

    ---

    Quality control metrics for assessing fUSI data quality.

- **[:lucide-images: Registration](registration.md)**

    ---

    Motion correction and spatial alignment tools for aligning fUSI data across time
    points or with anatomical references.

- **[:lucide-brain: Atlases](atlas.md)**

    ---

    Brain atlas integration for anatomical labeling and region-of-interest extraction.

- **[:lucide-cuboid: Spatial Processing](spatial.md)**

    ---

    Spatial processing tools for fUSI volumetric data, including Gaussian smoothing.

- **[:lucide-funnel: Signal Extraction](extract.md)**

    ---

    Extract and reconstruct signals from fUSI data using spatial masks.

- **[:lucide-brush-cleaning: Signal Processing](signal.md)**

    ---

    Signal processing tools for fUSI time series including denoising and motion artifact
    correction.

- **[:lucide-brain-circuit: Functional Connectivity](connectivity.md)**

    ---

    Functional connectivity analysis for fUSI data.

- **[:lucide-square-function: General Linear Model](glm.md)**

    ---

    Voxel-wise GLM for first- and second-level analysis of fUSI data, including design
    matrix construction and contrast computation.

- **[:lucide-brackets: Xarray Integration](xarray.md)**

    ---

    Xarray integration providing DataArray accessors for fUSI data.

- **[:lucide-database: Datasets](datasets.md)**

    ---

    Fetchers for publicly available fUSI datasets with automatic caching and
    offline support.

</div>
