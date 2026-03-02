# API Reference

Complete reference documentation for the ConfUSIus API. Please refer to the [user
guide](../user-guide/index.md) for more detailed explanations and usage examples.

## Core Modules

<div class="grid cards" markdown>

- **[:lucide-file-input: Input/Output](io.md)**

    ---

    Loading and saving fUSI data in various formats, including those from AUTC and
    EchoFrame systems, as well as general-purpose formats like Zarr and NIfTI.

- **[:lucide-check: Data Validation](validation.md)**

    ---

    Validation of fUSI data compatibility with expected formats in ConfUSIus.

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

- **[:lucide-funnel: Signal Extraction](extract.md)**

    ---

    Extract and reconstruct signals from fUSI data using spatial masks.

- **[:lucide-brush-cleaning: Signal Processing](signal.md)**

    ---

    Signal processing tools for fUSI time series including denoising and motion artifact
    correction.

- **[:lucide-brackets: Xarray Integration](xarray.md)**

    ---

    Xarray integration providing DataArray accessors for fUSI data.

</div>
