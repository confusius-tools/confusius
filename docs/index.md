---
hide:
    - navigation
    - toc
---

# Welcome to ConfUSIus

ConfUSIus is a Python package and napari plugin for handling, visualization,
preprocessing, and statistical analysis of functional ultrasound imaging (fUSI) data.

!!! info "Pre-Alpha Status"
    ConfUSIus is in **pre-alpha** and under active development—the API may change
    between releases and some features may be incomplete. We are happy to help you get
    started: join our [weekly drop-in hours](user-guide/getting-started.md#getting-help)
    on Discord or open an issue on
    [GitHub](https://github.com/confusius-tools/confusius/issues) for questions and feature
    requests.

## Quick Links

<div class="grid cards" markdown>

-   [:lucide-book-open-text:{ .lg .middle } **User Guide**](user-guide/getting-started.md)

    ---

    Learn how to use ConfUSIus for fUSI data analysis.

-   [:lucide-app-window:{ .lg .middle } **Graphical Interface**](gui/overview.md)

    ---

    Explore, inspect, and run QC on your data interactively in napari.

-   [:lucide-users:{ .lg .middle } **Community**](user-guide/getting-started.md#getting-help)

    ---

    Join our weekly drop-in hours on Discord (Thursdays 3-4pm UTC).

</div>

## Features

!!! info "Design Philosophy"
    ConfUSIus is not designed as an out-of-the-box, end-to-end fUSI analysis pipeline.
    Because the fUSI field has not yet converged on standard processing workflows,
    ConfUSIus instead aims to provide the fundamental building blocks needed to implement
    any processing workflow described in the fUSI literature, or to design entirely new
    ones. Researchers can combine these blocks to build analysis pipelines suited to
    their experimental needs.

- **I/O Operations**: Load and save fUSI data in various formats (AUTC, EchoFrame,
  Iconeus, NIfTI, Zarr), with automatic fUSI-BIDS sidecars for NIfTI.
- **Beamformed IQ Processing**: Process raw beamformed IQ signals into power Doppler,
  velocity, and other derived metrics.
- **Quality Control**: Compute quality metrics (DVARS, tSNR, CV) to assess data quality.
- **Registration**: Motion correction and spatial alignment tools.
- **Brain Atlas Integration**: Map fUSI data to standard brain atlases for region-based
  analysis.
- **Signal Extraction**: Extract and reconstruct signals from regions of interest using
  spatial masks.
- **Signal Processing**: Denoising, filtering, detrending, and confound regression for
  hemodynamic signals.
- **Visualization**: Rich plotting utilities for fUSI data exploration.
- **Napari Plugin**: Interactive data loading, live time-series inspection, and quality
  control directly in the napari viewer—no scripting required.
- **Xarray Integration**: Seamless integration with Xarray for labeled multi-dimensional
  arrays.

## Installation

```bash
pip install confusius
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv add confusius
```

See the [Installation Guide](installation.md) for detailed instructions.

## Quick Start

```python
import confusius as cf

# Load fUSI data.
data = cf.load("path/to/data.nii.gz")

# Perform motion correction.
corrected_data = data.fusi.register.volumewise()

# Visualize with napari.
corrected_data.fusi.plot()
```
