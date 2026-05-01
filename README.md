[![PyPI version](https://img.shields.io/pypi/v/confusius)](https://pypi.org/project/confusius/)
[![Python versions](https://img.shields.io/pypi/pyversions/confusius)](https://pypi.org/project/confusius/)
[![DOI](https://zenodo.org/badge/1155356116.svg)](https://doi.org/10.5281/zenodo.18611124)
[![codecov](https://codecov.io/gh/confusius-tools/confusius/graph/badge.svg?token=TL5AIRNSHS)](https://codecov.io/gh/confusius-tools/confusius)

# ConfUSIus <img src="docs/images/confusius-logo.svg" width="200" title="ConfUSIus" alt="ConfUSIus" align="right">

> [!NOTE]
> **Pre-Alpha Status** — ConfUSIus is in pre-alpha and under active development—the API
> may change between releases and some features may be incomplete. We are happy to help
> you get started: join our [weekly drop-in hours](https://confusius.tools/latest/user-guide/getting-started/#getting-help)
> on Discord or open an [issue on GitHub](https://github.com/confusius-tools/confusius/issues)
> for questions and feature requests.

ConfUSIus is a Python package and napari plugin for handling, visualization,
preprocessing, and statistical analysis of functional ultrasound imaging (fUSI) data.

## Features

> [!NOTE]
> ConfUSIus is not designed as an out-of-the-box, end-to-end fUSI analysis pipeline.
> Because the fUSI field has not yet converged on standard processing workflows,
> ConfUSIus instead aims to provide the fundamental building blocks needed to implement
> any processing workflow described in the fUSI literature, or to design entirely new
> ones. Researchers can combine these blocks to build analysis pipelines suited to
> their experimental needs.

- **I/O Operations**: Load and save fUSI data in various formats (AUTC, EchoFrame,
  Iconeus, NIfTI, Zarr), with automatic fUSI-BIDS sidecars for NIfTI.
- **Beamformed IQ Processing**: Process raw beamformed IQ signals into power Doppler,
  velocity, and other derived metrics.
- **Quality Control**: Compute quality metrics (DVARS, tSNR, CV) to assess data quality
- **Registration**: Motion correction and spatial alignment tools.
- **Brain Atlas Integration**: Map fUSI data to standard brain atlases for region-based
  analysis.
- **Signal Extraction**: Extract signals from regions of interest using spatial masks.
- **Signal Processing**: Denoising, filtering, detrending, and confound regression.
- **Visualization**: Rich plotting utilities for fUSI data exploration.
- **Napari Plugin**: Interactive data loading, live signals inspection, and quality
  control directly in the napari viewer—no scripting required.
- **Xarray Integration**: Seamless integration with Xarray for labeled multi-dimensional
  arrays.

## Installation

### 1. Setup a virtual environment

We recommend that you install ConfUSIus in a virtual environment to avoid dependency
conflicts with other Python packages. Using
[uv](https://docs.astral.sh/uv/guides/install-python/), you may create a new project
folder with a virtual environment as follows:

```bash
uv init new_project
```

If you already have a project folder, you may create a virtual environment as follows:

```bash
uv venv
```

### 2. Install ConfUSIus

ConfUSIus is available on PyPI. Install it using:

```bash
uv add confusius
```

Or with pip:

```bash
pip install confusius
```

To install the latest development version from GitHub:

```bash
uv add git+https://github.com/confusius-tools/confusius.git
```

### 3. Check installation

Check that ConfUSIus is correctly installed by opening a Python interpreter and
importing the package:

```python
import confusius
```

If no error is raised, you have installed ConfUSIus correctly.

## Quick Start

```python
import confusius as cf

# Load fUSI data
data = cf.load("path/to/data.nii.gz")

# Perform motion correction
corrected_data = data.fusi.register.volumewise()

# Visualize with napari
corrected_data.fusi.plot()
```

See the [documentation](https://confusius.tools/latest) for more detailed usage examples
and tutorials.

## Citing ConfUSIus

If you use ConfUSIus in your research, please cite it using the following reference:

> Le Meur-Diebolt, S., & Cybis Pereira, F. (2026). ConfUSIus (v0.0.1-a26). Zenodo.
> https://doi.org/10.5281/zenodo.18611124

Or in BibTeX format:

```bibtex
@software{confusius,
  author    = {Le Meur-Diebolt, Samuel and Cybis Pereira, Felipe},
  title     = {ConfUSIus},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.0.1-a26},
  doi       = {10.5281/zenodo.18611124},
  url       = {https://doi.org/10.5281/zenodo.18611124}
}
```
