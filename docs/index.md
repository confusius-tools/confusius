---
hide:
    - navigation
    - toc
---

<!-- Suppress the auto-injected page-title h1 from Zensical's content template.
     Visually hidden (not display:none) so screen readers still see the h1. -->
<h1 class="sr-only">ConfUSIus</h1>

<div class="hero-banner" markdown>

<img src="images/confusius-logo.svg" alt="ConfUSIus" class="hero-logo">

<p class="hero-title">Con<span class="fusi-accent">fUSI</span>us</p>

<p class="hero-tagline">Python package for analysis and visualization of functional ultrasound imaging data.</p>

<div class="hero-badges">
  <a href="https://pypi.org/project/confusius/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/pypi/v/confusius?style=flat-square&color=d93a54&label=PyPI" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/confusius/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/pypi/pyversions/confusius?style=flat-square&color=0099e5&logo=python&logoColor=white" alt="Python versions">
  </a>
  <a href="https://github.com/confusius-tools/confusius/blob/main/LICENSE" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/github/license/confusius-tools/confusius?style=flat-square&color=3ad9a4" alt="License">
  </a>
  <a href="https://discord.gg/mZd87tgmy2" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Discord-join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

</div>

!!! info "Beta Status"
    ConfUSIus is in **beta** and under active development. Core functionality is in place,
    but APIs may still evolve. Join our
    [weekly drop-in hours](user-guide/getting-started.md#getting-help) on Discord
    or open an [issue on GitHub](https://github.com/confusius-tools/confusius/issues)
    for questions and feature requests.

## :lucide-rocket: Features

ConfUSIus provides the fundamental building blocks for fUSI data analysis: not a fixed
pipeline, but composable tools you can assemble into any workflow the literature
describes or you invent.

<div class="grid cards" markdown>

-   [:lucide-hard-drive:{ .lg .middle } **I/O & BIDS**](user-guide/io.md)

    ---

    Load and save AUTC, EchoFrame, Iconeus, NIfTI, and Zarr formats with automatic
    fUSI-BIDS sidecar support.

-   [:lucide-radio:{ .lg .middle } **Beamformed IQ**](user-guide/beamformed-iq.md)

    ---

    Process raw IQ signals into power Doppler, velocity, and other derived metrics.

-   [:lucide-activity:{ .lg .middle } **Quality Control**](user-guide/quality-control.md)

    ---

    Compute DVARS, tSNR, and CV to assess data quality across sessions and subjects.

-   [:lucide-move-3d:{ .lg .middle } **Registration**](user-guide/registration.md)

    ---

    Motion correction and spatial alignment, including support for multi-pose imaging.

-   [:lucide-brain:{ .lg .middle } **Brain Atlases**](user-guide/atlas.md)

    ---

    Map fUSI data to standard brain atlases for region-of-interest and group analysis.

-   [:lucide-filter:{ .lg .middle } **Signal Processing**](user-guide/signal.md)

    ---

    Denoising, filtering, detrending, and confound regression for hemodynamic signals.

-   [:lucide-network:{ .lg .middle } **Functional Connectivity**](user-guide/connectivity.md)

    ---

    Seed-based and matrix-based connectivity measures for resting-state fUSI.

-   [:lucide-bar-chart-2:{ .lg .middle } **General Linear Model**](user-guide/glm.md)

    ---

    Task-based fUSI analysis with HRF convolution and contrast estimation.

-   [:lucide-sigma:{ .lg .middle } **Statistical Inference**](user-guide/stats.md)

    ---

    Multiple-comparison correction and thresholding helpers for p-value maps and
    z-maps.

-   [:lucide-app-window:{ .lg .middle } **Napari Plugin**](gui/overview.md)

    ---

    Interactive data loading, live signal inspection, and QC — no scripting required.

</div>

## :lucide-zap: Quick Start

<div class="quickstart-grid">
<div markdown>

```python
import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022

# Download dataset (cached after the first run, ~30 MB).
bids_root = fetch_nunez_elizalde_2022(
    subjects="CR022", sessions="20201011",
    tasks="spontaneous", acqs="slice03",
)

# Load power Doppler time series.
data = cf.load(
    bids_root
    / "sub-CR022/ses-20201011/fusi"
    / "sub-CR022_ses-20201011_task-spontaneous"
      "_acq-slice03_pwd.nii.gz"
)

# Average over time and convert to dB scale.
mean_db = data.mean("time").fusi.scale.db()

# Plot all z-slices.
mean_db.fusi.plot.volume(
    cmap="gray", cbar_label="Power Doppler (dB)"
)
```

</div>
<div>

<img src="images/home/quickstart-dark.png#only-dark" alt="Mean power Doppler in dB">
<img src="images/home/quickstart-light.png#only-light" alt="Mean power Doppler in dB">

</div>
</div>
