[![PyPI version](https://img.shields.io/pypi/v/confusius)](https://pypi.org/project/confusius/)
[![Python versions](https://img.shields.io/pypi/pyversions/confusius)](https://pypi.org/project/confusius/)
[![DOI](https://zenodo.org/badge/1155356116.svg)](https://doi.org/10.5281/zenodo.18611124)
[![codecov](https://codecov.io/github/sdiebolt/confusius/graph/badge.svg?token=TL5AIRNSHS)](https://codecov.io/github/sdiebolt/confusius)

# ConfUSIus <img src="docs/images/confusius-logo.svg" width="200" title="ConfUSIus" alt="ConfUSIus" align="right">

> [!WARNING]
> ConfUSIus is currently in pre-alpha and under active development. The API is subject
> to change, and features may be incomplete or unstable.

ConfUSIus is a Python package for handling, visualization, preprocessing, and
statistical analysis of functional ultrasound imaging (fUSI) data.

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
uv add git+https://github.com/sdiebolt/confusius.git
```

### 3. Check installation

Check that ConfUSIus is correctly installed by opening a Python interpreter and
importing the package:

```python
import confusius
```

If no error is raised, you have installed ConfUSIus correctly.

## Citing ConfUSIus

If you use ConfUSIus in your research, please cite it using the following reference:

> Le Meur-Diebolt, S. (2026). ConfUSIus (v0.0.1-a7). Zenodo.
> https://doi.org/10.5281/zenodo.18611124

Or in BibTeX format:

```bibtex
@software{confusius,
  author    = {Le Meur-Diebolt, Samuel},
  title     = {ConfUSIus},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.0.1-a7},
  doi       = {10.5281/zenodo.18611124},
  url       = {https://doi.org/10.5281/zenodo.18611124}
}
```
