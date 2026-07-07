---
icon: lucide/database
---

# Datasets

The [`confusius.datasets`][confusius.datasets] module provides fetchers for publicly
available atlases, templates, and fUSI datasets distributed in
[fUSI-BIDS](https://bids.neuroimaging.io/) format. Each fetcher downloads the dataset
on first call, caches it locally for offline reuse, and returns either the path to the
root directory, or a more specific object (e.g., a DataArray for templates or an
[atlas Dataset][confusius.atlas] for atlases]).

!!! tip "Try before you buy"
    Fetchers generally accept filters (subjects, sessions, tasks, derivatives, etc.) so
    you can download a small subset first and decide later whether you want the full
    dataset. Cached files are never re-downloaded.

## Quick Start

The fastest way to get started is to fetch a single subject from the Nunez-Elizalde
2022 dataset[^nunez2022] and load one of its NIfTI files:

```python
import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022

# Download data for one mouse, one task, one brain slice (~30 MB).
root = fetch_nunez_elizalde_2022(
    subjects=["CR020"],
    sessions=["20191122"],
    tasks=["spontaneous"], 
    acqs=["slice03"],
)

# Load a power Doppler acquisition from the returned BIDS tree.
pwd = cf.load(
   root 
    / "sub-CR020"
    / "ses-20191122"
    / "fusi"
    / "sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz"
)
print(pwd.dims)
# Output: ('time', 'z', 'y', 'x')
```


## Datasets Storage

ConfUSIus resolves the cache directory using the following priority chain:

1. The `data_dir` argument passed to the fetcher.
2. The `CONFUSIUS_DATA` environment variable.
3. The platform cache directory (e.g. `~/.cache/confusius` on Linux,
   `~/Library/Caches/confusius` on macOS, `%LOCALAPPDATA%\confusius\Cache` on Windows).

You can inspect the resolved directory at any time with
[`get_datasets_dir`][confusius.datasets.get_datasets_dir]:

```python
from confusius.datasets import get_datasets_dir

print(get_datasets_dir())
# /home/alice/.cache/confusius
```

Each fetcher creates its own BIDS-root subdirectory under this path (e.g.
`nunez-elizalde-2022-bids/`), so multiple datasets can coexist safely.

## Listing Available Datasets

Use [`list_datasets`][confusius.datasets.list_datasets] to print a table of available
fetchers and their full download sizes:

```python
from confusius.datasets import list_datasets

list_datasets()
```

```text
                Available Datasets
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Fetch function                   ┃     Size ┃ On disk ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ fetch_cybis_pereira_2026         │ 12.88 GB │    ✗    │
│ fetch_landemard_2026             │ 42.04 GB │    ✗    │
│ fetch_nunez_elizalde_2022        │ 6.983 GB │    ✗    │
│ fetch_template_huang_2025        │ 16.34 MB │    ✗    │
│ fetch_template_pepe_mariani_2026 │ 5.508 MB │    ✗    │
└──────────────────────────────────┴──────────┴─────────┘
```

The sizes shown are for the **full** dataset. Filtered fetches are typically a small
fraction of this (see the examples below).

The same table is available from the command line:

```bash
confusius datasets --list
```

## Available fUSI-BIDS Datasets

=== "Nunez-Elizalde 2022"

    Simultaneous neural activity and cerebral blood volume recordings in awake mice,
    from Nunez-Elizalde et al. (2022)[^nunez2022], converted to fUSI-BIDS format and
    hosted on [OSF (43skw)](https://osf.io/43skw/). Total size: **~7 GB**.

    Use [`fetch_nunez_elizalde_2022`][confusius.datasets.fetch_nunez_elizalde_2022] to
    download the dataset. Four filters narrow the download:

    | Filter     | BIDS entity | Example               |
    |------------|-------------|-----------------------|
    | `subjects` | `sub-`      | `["CR017", "CR020"]`  |
    | `sessions` | `ses-`      | `["20191122"]`        |
    | `tasks`    | `task-`     | `["spontaneous"]`     |
    | `acqs`     | `acq-`      | `["slice03"]`         |

    Angiography files are always included regardless of the `tasks`/`acqs` filters,
    since they are useful as anatomical references.

    ```python
    from confusius.datasets import fetch_nunez_elizalde_2022

    # Two subjects, one task, one slice.
    bids_root = fetch_nunez_elizalde_2022(
        subjects=["CR017", "CR020"],
        tasks=["spontaneous"],
        acqs=["slice03"],
    )
    ```

=== "Cybis Pereira 2026"

    Functional ultrasound imaging data from freely-moving rats investigating vascular
    coding of speed in the spatial navigation system, from Cybis Pereira et al.
    (2026)[^cybis2026], hosted on [OSF (2v6f7)](https://osf.io/2v6f7/).
    Total size: **~13 GB**.

    Use [`fetch_cybis_pereira_2026`][confusius.datasets.fetch_cybis_pereira_2026] to
    download the dataset. Four filters narrow the download:

    | Filter     | BIDS entity / scope | Example                              |
    |------------|---------------------|--------------------------------------|
    | `datasets` | dataset name        | `"glm-speed"`, `["rawdata", "dlc-videos"]` |
    | `subjects` | `sub-`              | `["rat75", "rat73"]`                 |
    | `sessions` | `ses-`              | `["20220523"]`                       |
    | `acqs`     | `acq-`              | `["slice32"]`                        |

    Files that lack a session or acquisition entity (e.g. subject-level
    statmaps in `glm-speed`, or `decode-speed` aggregates) are kept regardless
    of the `sessions` / `acqs` filters, since they aggregate across those
    entities.

    The `datasets` filter accepts:

    | Name                        | Contents                                   |
    |-----------------------------|--------------------------------------------|
    | `rawdata`                   | Raw fUSI acquisitions per subject          |
    | `glm-speed`                 | GLM outputs for linear speed               |
    | `glm-angular-speed`         | GLM outputs for angular speed              |
    | `decode-speed`              | Within-animal speed decoding results       |
    | `interanimal-decode-speed`  | Inter-animal speed decoding results        |
    | `dlc-videos`                | DeepLabCut behavior tracking videos        |

    Derivatives are small relative to `rawdata`, so fetching a single derivative
    across all subjects is often the right first step:

    ```python
    from confusius.datasets import fetch_cybis_pereira_2026

    # All subjects, GLM-speed derivative only.
    bids_root = fetch_cybis_pereira_2026(datasets="glm-speed")

    # Or raw data for a single subject, session, and acquisition slice.
    bids_root = fetch_cybis_pereira_2026(
        datasets="rawdata",
        subjects="rat73",
        sessions="20220523",
        acqs="slice32",
    )
    ```

=== "Landemard 2026"

    Functional ultrasound imaging recordings from awake, head-fixed,
    freely-running mice, from Landemard et al. (2026)[^landemard2026],
    re-exported to fUSI-BIDS format and hosted on
    [OSF (dkseb)](https://osf.io/dkseb/overview).
    Total size: **~42 GB**.

    Use [`fetch_landemard_2026`][confusius.datasets.fetch_landemard_2026] to
    download the dataset. Four filters narrow the download:

    | Filter      | BIDS entity / scope | Example                          |
    |-------------|---------------------|----------------------------------|
    | `datasets`  | dataset name        | `"atlas_mapping"`, `["rawdata", "processed_data"]` |
    | `subjects`  | `sub-`              | `["ALD001", "ALD019"]`           |
    | `acqs`      | `acq-`              | `["ref04", "ref11"]`             |
    | `datatypes` | datatype directory  | `"fusi"`, `["fusi", "angio"]`    |

    The Landemard dataset has no session layer: every recording sits directly under
    `sub-*/fusi/` or `sub-*/angio/`. Files that lack an `acq-` entity (e.g.
    `sub-ALD001_scans.tsv`, `sub-ALD001/angio/sub-ALD001_pwd.nii.gz`) are always
    included regardless of the `acqs` filter.

    The `datasets` filter accepts:

    | Name              | Contents                                                              |
    |-------------------|-----------------------------------------------------------------------|
    | `rawdata`         | Raw fUSI and angiography acquisitions per subject                     |
    | `atlas_mapping`   | Per-subject Allen atlas alignment and region/session matches          |
    | `processed_data`  | Dataset-level compact representations, PSTHs, and ephys→fUSI filters  |

    ```python
    from confusius.datasets import fetch_landemard_2026

    # All subjects, atlas_mapping derivative only.
    bids_root = fetch_landemard_2026(datasets="atlas_mapping")

    # Or raw fUSI data for a single subject, with a single acquisition.
    bids_root = fetch_landemard_2026(
        datasets="rawdata",
        subjects="ALD001",
        acqs="ref04",
        datatypes="fusi",
    )
    ```

## Working with fUSI-BIDS Datasets

fUSI-BIDS dataset fetchers return a [`pathlib.Path`][pathlib.Path] to the dataset's root
directory. You may want to use the [PyBIDS](https://bids-standard.github.io/pybids/)
package for querying files, or simply use [`pathlib`][pathlib] for quick exploration:

```python
# List every power Doppler NIfTI for one subject.
for nii in sorted((bids_root / "sub-CR020").rglob("*_pwd.nii.gz")):
    print(nii.relative_to(bids_root))
```

See the [I/O guide](io.md) for loading NIfTI, Zarr, and Iconeus SCAN files into
Xarray DataArrays.

### Refreshing the Dataset Index

Each fUSI-BIDS dataset fetcher caches a `dataset_index.json` file mapping BIDS-relative
paths to OSF file metadata (download path and size). Pass `refresh=True` to re-fetch
this index from OSF and download any new files that appeared since the last call:

```python
# Pick up any files added to OSF since the last fetch.
bids_root = fetch_nunez_elizalde_2022(subjects=["CR020"], refresh=True)
```

Existing local files are never re-downloaded—`refresh=True` only adds what is missing.

## Available Templates

=== "Huang 2025"

    A vascular mouse fUSI template derived from Huang et al. (2025)[^huang2025] and
    distributed as a single ConfUSIus-compatible NIfTI on
    [OSF (am3jw)](https://osf.io/am3jw/). Total size: **~16.3 MB**.

    Use [`fetch_template_huang_2025`][confusius.datasets.fetch_template_huang_2025] to
    download and load the template directly:

    ```python
    from confusius.atlas import atlas_from_brainglobe
    from confusius.datasets import fetch_template_huang_2025

    template = fetch_template_huang_2025()
    atlas = atlas_from_brainglobe("allen_mouse_50um")
    resampled_atlas = atlas.atlas.resample_like(
        template,
        template.attrs["affines"]["physical_to_sform"],
    )
    ```

=== "Pepe Mariani 2026"

    A mouse fUSI template derived from Pepe, Mariani et al. (2026)[^pepe_mariani2026] and
    distributed as a single ConfUSIus-compatible NIfTI on
    [OSF (43tu9)](https://osf.io/43tu9/). Total size: **~5.5 MB**.

    Use [`fetch_template_pepe_mariani_2026`][confusius.datasets.fetch_template_pepe_mariani_2026] to
    download and load the template directly:

    ```python
    from confusius.atlas import atlas_from_brainglobe
    from confusius.datasets import fetch_template_pepe_mariani_2026

    template = fetch_template_pepe_mariani_2026()
    atlas = atlas_from_brainglobe("allen_mouse_100um")
    resampled_atlas = atlas.atlas.resample_like(
        template,
        template.attrs["affines"]["physical_to_sform"],
    )
    ```

## API Reference

See the [`confusius.datasets` API reference][confusius.datasets] for the full list of
parameters and return types.

[^nunez2022]:
    Nunez-Elizalde, A.O. et al. (2022). Neural correlates of blood flow measured by
    ultrasound. *Neuron*, 110(10), 1631–1640.
    <https://doi.org/10.1016/j.neuron.2022.02.012>

[^cybis2026]:
    Cybis Pereira, F. et al. (2026). A vascular code for speed in the spatial
    navigation system. *Cell Reports*, 45(1).
    <https://doi.org/10.1016/j.celrep.2025.116791>

[^landemard2026]:
    Landemard, A., Krumin, M., Harris, K. D., & Carandini, M. (2026). Brainwide
    blood volume reflects opposing neural populations. *Nature*.
    <https://doi.org/10.1038/s41586-026-10350-9>

[^huang2025]:
    Huang, Y.-A. et al. (2025). OfUSA: OpenfUS Analyzer, a versatile open-source
    framework for the analysis and visualization of functional ultrasound imaging data
    across animal models.
    <https://doi.org/10.1101/2025.09.16.676515>

[^pepe_mariani2026]:
    Pepe, C. et al. (2026). Structural and dynamic embedding of the mouse
    functional connectome revealed by functional ultrasound imaging (fUSI).
    <https://doi.org/10.64898/2026.02.05.704055>
