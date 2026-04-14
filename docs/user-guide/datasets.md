---
icon: lucide/database
---

# Datasets

The [`confusius.datasets`][confusius.datasets] module provides fetchers for publicly available
fUSI datasets, distributed in [fUSI-BIDS](https://bids.neuroimaging.io/) format. Each
fetcher downloads the dataset on first call, caches it locally for offline re-use, and
returns the path to the BIDS root directory—ready to be loaded with
[`confusius.load`][confusius.load] or explored with your favorite BIDS-aware tool.

!!! tip "Try before you buy"
    Each fetcher accepts filters (subjects, sessions, tasks, derivatives, …) so you can
    download a small slice first and decide whether you want the full dataset later.
    Cached files are never re-downloaded.

## Quick Start

The fastest way to get started is to fetch a single subject from the Nunez-Elizalde
2022 dataset[^nunez2022] and load one of its NIfTI files:

```python
import confusius as cf
from confusius.datasets import fetch_nunez_elizalde_2022

# Download one subject, one task (~30 MB).
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


## Where Data Is Cached

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

## Discovering Available Datasets

Use [`list_datasets`][confusius.datasets.list_datasets] to print a table of available
fetchers and their full download sizes:

```python
from confusius.datasets import list_datasets

list_datasets()
```

```text
           Available Datasets           
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Fetch function             ┃     Size ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ fetch_cybis_pereira_2026   │    12 GB │
│ fetch_nunez_elizalde_2022  │ 6.503 GB │
└────────────────────────────┴──────────┘
```

The sizes shown are for the **full** dataset. Filtered fetches are typically a small
fraction of this (see the examples below).

## Available Datasets

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

## Refreshing the Dataset Index

Each fetcher caches a `dataset_index.json` mapping BIDS-relative paths to OSF file
IDs. Pass `refresh=True` to re-fetch this index from OSF and download any new files
that appeared since the last call:

```python
# Pick up any files added to OSF since the last fetch.
bids_root = fetch_nunez_elizalde_2022(subjects=["CR020"], refresh=True)
```

Existing local files are never re-downloaded—`refresh=True` only adds what is
missing.

## Working with the Returned BIDS Root

Each fetcher returns a [`pathlib.Path`][pathlib.Path] to the dataset's BIDS root
directory. You can walk it with any BIDS-aware tool, or simply use
[`pathlib`][pathlib] globs for quick exploration:

```python
# List every power Doppler NIfTI for one subject.
for nii in sorted((bids_root / "sub-CR020").rglob("*_pwd.nii.gz")):
    print(nii.relative_to(bids_root))
```

See the [I/O guide](io.md) for loading NIfTI, Zarr, and Iconeus SCAN files into
Xarray DataArrays.

## API Reference

See the [`confusius.datasets` API reference][confusius.datasets] for the full list
of parameters and return types.

[^nunez2022]:
    Nunez-Elizalde, A.O. et al. (2022). Neural correlates of blood flow measured by
    ultrasound. *Neuron*, 110(10), 1631–1640.
    <https://doi.org/10.1016/j.neuron.2022.02.012>

[^cybis2026]:
    Cybis Pereira, F. et al. (2026). A vascular code for speed in the spatial
    navigation system. *Cell Reports*, 45(1).
    <https://doi.org/10.1016/j.celrep.2025.116791>
