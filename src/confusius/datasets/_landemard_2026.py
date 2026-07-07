"""Fetcher for the Landemard et al. (2026) fUSI-BIDS dataset."""

from __future__ import annotations

import re
from pathlib import Path

from ._osf import OsfFileInfo, download_missing_osf_files, get_index
from ._utils import get_datasets_dir

_OSF_PROJECT_ID = "dkseb"
_BIDS_ROOT = "landemard-2026-bids"
_TOTAL_SIZE_BYTES = 42_036_009_786


_VALID_DATASETS = frozenset({"rawdata", "atlas_mapping", "processed_data"})
"""Valid values for the `datasets` parameter of `fetch_landemard_2026`."""


_VALID_DATATYPES = frozenset({"fusi", "angio"})
"""Valid values for the `datatypes` parameter of `fetch_landemard_2026`."""


def _filter_files(
    index: dict[str, OsfFileInfo],
    datasets: list[str] | None,
    subjects: list[str] | None,
    acqs: list[str] | None,
    datatypes: list[str] | None,
) -> dict[str, OsfFileInfo]:
    """Filter the index to files matching the requested datasets and subjects.

    Top-level BIDS metadata files (dataset_description.json, participants.*, etc.) and
    subject-level files (e.g. `sub-ALD001_scans.tsv`) are always included. Unlike
    `fetch_cybis_pereira_2026`, the Landemard dataset has no session layer: every
    recording sits directly under `sub-*/fusi/` or `sub-*/angio/`.

    Parameters
    ----------
    index : dict[str, OsfFileInfo]
        Full dataset index as returned by `get_index`.
    datasets : list[str] or None
        Datasets to include. Use `"rawdata"` for the raw fUSI/angio data and
        derivative names for processed outputs: `"atlas_mapping"`,
        `"processed_data"`. If `None`, all datasets are included.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix), e.g. `["ALD001"]`. If `None`,
        all subjects are included.
    acqs : list[str] or None
        Acquisition labels to include (without "acq-" prefix), e.g. `["ref04",
        "ref11"]`. If `None`, all acquisitions are included. Files with no `acq-` entity
        are passed through.
    datatypes : list[str] or None
        BIDS datatype directories to include, e.g. `["fusi"]` or `["fusi", "angio"]`.
        Valid values are `"fusi"` and `"angio"`. If `None`, all datatypes are included.
        Files that do not sit under a datatype directory (e.g. subject-level
        `scans.tsv`) are passed through.

    Returns
    -------
    dict[str, OsfFileInfo]
        Subset of the index matching the filters.
    """
    filtered: dict[str, OsfFileInfo] = {}

    for path, file_info in index.items():
        parts = Path(path).parts

        # Derivatives.
        if parts[0] == "derivatives":
            # Filter by derivative name.
            if len(parts) >= 2:
                deriv_name = parts[1]
                if datasets is not None and deriv_name not in datasets:
                    continue

            # Subject filter within derivatives.
            derivative_sub = next((p for p in parts if p.startswith("sub-")), None)
            if derivative_sub is not None:
                sub_id = derivative_sub.removeprefix("sub-")
                if subjects is not None and sub_id not in subjects:
                    continue

            filtered[path] = file_info
            continue

        # Always include top-level BIDS metadata files.
        if not parts[0].startswith("sub-"):
            filtered[path] = file_info
            continue

        # Rawdata (sub-* at root).
        if datasets is not None and "rawdata" not in datasets:
            continue

        sub_id = parts[0].removeprefix("sub-")
        if subjects is not None and sub_id not in subjects:
            continue

        # Subject-level files (e.g. sub-ALD001/sub-ALD001_scans.tsv) pass
        # through: they sit directly under the subject folder with no
        # datatype layer, so the acq filter does not apply.
        if len(parts) == 1 or parts[1] not in _VALID_DATATYPES:
            filtered[path] = file_info
            continue

        # Datatype filter (only applies when sitting under a known datatype).
        if datatypes is not None and parts[1] not in datatypes:
            continue

        # Acquisition filter.
        if acqs is not None:
            match = re.search(r"acq-([^_]+)", parts[-1])
            if match is not None and match.group(1) not in acqs:
                continue

        filtered[path] = file_info

    return filtered


def fetch_landemard_2026(
    data_dir: str | Path | None = None,
    datasets: str | list[str] | None = None,
    subjects: str | list[str] | None = None,
    acqs: str | list[str] | None = None,
    datatypes: str | list[str] | None = None,
    refresh: bool = False,
) -> Path:
    """Fetch the Landemard 2026 fUSI-BIDS dataset.

    Downloads functional ultrasound imaging recordings from awake, head-fixed,
    freely-running mice, re-exported to fUSI-BIDS format from Landemard et al. (2026).

    Files are downloaded on first call and cached locally. Subsequent calls with the
    same `data_dir` return immediately for already-cached files.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the dataset. Defaults to the platform cache
        directory (e.g. `~/.cache/confusius` on Linux, `~/Library/Caches/confusius` on
        macOS, `%LOCALAPPDATA%\\confusius\\Cache` on Windows), overridable via the
        `CONFUSIUS_DATA` environment variable.
    datasets : str or list[str], optional
        Datasets to download. Use `"rawdata"` for the raw fUSI/angio data and derivative
        names for processed outputs: `"atlas_mapping"`, `"processed_data"`. Accepts a
        single string or a list. If not provided, all datasets are downloaded.
    subjects : str or list[str], optional
        Subject IDs to download (without "sub-" prefix), e.g. `"ALD001"` or `["ALD001",
        "ALD019"]`. If not provided, all subjects are downloaded.
    acqs : str or list[str], optional
        Acquisition labels to download (without "acq-" prefix), e.g. `"ref04"` or
        `["ref04", "ref11"]`. If not provided, all acquisitions are downloaded. Files
        with no `acq-` entity (e.g. `sub-ALD001_scans.tsv`,
        `sub-ALD001/angio/sub-ALD001_pwd.nii.gz`) are always included. The `run-` entity
        is not exposed as a filter.
    datatypes : str or list[str], optional
        BIDS datatype directories to download, e.g. `"fusi"`, `"angio"`, `["fusi",
        "angio"]`. Valid values are `"fusi"` and `"angio"`. If not provided, all
        datatypes are downloaded. Files that do not sit under a datatype directory (e.g.
        subject-level `scans.tsv`) are always included.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and reconcile local files against
        it: missing files are downloaded, and cached files whose MD5 no longer matches the
        index are re-downloaded. If `False` and all requested files are already cached, the
        function returns immediately without any network access.

    Returns
    -------
    pathlib.Path
        Path to the BIDS root directory of the cached dataset.

    Raises
    ------
    ValueError
        If an unknown dataset name is passed in `datasets`, or an unknown datatype is
        passed in `datatypes`.

    References
    ----------
    [^1]:
        Landemard, A., Krumin, M., Harris, K. D., & Carandini, M. (2026).
        Brainwide blood volume reflects opposing neural populations. *Nature*.
        [https://doi.org/10.1038/s41586-026-10350-9](https://doi.org/10.1038/s41586-026-10350-9)

    [^2]:
        fUSI-BIDS dataset on OSF:
        [https://osf.io/dkseb/](https://osf.io/dkseb/overview)

    [^3]:
        Dataset license (CC BY-NC 4.0):
        [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(acqs, str):
        acqs = [acqs]
    if isinstance(datatypes, str):
        datatypes = [datatypes]

    if datasets is not None:
        invalid = set(datasets) - _VALID_DATASETS
        if invalid:
            raise ValueError(
                f"Unknown dataset(s): {invalid}. "
                f"Valid options: {sorted(_VALID_DATASETS)}"
            )

    if datatypes is not None:
        invalid = set(datatypes) - _VALID_DATATYPES
        if invalid:
            raise ValueError(
                f"Unknown datatype(s): {invalid}. "
                f"Valid options: {sorted(_VALID_DATATYPES)}"
            )

    index = get_index(bids_dir, _OSF_PROJECT_ID, _BIDS_ROOT, refresh=refresh)
    files = _filter_files(index, datasets, subjects, acqs, datatypes)

    download_missing_osf_files(bids_dir, files, refresh=refresh)

    return bids_dir
