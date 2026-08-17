"""Fetcher for the Pepe Mariani et al. (2026) fUSI-BIDS dataset."""

from __future__ import annotations

import re
from pathlib import Path

from ._osf import (
    OsfFileInfo,
    download_osf_files,
    get_index,
    read_cached_index,
    update_cached_index,
)
from ._utils import get_datasets_dir, print_citation_message

_OSF_PROJECT_ID = "7yhdc"
_BIDS_ROOT = "pepe-mariani-2026-bids"
_TOTAL_SIZE_BYTES = 37_557_512_137
_CITATION = (
    "Pepe, C., Mariani, J.-C., Urosevic, M., Gini, S., Stuefer, A., Ricci, F., "
    "Galbusera, A., Iurilli, G., & Gozzi, A. (2026). [citation.title]Structural "
    "and dynamic embedding of the mouse functional connectome revealed by functional "
    "ultrasound imaging (Fusi).[/citation.title] "
    "[citation.doi]https://doi.org/10.64898/2026.02.05.704055[/citation.doi]"
)

_VALID_DATASETS = frozenset({"rawdata", "registered", "preprocessed", "Params"})
"""Valid values for the `datasets` parameter of `fetch_pepe_mariani_2026`."""

_VALID_DATATYPES = frozenset({"fusi", "angio"})
"""Valid values for the `datatypes` parameter of `fetch_pepe_mariani_2026`."""


def _filter_files(
    index: dict[str, OsfFileInfo],
    datasets: list[str] | None,
    subjects: list[str] | None,
    sessions: list[str] | None,
    acqs: list[str] | None,
    datatypes: list[str] | None,
) -> dict[str, OsfFileInfo]:
    """Filter the index to files matching the requested datasets and entities.

    Parameters
    ----------
    index : dict[str, OsfFileInfo]
        Full dataset index as returned by `get_index`.
    datasets : list[str] or None
        Datasets to include. Use `"rawdata"` for the raw fUSI/angio data and derivative
        names for processed outputs: `"registered"`, `"preprocessed"`, `"Params"`. If
        `None`, all datasets are included.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix). If `None`, all subjects are
        included.
    sessions : list[str] or None
        Session IDs to include (without "ses-" prefix). If `None`, all sessions are
        included. Files with no `ses-` entity are passed through.
    acqs : list[str] or None
        Acquisition labels to include (without "acq-" prefix). If `None`, all
        acquisitions are included. Files with no `acq-` entity are passed through.
    datatypes : list[str] or None
        BIDS datatype directories to include. If `None`, all datatypes are included.
        Files that do not sit under a datatype directory are passed through.

    Returns
    -------
    dict[str, OsfFileInfo]
        Subset of the index matching the filters.
    """
    filtered: dict[str, OsfFileInfo] = {}

    for path, file_info in index.items():
        parts = Path(path).parts

        if parts[0] == "derivatives":
            if len(parts) >= 2 and datasets is not None and parts[1] not in datasets:
                continue
            derivative_sub = next((p for p in parts if p.startswith("sub-")), None)
            if derivative_sub is not None:
                sub_id = derivative_sub.removeprefix("sub-")
                if subjects is not None and sub_id not in subjects:
                    continue
            if not _matches_entities(parts, sessions, acqs, datatypes):
                continue
            filtered[path] = file_info
            continue

        if not parts[0].startswith("sub-"):
            filtered[path] = file_info
            continue

        if datasets is not None and "rawdata" not in datasets:
            continue

        sub_id = parts[0].removeprefix("sub-")
        if subjects is not None and sub_id not in subjects:
            continue
        if not _matches_entities(parts, sessions, acqs, datatypes):
            continue
        filtered[path] = file_info

    return filtered


def _matches_entities(
    parts: tuple[str, ...],
    sessions: list[str] | None,
    acqs: list[str] | None,
    datatypes: list[str] | None,
) -> bool:
    """Return True when `parts` satisfies the requested entity filters.

    Parameters
    ----------
    parts : tuple[str, ...]
        BIDS-relative path components.
    sessions : list[str] or None
        Session IDs to include.
    acqs : list[str] or None
        Acquisition labels to include.
    datatypes : list[str] or None
        Datatype directory names to include.

    Returns
    -------
    bool
        Whether the path matches all declared filters.
    """
    if sessions is not None:
        ses_dir = next((p for p in parts if p.startswith("ses-")), None)
        if ses_dir is not None and ses_dir.removeprefix("ses-") not in sessions:
            return False

    if acqs is not None and parts:
        match = re.search(r"acq-([^_]+)", parts[-1])
        if match is not None and match.group(1) not in acqs:
            return False

    if datatypes is not None:
        datatype = _datatype_from_parts(parts)
        if datatype is not None and datatype not in datatypes:
            return False

    return True


def _datatype_from_parts(parts: tuple[str, ...]) -> str | None:
    """Return the BIDS datatype directory in `parts`, or None if absent.

    Parameters
    ----------
    parts : tuple[str, ...]
        BIDS-relative path components.

    Returns
    -------
    str or None
        Datatype directory name when present.
    """
    for i, part in enumerate(parts):
        if part.startswith("ses-") and i + 1 < len(parts) - 1:
            candidate = parts[i + 1]
            if candidate in _VALID_DATATYPES:
                return candidate
    return None


def fetch_pepe_mariani_2026(
    data_dir: str | Path | None = None,
    datasets: str | list[str] | None = None,
    subjects: str | list[str] | None = None,
    sessions: str | list[str] | None = None,
    acqs: str | list[str] | None = None,
    datatypes: str | list[str] | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> Path:
    """Fetch the Pepe Mariani 2026 fUSI-BIDS dataset.

    Downloads transcranial mouse resting-state fUSI recordings and derivatives,
    re-exported to fUSI-BIDS format from Pepe et al. (2026).

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the dataset. Defaults to the platform cache
        directory, overridable via the `CONFUSIUS_DATA` environment variable.
    datasets : str or list[str], optional
        Datasets to download. Use `"rawdata"` for raw fUSI/angio data and derivative
        names for processed outputs: `"registered"`, `"preprocessed"`, `"Params"`.
        Accepts a single string or a list. If not provided, all datasets are downloaded.
    subjects : str or list[str], optional
        Subject IDs to download (without "sub-" prefix). If not provided, all subjects
        are downloaded.
    sessions : str or list[str], optional
        Session IDs to download (without "ses-" prefix). If not provided, all sessions
        are downloaded. Files with no session entity are always included.
    acqs : str or list[str], optional
        Acquisition labels to download (without "acq-" prefix). If not provided, all
        acquisitions are downloaded. Files with no acquisition entity are always included.
    datatypes : str or list[str], optional
        BIDS datatype directories to download, e.g. `"fusi"` or `"angio"`. If not
        provided, all datatypes are downloaded. Files that do not sit under a datatype
        directory are always included.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and reconcile local files against
        it.
    print_citation : bool, default: True
        Whether to print the citation for the dataset.

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
        Pepe, C. et al. (2026). Structural and dynamic embedding of the mouse functional
        connectome revealed by functional ultrasound imaging (Fusi).
        [https://doi.org/10.64898/2026.02.05.704055](https://doi.org/10.64898/2026.02.05.704055)
    [^2]:
        fUSI-BIDS dataset on OSF: [https://osf.io/7yhdc/](https://osf.io/7yhdc/)
    [^3]:
        Dataset license (CC BY 4.0):
        [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(acqs, str):
        acqs = [acqs]
    if isinstance(datatypes, str):
        datatypes = [datatypes]

    if datasets is not None:
        invalid = set(datasets) - _VALID_DATASETS
        if invalid:
            raise ValueError(
                f"Unknown dataset(s): {invalid}. Valid options: {sorted(_VALID_DATASETS)}"
            )
    if datatypes is not None:
        invalid = set(datatypes) - _VALID_DATATYPES
        if invalid:
            raise ValueError(
                f"Unknown datatype(s): {invalid}. Valid options: {sorted(_VALID_DATATYPES)}"
            )

    previous_index = read_cached_index(bids_dir) if refresh else None
    index = get_index(bids_dir, _OSF_PROJECT_ID, _BIDS_ROOT, refresh=refresh)
    files = _filter_files(index, datasets, subjects, sessions, acqs, datatypes)

    download_osf_files(bids_dir, files, previous_index, refresh=refresh)
    if refresh:
        update_cached_index(bids_dir, index, previous_index or {}, files)

    if print_citation:
        print_citation_message(_CITATION, "dataset")
    return bids_dir
