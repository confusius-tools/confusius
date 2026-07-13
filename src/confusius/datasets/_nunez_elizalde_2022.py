"""Fetcher for the Nunez-Elizalde et al. (2022) fUSI-BIDS dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ._osf import (
    OsfFileInfo,
    download_osf_files,
    get_index,
    read_cached_index,
    update_cached_index,
)
from ._utils import get_datasets_dir, print_citation_message

if TYPE_CHECKING:
    from collections.abc import Callable

_OSF_PROJECT_ID = "43skw"
_BIDS_ROOT = "nunez-elizalde-2022-bids"
_TOTAL_SIZE_BYTES = 6_982_575_320
_CITATION = (
    "Nunez-Elizalde, A. O., Krumin, M., Reddy, C. B., Montaldo, G., Urban, A., "
    "Harris, K. D., & Carandini, M. (2022). [citation.title]Neural correlates of blood "
    "flow measured by ultrasound.[/citation.title] [italic]Neuron[/italic], 110(10), "
    "1631–1640.e4. "
    "[citation.doi]https://doi.org/10.1016/j.neuron.2022.02.012[/citation.doi]"
)

_VALID_DATASETS = frozenset({"rawdata", "allenccf_align"})
"""Valid values for the `datasets` parameter of `fetch_nunez_elizalde_2022`."""

_VALID_DATATYPES = frozenset({"fusi", "angio"})
"""Valid values for the `datatypes` parameter of `fetch_nunez_elizalde_2022`."""


def _filter_files(
    index: dict[str, OsfFileInfo],
    datasets: list[str] | None,
    subjects: list[str] | None,
    sessions: list[str] | None,
    tasks: list[str] | None,
    acqs: list[str] | None,
    datatypes: list[str] | None,
) -> dict[str, OsfFileInfo]:
    """Filter the index to files matching the requested datasets and entities.

    Top-level BIDS metadata files (dataset_description.json, participants.*,
    etc.) are always included. The `sessions`, `tasks`, `acqs`, and `datatypes`
    filters only exclude files that carry the corresponding BIDS entity or live
    under a datatype directory; subject- or derivative-level metadata files are
    kept when they match every entity they do declare.

    Parameters
    ----------
    index : dict[str, OsfFileInfo]
        Full dataset index as returned by `get_index`.
    datasets : list[str] or None
        Datasets to include. Use `"rawdata"` for the raw subject data and
        `"allenccf_align"` for the atlas-alignment derivatives. If `None`, all
        datasets are included.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix), e.g. `["CR020"]`. If `None`,
        all subjects are included.
    sessions : list[str] or None
        Session IDs to include (without "ses-" prefix), e.g. `["20191122"]`. If
        `None`, all sessions are included. Files with no session entity are passed
        through.
    tasks : list[str] or None
        Task names to include, e.g. `["kalatsky", "spontaneous"]`. If `None`, all
        tasks are included. Only applies to `fusi/` files.
    acqs : list[str] or None
        Acquisition labels to include (without `acq-`), e.g. `["slice03"]`. If `None`,
        all acquisitions are included. Only applies to `fusi/` files.
    datatypes : list[str] or None
        BIDS datatype directories to include, e.g. `["fusi", "angio"]`. If `None`,
        all datatypes are included. Files that do not sit under a datatype directory
        are passed through.

    Returns
    -------
    dict[str, OsfFileInfo]
        Subset of the index matching the filters.
    """
    filtered: dict[str, OsfFileInfo] = {}

    for path, file_info in index.items():
        parts = Path(path).parts

        if parts[0] in {"derivatives", "sourcedata"}:
            if len(parts) >= 2 and datasets is not None and parts[1] not in datasets:
                continue

            dataset_sub = next((p for p in parts if p.startswith("sub-")), None)
            if dataset_sub is not None:
                sub_id = dataset_sub.removeprefix("sub-")
                if subjects is not None and sub_id not in subjects:
                    continue

            if not _matches_entities(parts, sessions, tasks, acqs, datatypes):
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

        if not _matches_entities(parts, sessions, tasks, acqs, datatypes):
            continue

        filtered[path] = file_info

    return filtered


def _matches_entities(
    parts: tuple[str, ...],
    sessions: list[str] | None,
    tasks: list[str] | None,
    acqs: list[str] | None,
    datatypes: list[str] | None,
) -> bool:
    """Return True when the path satisfies the entity filters."""
    if sessions is not None:
        ses_dir = next((p for p in parts if p.startswith("ses-")), None)
        if ses_dir is not None and ses_dir.removeprefix("ses-") not in sessions:
            return False

    datatype = _datatype_from_parts(parts)
    if datatypes is not None and datatype is not None and datatype not in datatypes:
        return False

    if datatype == "fusi" and parts:
        if tasks is not None:
            match = re.search(r"task-([^_]+)", parts[-1])
            if match is None or match.group(1) not in tasks:
                return False

        if acqs is not None:
            match = re.search(r"acq-([^_]+)", parts[-1])
            if match is None or match.group(1) not in acqs:
                return False

    return True


def _datatype_from_parts(parts: tuple[str, ...]) -> str | None:
    """Return the BIDS datatype directory in `parts`, or None if absent."""
    for i, part in enumerate(parts):
        if part.startswith("ses-") and i + 1 < len(parts) - 1:
            candidate = parts[i + 1]
            if candidate in _VALID_DATATYPES:
                return candidate
    return None


def fetch_nunez_elizalde_2022(
    data_dir: str | Path | None = None,
    datasets: str | list[str] | None = None,
    subjects: str | list[str] | None = None,
    sessions: str | list[str] | None = None,
    tasks: str | list[str] | None = None,
    acqs: str | list[str] | None = None,
    datatypes: str | list[str] | None = None,
    refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    print_citation: bool = True,
) -> Path:
    """Fetch the Nunez-Elizalde 2022 fUSI-BIDS dataset.

    Downloads simultaneous neural activity and cerebral blood volume recordings
    in awake mice, converted to fUSI-BIDS format from Nunez-Elizalde et al.
    (2022).

    Files are downloaded on first call and cached locally. Subsequent calls
    with the same `data_dir` return immediately for already-cached files.

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the dataset. Defaults to the platform
        cache directory (e.g. `~/.cache/confusius` on Linux,
        `~/Library/Caches/confusius` on macOS,
        `%LOCALAPPDATA%\\confusius\\Cache` on Windows),
        overridable via the `CONFUSIUS_DATA` environment variable.
    datasets : str or list[str], optional
        Datasets to download. Use `"rawdata"` for the raw subject data and
        `"allenccf_align"` for the atlas-alignment derivatives. Accepts a
        single string or a list. If not provided, all datasets are downloaded.
    subjects : str or list[str], optional
        Subject IDs to download (without "sub-" prefix), e.g. "CR020" or
        `["CR020"]`. Accepts a single string or a list. If not provided, all
        subjects are downloaded.
    sessions : str or list[str], optional
        Session IDs to download (without "ses-" prefix), e.g. "20191122" or
        `["20191122"]`. Accepts a single string or a list. If not provided,
        all sessions are downloaded. Files with no session entity are always
        included.
    tasks : str or list[str], optional
        Task names to download, e.g. "kalatsky" or
        `["kalatsky", "spontaneous"]`. Accepts a single string or a list. If
        not provided, all tasks are downloaded. Only applies to `fusi/` files.
    acqs : str or list[str], optional
        Acquisition labels to download (without `acq-`), e.g. "slice03" or
        `["slice03"]`. Accepts a single string or a list. If not provided,
        all acquisitions are downloaded. Only applies to `fusi/` files.
    datatypes : str or list[str], optional
        BIDS datatype directories to download, e.g. `"fusi"`, `"angio"`,
        `["fusi", "angio"]`. Valid values are `"fusi"` and `"angio"`.
        If not provided, all datatypes are downloaded. Files that do not sit
        under a datatype directory (e.g. subject-level metadata) are always
        included.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and reconcile local files against
        it: missing files are downloaded, and cached files whose MD5 changed upstream
        (comparing the cached index against the refreshed one) are re-downloaded. If
        `False` and all requested files are already cached, the function returns
        immediately without any network access.
    progress_callback : Callable[[int, int, str], None], optional
        Callback receiving cumulative downloaded bytes, total bytes to download,
        and a user-facing description. Intended for GUI progress bars.
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
        Nunez-Elizalde, A.O. et al. (2022). Neural correlates of blood flow measured by
        ultrasound. *Neuron*, 110(10), 1631–1640.
        [https://doi.org/10.1016/j.neuron.2022.02.012](https://doi.org/10.1016/j.neuron.2022.02.012)

    [^2]:
        fUSI-BIDS dataset on OSF: [https://osf.io/43skw/](https://osf.io/43skw/)

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
    if isinstance(tasks, str):
        tasks = [tasks]
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

    previous_index = read_cached_index(bids_dir) if refresh else None
    index = get_index(bids_dir, _OSF_PROJECT_ID, _BIDS_ROOT, refresh=refresh)
    files = _filter_files(
        index,
        datasets,
        subjects,
        sessions,
        tasks,
        acqs,
        datatypes,
    )

    download_osf_files(
        bids_dir,
        files,
        previous_index,
        refresh=refresh,
        progress_callback=progress_callback,
    )
    if refresh:
        update_cached_index(bids_dir, index, previous_index or {}, files)

    if print_citation:
        print_citation_message(_CITATION, "dataset")
    return bids_dir
