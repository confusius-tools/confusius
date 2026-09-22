"""Fetcher for the Pereira et al. (2025) fUSI-BIDS dataset."""

from __future__ import annotations

from pathlib import Path

from ._osf import (
    OsfFileInfo,
    download_osf_files,
    get_index,
    read_cached_index,
    update_cached_index,
)
from ._utils import get_datasets_dir, print_citation_message

_OSF_PROJECT_ID = "pqa65"
_BIDS_ROOT = "pereira-2025-bids"
_TOTAL_SIZE_BYTES = 29_963_983_096
_CITATION = (
    "Pereira, M., Droguerre, M., Valdebenito, M., Vidal, L., Marcy, G., "
    "Benkeder, S., Marchal, P., Comte, J.-C., Pascual, O., Zimmer, L., "
    "& Vidal, B. (2025). [citation.title]Induction of haemodynamic travelling "
    "waves by glial-related vasomotion in a rat model of neuroinflammation: "
    "Implications for functional neuroimaging.[/citation.title] "
    "[italic]eBioMedicine[/italic], 116, 105777. "
    "[citation.doi]https://doi.org/10.1016/j.ebiom.2025.105777[/citation.doi]"
)


def _filter_files(
    index: dict[str, OsfFileInfo],
    subjects: list[str] | None,
    sessions: list[str] | None,
    tasks: list[str] | None,
) -> dict[str, OsfFileInfo]:
    """Filter the index to files matching the requested entities.

    Parameters
    ----------
    index : dict[str, OsfFileInfo]
        Full dataset index as returned by `get_index`.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix). If `None`, all subjects are
        included.
    sessions : list[str] or None
        Session IDs to include (without "ses-" prefix). If `None`, all sessions are
        included. Files with no `ses-` entity are passed through.
    tasks : list[str] or None
        Task labels to include (without "task-" prefix). If `None`, all tasks are
        included. Files with no `task-` entity are passed through.

    Returns
    -------
    dict[str, OsfFileInfo]
        Subset of the index matching the filters.
    """
    filtered: dict[str, OsfFileInfo] = {}
    for path, file_info in index.items():
        parts = Path(path).parts

        if not parts[0].startswith("sub-"):
            filtered[path] = file_info
            continue

        sub_id = parts[0].removeprefix("sub-")
        if subjects is not None and sub_id not in subjects:
            continue
        ses_dir = next((p for p in parts if p.startswith("ses-")), None)
        if (
            sessions is not None
            and ses_dir is not None
            and ses_dir.removeprefix("ses-") not in sessions
        ):
            continue

        if tasks is not None and "task-" in parts[-1]:
            task = parts[-1].split("task-", 1)[1].split("_", 1)[0]
            if task not in tasks:
                continue

        filtered[path] = file_info

    return filtered


def fetch_pereira_2025(
    data_dir: str | Path | None = None,
    subjects: str | list[str] | None = None,
    sessions: str | list[str] | None = None,
    tasks: str | list[str] | None = None,
    refresh: bool = False,
    print_citation: bool = True,
) -> Path:
    """Fetch the Pereira 2025 fUSI-BIDS dataset.

    Downloads fUSI recordings from a rat model of neuroinflammation, re-exported to
    fUSI-BIDS format from Pereira et al. (2025).

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Directory in which to cache the dataset. Defaults to the platform cache
        directory, overridable via the `CONFUSIUS_DATA` environment variable.
    subjects : str or list[str], optional
        Subject IDs to download (without "sub-" prefix), e.g. `"r11582"`. If not
        provided, all subjects are downloaded.
    sessions : str or list[str], optional
        Session IDs to download (without "ses-" prefix), e.g. `"awakebaseline"`. If not
        provided, all sessions are downloaded. Files with no session entity are always
        included.
    tasks : str or list[str], optional
        Task labels to download (without "task-" prefix), e.g. `"rest"` or `"stim"`.
        If not provided, all tasks are downloaded. Files with no task entity are always
        included.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and reconcile local files against
        it.
    print_citation : bool, default: True
        Whether to print the citation for the dataset.

    Returns
    -------
    pathlib.Path
        Path to the BIDS root directory of the cached dataset.

    References
    ----------
    [^1]:
        Pereira, M. et al. (2025). Induction of haemodynamic travelling waves by
        glial-related vasomotion in a rat model of neuroinflammation: Implications for
        functional neuroimaging. *eBioMedicine*, 116, 105777.
        [https://doi.org/10.1016/j.ebiom.2025.105777](https://doi.org/10.1016/j.ebiom.2025.105777)
    [^2]:
        fUSI-BIDS dataset on OSF: [https://osf.io/pqa65/](https://osf.io/pqa65/)
    [^3]:
        Dataset license (CC BY 4.0):
        [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(tasks, str):
        tasks = [tasks]

    previous_index = read_cached_index(bids_dir) if refresh else None
    index = get_index(bids_dir, _OSF_PROJECT_ID, _BIDS_ROOT, refresh=refresh)
    files = _filter_files(index, subjects, sessions, tasks)

    download_osf_files(bids_dir, files, previous_index, refresh=refresh)
    if refresh:
        update_cached_index(bids_dir, index, previous_index or {}, files)

    if print_citation:
        print_citation_message(_CITATION, "dataset")
    return bids_dir
