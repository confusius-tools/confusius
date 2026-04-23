"""Fetcher for the Cybis Pereira et al. (2026) fUSI-BIDS dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from ._osf import OsfFileInfo, download_missing_osf_files, get_index
from ._utils import get_datasets_dir

_OSF_PROJECT_ID = "2v6f7"
_BIDS_ROOT = "cybis-pereira-2026-bids"
_TOTAL_SIZE_BYTES = 12_883_924_421


_VALID_DATASETS = frozenset(
    {
        "rawdata",
        "glm-speed",
        "glm-angular-speed",
        "decode-speed",
        "interanimal-decode-speed",
        "dlc-videos",
    }
)
"""Valid values for the `datasets` parameter of `fetch_cybis_pereira_2026`."""


def _filter_files(
    index: dict[str, OsfFileInfo],
    datasets: list[str] | None,
    subjects: list[str] | None,
    sessions: list[str] | None,
    acqs: list[str] | None,
) -> dict[str, OsfFileInfo]:
    """Filter the index to files matching the requested datasets and subjects.

    Top-level BIDS metadata files (dataset_description.json, participants.*,
    etc.) are always included. The `sessions` and `acqs` filters only
    exclude files that carry the corresponding BIDS entity; subject- or
    dataset-level aggregate files (e.g. `sub-rat73_acq-slice32_statmap.nii`
    in derivatives, or `decode-speed/sub-rat73/...`) are kept when they
    match every entity they do declare.

    Parameters
    ----------
    index : dict[str, OsfFileInfo]
        Full dataset index as returned by `get_index`.
    datasets : list[str] or None
        Datasets to include. Use `"rawdata"` for the raw subject data and
        any derivative name (e.g. `"glm-speed"`, `"decode-speed"`) for
        specific derivatives. If `None`, all datasets are included.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix), e.g. `["rat83"]`.
        If `None`, all subjects are included.
    sessions : list[str] or None
        Session IDs to include (without "ses-" prefix), e.g.
        `["20220523"]`. If `None`, all sessions are included. Files with
        no `ses-` entity are passed through (e.g. subject-level
        derivatives that aggregate across sessions).
    acqs : list[str] or None
        Acquisition labels to include (without "acq-" prefix), e.g.
        `["slice32"]`. If `None`, all acquisitions are included. Files
        with no `acq-` entity are passed through.

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

            if not _matches_entities(parts, sessions, acqs):
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

        if not _matches_entities(parts, sessions, acqs):
            continue

        filtered[path] = file_info

    return filtered


def _matches_entities(
    parts: tuple[str, ...],
    sessions: list[str] | None,
    acqs: list[str] | None,
) -> bool:
    """Return True when the path satisfies the session and acquisition filters.

    A file matches when, for each of `sessions` and `acqs`, it either
    omits the corresponding BIDS entity entirely or declares a value
    that is in the requested list. The session is read from any
    `ses-*` directory in `parts`; the acquisition is read from the
    `acq-*` entity in the filename.
    """
    if sessions is not None:
        ses_dir = next((p for p in parts if p.startswith("ses-")), None)
        if ses_dir is not None:
            ses_id = ses_dir.removeprefix("ses-")
            if ses_id not in sessions:
                return False

    if acqs is not None and parts:
        match = re.search(r"acq-([^_]+)", parts[-1])
        if match is not None and match.group(1) not in acqs:
            return False

    return True


def fetch_cybis_pereira_2026(
    data_dir: str | Path | None = None,
    datasets: str | list[str] | None = None,
    subjects: str | list[str] | None = None,
    sessions: str | list[str] | None = None,
    acqs: str | list[str] | None = None,
    refresh: bool = False,
) -> Path:
    """Fetch the Cybis Pereira 2026 fUSI-BIDS dataset.

    Downloads functional ultrasound imaging data from freely-moving rats
    investigating vascular coding of speed in the spatial navigation system,
    converted to fUSI-BIDS format from Cybis Pereira et al. (2026).

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
        Datasets to download. Use `"rawdata"` for the raw fUSI data and
        derivative names for processed outputs: `"glm-speed"`,
        `"glm-angular-speed"`, `"decode-speed"`,
        `"interanimal-decode-speed"`, `"dlc-videos"`. Accepts a single
        string or a list. If not provided, all datasets are downloaded.
    subjects : str or list[str], optional
        Subject IDs to download (without "sub-" prefix), e.g. `"rat73"` or
        `["rat75", "rat73"]`. If not provided, all subjects are downloaded.
    sessions : str or list[str], optional
        Session IDs to download (without "ses-" prefix), e.g.
        `"20220523"` or `["20220523", "20220524"]`. If not provided,
        all sessions are downloaded. Files with no session entity
        (e.g. subject-level derivatives) are always included.
    acqs : str or list[str], optional
        Acquisition labels to download (without "acq-" prefix), e.g.
        `"slice32"` or `["slice32", "slice42"]`. If not provided, all
        acquisitions are downloaded. Files with no acquisition entity
        are always included.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and download any files
        that are missing locally. If `False` and all requested files are
        already cached, the function returns immediately without any network
        access.

    Returns
    -------
    pathlib.Path
        Path to the BIDS root directory of the cached dataset.

    Raises
    ------
    ValueError
        If an unknown dataset name is passed in `datasets`.

    References
    ----------
    [^1]:
        Cybis Pereira, F. et al. (2026). A vascular code for speed in the spatial
        navigation system. *Cell Reports*, 45(1).
        [https://doi.org/10.1016/j.celrep.2025.116791](https://doi.org/10.1016/j.celrep.2025.116791)

    [^2]:
        fUSI-BIDS dataset on OSF: [https://osf.io/2v6f7/](https://osf.io/2v6f7/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    # Normalize str to list.
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(subjects, str):
        subjects = [subjects]
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(acqs, str):
        acqs = [acqs]

    if datasets is not None:
        invalid = set(datasets) - _VALID_DATASETS
        if invalid:
            raise ValueError(
                f"Unknown dataset(s): {invalid}. "
                f"Valid options: {sorted(_VALID_DATASETS)}"
            )

    index = cast(
        "dict[str, OsfFileInfo]",
        get_index(bids_dir, _OSF_PROJECT_ID, _BIDS_ROOT, refresh=refresh),
    )
    files = _filter_files(index, datasets, subjects, sessions, acqs)

    download_missing_osf_files(bids_dir, files)

    return bids_dir
