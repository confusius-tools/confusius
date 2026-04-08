"""Fetcher for the Nunez-Elizalde et al. (2022) fUSI-BIDS dataset."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pooch
import requests
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ._utils import get_datasets_dir

_OSF_PROJECT_ID = "43skw"
_BIDS_ROOT = "nunez-elizalde-2022-bids"
_INDEX_FILENAME = "dataset_index.json"
_OSF_DOWNLOAD_BASE = "https://osf.io/download/{}/"


def _resolve_index_url() -> str:
    """Find the dataset_index.json download URL via the OSF API.

    Makes two API calls: one to get the BIDS root folder, one to find the
    index file within it.

    Returns
    -------
    str
        Direct download URL for dataset_index.json.

    Raises
    ------
    RuntimeError
        If the index file is not found. This means nunez-upload --index-only
        has not been run yet.
    """
    # Get root storage listing to find the BIDS root folder.
    resp = requests.get(
        f"https://api.osf.io/v2/nodes/{_OSF_PROJECT_ID}/files/osfstorage/"
    )
    resp.raise_for_status()

    folder_url = None
    for item in resp.json()["data"]:
        if item["attributes"]["name"] == _BIDS_ROOT:
            folder_url = item["relationships"]["files"]["links"]["related"]["href"]
            break

    if folder_url is None:
        raise RuntimeError(
            f"Could not find the {_BIDS_ROOT!r} folder on OSF "
            f"(project {_OSF_PROJECT_ID})."
        )

    # Find dataset_index.json inside the BIDS root folder.
    resp = requests.get(folder_url)
    resp.raise_for_status()

    for item in resp.json()["data"]:
        if item["attributes"]["name"] == _INDEX_FILENAME:
            return item["links"]["download"]

    raise RuntimeError(
        f"{_INDEX_FILENAME!r} was not found on OSF (project {_OSF_PROJECT_ID}). "
        "Run 'nunez-upload --index-only' from the nunez-elizalde-2022-bids "
        "repository to generate it."
    )


def _get_index(data_dir: Path, refresh: bool = False) -> dict[str, str]:
    """Return the dataset index.

    Uses the locally cached index when available and `refresh` is False,
    enabling offline use. When `refresh` is True or no cached index exists,
    fetches the latest version from OSF.

    Parameters
    ----------
    data_dir : pathlib.Path
        Local directory where the index is cached.
    refresh : bool, default: False
        If True, always fetch the latest index from OSF even if a local copy
        exists.

    Returns
    -------
    dict[str, str]
        Mapping from BIDS-relative file paths to OSF file IDs.
    """
    index_path = data_dir / _INDEX_FILENAME
    if not refresh and index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

    url = _resolve_index_url()
    response = requests.get(url)
    response.raise_for_status()
    index = response.json()
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def _filter_files(
    index: dict[str, str],
    subjects: list[str] | None,
    sessions: list[str] | None,
    tasks: list[str] | None,
) -> dict[str, str]:
    """Filter the index to files matching the requested subjects/sessions/tasks.

    Top-level BIDS metadata files (dataset_description.json, participants.*,
    etc.) and subject-level files (sessions.tsv/.json) are always included.

    Parameters
    ----------
    index : dict[str, str]
        Full dataset index as returned by `_get_index`.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix), e.g. `["CR020"]`. If `None`, all
        subjects are included.
    sessions : list[str] or None
        Session IDs to include (without "ses-" prefix), e.g. `["20191122"]`. If `None`,
        all sessions are included.
    tasks : list[str] or None
        Task names to include, e.g. `["kalatsky", "spontaneous"]`. If `None`, all tasks
        are included. Only applies to `fusi/` files; `angio/` files are always included.

    Returns
    -------
    dict[str, str]
        Subset of the index matching the filters.
    """
    filtered: dict[str, str] = {}

    for path, osf_id in index.items():
        parts = Path(path).parts

        # Always include top-level BIDS files (no subject folder).
        if not parts[0].startswith("sub-"):
            filtered[path] = osf_id
            continue

        # Subject filter.
        sub_id = parts[0].removeprefix("sub-")
        if subjects is not None and sub_id not in subjects:
            continue

        # Subject-level files (e.g. sub-CR020_sessions.tsv).
        if len(parts) == 1 or not parts[1].startswith("ses-"):
            filtered[path] = osf_id
            continue

        # Session filter.
        ses_id = parts[1].removeprefix("ses-")
        if sessions is not None and ses_id not in sessions:
            continue

        # Task filter (only applies to fusi/ files).
        if tasks is not None and len(parts) >= 3 and parts[2] == "fusi":
            match = re.search(r"task-([^_]+)", parts[-1])
            if match is None or match.group(1) not in tasks:
                continue

        filtered[path] = osf_id

    return filtered


def fetch_nunez_elizalde_2022(
    data_dir: str | Path | None = None,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    tasks: list[str] | None = None,
    refresh: bool = False,
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
    subjects : list[str], optional
        Subject IDs to download (without "sub-" prefix), e.g. `["CR020"]`. If not
        provided, all subjects are downloaded.
    sessions : list[str], optional
        Session IDs to download (without "ses-" prefix), e.g. `["20191122"]`. If not
        provided, all sessions are downloaded.
    tasks : list[str], optional
        Task names to download, e.g. `["kalatsky", "spontaneous"]`. If not provided, all
        tasks are downloaded. Angiography files are always included regardless of this
        filter.
    refresh : bool, default: False
        Whether to re-fetch the dataset index from OSF and download any files that are
        missing locally. If `False` and all requested files are already cached, the
        function returns immediately without any network access.

    Returns
    -------
    pathlib.Path
        Path to the BIDS root directory of the cached dataset.

    References
    ----------
    [^1]:
        Nunez-Elizalde, A.O. et al. (2022). Neural correlates of blood flow measured by
        ultrasound. *Neuron*, 110(10), 1631–1640.
        [https://doi.org/10.1016/j.neuron.2022.02.012](https://doi.org/10.1016/j.neuron.2022.02.012)

    [^2]:
        fUSI-BIDS dataset on OSF: [https://osf.io/43skw/](https://osf.io/43skw/)
    """
    bids_dir = get_datasets_dir(data_dir) / _BIDS_ROOT
    bids_dir.mkdir(parents=True, exist_ok=True)

    index = _get_index(bids_dir, refresh=refresh)
    files = _filter_files(index, subjects, sessions, tasks)

    missing = {p: i for p, i in files.items() if not (bids_dir / p).exists()}
    if not missing:
        return bids_dir

    # Suppress pooch's INFO-level messages (SHA256 suggestions, download URLs) and route
    # any warnings/errors through rich so they don't break the progress bar layout.
    # Pooch uses logging.Logger("pooch") directly rather than
    # logging.getLogger("pooch"), so we must go through pooch.get_logger().
    pooch_logger = pooch.get_logger()
    original_handlers = pooch_logger.handlers[:]
    original_level = pooch_logger.level

    for handler in original_handlers:
        pooch_logger.removeHandler(handler)
    pooch_logger.addHandler(
        RichHandler(level=logging.WARNING, show_time=False, show_path=False)
    )
    pooch_logger.setLevel(logging.WARNING)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Downloading dataset...", total=len(missing))

            for rel_path, osf_id in missing.items():
                dest = bids_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                progress.update(
                    task,
                    description=f"Downloading [bold]{Path(rel_path).name}[/bold]",
                )
                pooch.retrieve(
                    url=_OSF_DOWNLOAD_BASE.format(osf_id.lstrip("/")),
                    known_hash=None,
                    fname=dest.name,
                    path=dest.parent,
                    progressbar=False,
                )
                progress.advance(task)

            progress.update(task, description="Download complete.")
    finally:
        for handler in pooch_logger.handlers[:]:
            pooch_logger.removeHandler(handler)
        for handler in original_handlers:
            pooch_logger.addHandler(handler)
        pooch_logger.setLevel(original_level)

    return bids_dir
