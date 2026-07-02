"""Shared OSF API helpers for dataset fetchers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import pooch
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ._pooch import _RichProgressAdapter, quiet_pooch_logger, retrieve_with_retries

_OSF_DOWNLOAD_BASE = "https://osf.io/download/{}/"
"""URL template for direct downloads of an OSF file by id."""

_INDEX_FILENAME = "dataset_index.json"
"""Filename of the per-dataset index mapping BIDS-relative paths to metadata."""


class OsfFileInfo(TypedDict):
    """Per-file metadata entry in an OSF-backed dataset index."""

    osf_path: str
    size: int


def resolve_index_url(project_id: str, bids_root: str) -> str:
    """Return the OSF download URL for a dataset's index file.

    Makes two OSF API calls: one to locate the BIDS root folder within the
    project's osfstorage, and one to locate `dataset_index.json` inside it.

    Parameters
    ----------
    project_id : str
        OSF project identifier, e.g. `"43skw"`.
    bids_root : str
        Name of the BIDS root folder on OSF,
        e.g. `"nunez-elizalde-2022-bids"`.

    Returns
    -------
    str
        Direct download URL for `dataset_index.json`.

    Raises
    ------
    RuntimeError
        If the BIDS root folder or the index file is not found on OSF.
    """
    resp = requests.get(f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/")
    resp.raise_for_status()

    folder_url = None
    for item in resp.json()["data"]:
        if item["attributes"]["name"] == bids_root:
            folder_url = item["relationships"]["files"]["links"]["related"]["href"]
            break

    if folder_url is None:
        raise RuntimeError(
            f"Could not find the {bids_root!r} folder on OSF (project {project_id})."
        )

    resp = requests.get(folder_url)
    resp.raise_for_status()

    for item in resp.json()["data"]:
        if item["attributes"]["name"] == _INDEX_FILENAME:
            return item["links"]["download"]

    raise RuntimeError(
        f"{_INDEX_FILENAME!r} was not found on OSF (project {project_id})."
    )


def get_index(
    data_dir: Path,
    project_id: str,
    bids_root: str,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return the dataset index, preferring a locally cached copy.

    When `refresh` is `False` and a cached index exists in `data_dir`,
    it is decoded and returned directly (offline-friendly). Otherwise the
    index is re-fetched from OSF and persisted to disk.

    Parameters
    ----------
    data_dir : pathlib.Path
        Local directory in which the index is cached.
    project_id : str
        OSF project identifier (see
        [`resolve_index_url`][confusius.datasets._osf.resolve_index_url]).
    bids_root : str
        Name of the BIDS root folder on OSF (see
        [`resolve_index_url`][confusius.datasets._osf.resolve_index_url]).
    refresh : bool, default: False
        If `True`, always re-fetch the latest index from OSF even if a
        local copy exists.

    Returns
    -------
    dict[str, Any]
        Mapping from BIDS-relative file paths to OSF-side identifiers.
        The value schema is dataset-specific; callers narrow the type at
        their assignment site.
    """
    index_path = data_dir / _INDEX_FILENAME
    if not refresh and index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))

    url = resolve_index_url(project_id, bids_root)
    response = requests.get(url)
    response.raise_for_status()
    index = response.json()
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def download_missing_osf_files(bids_dir: Path, files: dict[str, OsfFileInfo]) -> None:
    """Download missing OSF files described by an index mapping.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local BIDS root directory where files are cached.
    files : dict[str, OsfFileInfo]
        Mapping from BIDS-relative paths to OSF download metadata.
    """
    missing = {p: info for p, info in files.items() if not (bids_dir / p).exists()}
    if not missing:
        return

    total_bytes = sum(info["size"] for info in missing.values())

    with quiet_pooch_logger():
        pooch_logger = pooch.get_logger()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Downloading dataset...", total=total_bytes)

            for rel_path, file_info in missing.items():
                dest = bids_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                progress.update(
                    task,
                    description=f"Downloading [bold]{Path(rel_path).name}[/bold]",
                )
                adapter = _RichProgressAdapter(progress, task)
                osf_path = file_info["osf_path"]
                retrieve_with_retries(
                    url=_OSF_DOWNLOAD_BASE.format(osf_path.lstrip("/")),
                    dest=dest,
                    logger=pooch_logger,
                    progressbar=adapter,
                    on_retry=adapter.rewind,
                )

            progress.update(task, description="Download complete.")
