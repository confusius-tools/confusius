"""Shared OSF API helpers for dataset fetchers."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import NotRequired, TypedDict

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
    """Per-file metadata entry in an OSF-backed dataset index.

    Indices written before md5 tracking omit the `md5` key entirely; newer
    ones may still carry `md5: null` for a file whose hash is unknown. Both
    cases read back as a missing hash via `dict.get("md5")`.
    """

    osf_path: str
    size: int
    md5: NotRequired[str | None]


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
) -> dict[str, OsfFileInfo]:
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
    dict[str, OsfFileInfo]
        Mapping from BIDS-relative file paths to
        [`OsfFileInfo`][confusius.datasets._osf.OsfFileInfo] entries (`osf_path`,
        `size` in bytes, and an optional `md5` hex digest). The schema is the same
        for every dataset: the index is produced by the per-dataset upload script
        in the confusius-tools GitHub organisation and stored on OSF as
        `dataset_index.json`. Indices written before md5 tracking omit `md5`.
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


def _file_md5(path: Path) -> str:
    """Return the hex-encoded MD5 digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal MD5 digest, matching the `md5` values stored in
        `dataset_index.json`.
    """
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _select_downloads(
    bids_dir: Path,
    files: dict[str, OsfFileInfo],
    refresh: bool,
) -> dict[str, OsfFileInfo]:
    """Return the subset of `files` that must be (re)downloaded.

    A file is selected when it is absent locally. When `refresh` is `True`, a
    file that is present but whose local MD5 differs from the index `md5` is
    also selected. Entries without an `md5` (an index predating md5 tracking,
    or a `null` hash) fall back to the "download only if missing" behaviour and
    trigger a single warning, since their content cannot be checked.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local BIDS root directory where files are cached.
    files : dict[str, OsfFileInfo]
        Mapping from BIDS-relative paths to OSF download metadata.
    refresh : bool
        Whether to re-download cached files whose local MD5 no longer matches
        the index.

    Returns
    -------
    dict[str, OsfFileInfo]
        Subset of `files` to download, preserving the input order.
    """
    selected: dict[str, OsfFileInfo] = {}
    unhashable = False

    for rel_path, info in files.items():
        dest = bids_dir / rel_path
        if not dest.exists():
            selected[rel_path] = info
            continue
        if not refresh:
            continue
        md5 = info.get("md5")
        if not md5:
            unhashable = True
            continue
        if _file_md5(dest) != md5:
            selected[rel_path] = info

    if refresh and unhashable:
        warnings.warn(
            "Some cached files could not be checked for upstream changes because "
            "the dataset index has no md5 for them. Refresh only redownloaded "
            "missing files for those entries. Re-run the dataset's upload script "
            "to populate md5 hashes.",
            stacklevel=2,
        )

    return selected


def download_missing_osf_files(
    bids_dir: Path,
    files: dict[str, OsfFileInfo],
    refresh: bool = False,
) -> None:
    """Download OSF files described by an index mapping.

    Files absent from `bids_dir` are always downloaded. When `refresh` is
    `True`, cached files whose local MD5 differs from the index `md5` are
    re-downloaded as well; see
    [`_select_downloads`][confusius.datasets._osf._select_downloads].

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local BIDS root directory where files are cached.
    files : dict[str, OsfFileInfo]
        Mapping from BIDS-relative paths to OSF download metadata.
    refresh : bool, default: False
        Whether to re-download cached files whose local MD5 no longer matches
        the index.
    """
    to_download = _select_downloads(bids_dir, files, refresh)
    if not to_download:
        return

    total_bytes = sum(info["size"] for info in to_download.values())

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

            for rel_path, file_info in to_download.items():
                dest = bids_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                progress.update(
                    task,
                    description=f"Downloading [bold]{Path(rel_path).name}[/bold]",
                )
                adapter = _RichProgressAdapter(progress, task)
                osf_path = file_info["osf_path"]
                md5 = file_info.get("md5")
                retrieve_with_retries(
                    url=_OSF_DOWNLOAD_BASE.format(osf_path.lstrip("/")),
                    dest=dest,
                    logger=pooch_logger,
                    progressbar=adapter,
                    on_retry=adapter.rewind,
                    known_hash=f"md5:{md5}" if md5 else None,
                )

            progress.update(task, description="Download complete.")
