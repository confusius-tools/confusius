"""Shared OSF API helpers for dataset fetchers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

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

from ._pooch import (
    _CallbackProgressAdapter,
    _RichProgressAdapter,
    quiet_pooch_logger,
    retrieve_with_retries,
)

_OSF_DOWNLOAD_BASE = "https://osf.io/download/{}/"
"""URL template for direct downloads of an OSF file by id."""

_INDEX_FILENAME = "dataset_index.json"
"""Filename of the per-dataset index mapping BIDS-relative paths to metadata."""

_INDEX_ENTRY_KEYS = frozenset({"osf_path", "size", "md5"})
"""Keys every entry in a current `dataset_index.json` must define.

A cached index whose entries do not all define these keys is from an older,
incompatible confusius release and is rejected by
[`read_cached_index`][confusius.datasets._osf.read_cached_index]."""

if TYPE_CHECKING:
    from collections.abc import Callable


class OsfFileInfo(TypedDict):
    """Per-file metadata entry in an OSF-backed dataset index.

    Every current index defines `md5` for each file (its value may be `null`
    for a file whose hash is unknown). A locally cached index missing the key
    is from an older release and is rejected on load, so consumers can assume
    the key is present.
    """

    osf_path: str
    size: int
    md5: str | None


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

    When `refresh` is `False` and a cached index exists in `data_dir`, it is decoded and
    returned directly (offline-friendly). Otherwise the latest index is fetched from OSF
    and returned.

    The freshly fetched index is only persisted to disk on the very first fetch (when no
    cache exists yet). On a refresh of an existing cache, local and remote indices are
    merged via [`update_cached_index`][confusius.datasets._osf.update_cached_index], so
    that files the caller did not update keep their recorded md5.

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
        [`OsfFileInfo`][confusius.datasets._osf.OsfFileInfo] entries (`osf_path`, `size`
        in bytes, and an `md5` hex digest). The schema is the same for every dataset: the
        index is produced by the per-dataset upload script in the confusius-tools GitHub
        organisation and stored on OSF as `dataset_index.json`.

    Raises
    ------
    RuntimeError
        If a cached index exists but does not match the current schema (see
        [`read_cached_index`][confusius.datasets._osf.read_cached_index]).
    """
    index_path = data_dir / _INDEX_FILENAME
    cache_exists = index_path.exists()
    if not refresh and cache_exists:
        return read_cached_index(data_dir)

    url = resolve_index_url(project_id, bids_root)
    response = requests.get(url)
    response.raise_for_status()
    index = response.json()
    if not cache_exists:
        # First fetch: no cached md5s to preserve, so persist the full index.
        _write_index(index_path, index)
    return index


def _write_index(index_path: Path, index: dict[str, OsfFileInfo]) -> None:
    """Write a dataset index to disk as sorted, indented JSON.

    Parameters
    ----------
    index_path : pathlib.Path
        Destination path for `dataset_index.json`.
    index : dict[str, OsfFileInfo]
        Index mapping to serialise.
    """
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def update_cached_index(
    data_dir: Path,
    remote_index: dict[str, OsfFileInfo],
    previous_index: dict[str, OsfFileInfo],
    files: dict[str, OsfFileInfo],
) -> None:
    """Persist a refreshed index by merging, not replacing, the cached one.

    The cached `dataset_index.json` is the record of what is on disk, so it is not
    overwritten with the remote index. Instead the updated index is:

    - the remote entry for every newly added or updated file;
    - the local entry for any unchanged file, so that future downloads can still detect
      a mismatched md5;

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory holding the cached index.
    remote_index : dict[str, OsfFileInfo]
        Full index freshly fetched from OSF.
    previous_index : dict[str, OsfFileInfo]
        Index cached before the refresh.
    files : dict[str, OsfFileInfo]
        Requested subset of `remote_index` that was reconciled with disk.
    """
    merged: dict[str, OsfFileInfo] = {**remote_index, **previous_index, **files}
    _write_index(data_dir / _INDEX_FILENAME, merged)


def _validate_index(index: object, data_dir: Path) -> None:
    """Raise a helpful error if a cached index does not match the current schema.

    The check is deliberately structural: every entry must define the keys in
    `_INDEX_ENTRY_KEYS`. It future-proofs later schema changes—any older on-disk
    layout fails here rather than being silently mis-handled—and points the user at
    the dataset directory to delete.

    Parameters
    ----------
    index : object
        Decoded contents of a local `dataset_index.json`.
    data_dir : pathlib.Path
        Dataset root directory holding the index, shown to the user so they know
        which directory to remove.

    Raises
    ------
    RuntimeError
        If `index` is not a mapping whose every entry defines the expected keys.
    """
    if isinstance(index, dict) and all(
        isinstance(entry, dict) and _INDEX_ENTRY_KEYS <= entry.keys()
        for entry in index.values()
    ):
        return

    raise RuntimeError(
        f"The dataset index in {data_dir} has an outdated or unrecognised structure "
        f"(every entry must define {sorted(_INDEX_ENTRY_KEYS)}). This local dataset was "
        f"likely downloaded with an older version of confusius. Delete the dataset "
        f"directory and fetch it again to update it:\n\n    {data_dir}\n"
    )


def read_cached_index(data_dir: Path) -> dict[str, OsfFileInfo]:
    """Return the locally cached dataset index, or `{}` if none exists.

    Unlike [`get_index`][confusius.datasets._osf.get_index] this never touches the
    network: it only decodes the on-disk `dataset_index.json`. Callers read it before a
    refresh to capture the md5s the cached files were downloaded with, then compare
    those against the freshly fetched remote index to decide which files changed
    upstream.

    A cached index that is present is validated against the current schema, so a stale
    index from an older confusius release is reported rather than mis-handled.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory holding the cached index.

    Returns
    -------
    dict[str, OsfFileInfo]
        Cached index mapping, or an empty mapping when no cache exists.

    Raises
    ------
    RuntimeError
        If a cached index exists but does not match the current schema; the message
        names the dataset directory to delete and re-fetch.
    """
    index_path = data_dir / _INDEX_FILENAME
    if not index_path.exists():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _validate_index(index, data_dir)
    return index


def _select_downloads(
    bids_dir: Path,
    files: dict[str, OsfFileInfo],
    previous_index: dict[str, OsfFileInfo],
    refresh: bool,
) -> dict[str, OsfFileInfo]:
    """Return the subset of `files` that must be (re)downloaded.

    A file is selected when it is absent from `bids_dir`. When `refresh` is `True`, a
    cached file is also selected when its md5 in `previous_index` (the index it was last
    downloaded with) differs from—or is absent next to—its md5 in `files` (the freshly
    fetched remote index). The cached index is trusted as the record of what is on disk,
    so files are never re-hashed; a file whose cached entry has no usable md5 (a null
    hash, or a path not previously catalogued) cannot be vouched for, so it is
    re-downloaded and its entry refreshed rather than left in an unverifiable state.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local BIDS root directory where files are cached.
    files : dict[str, OsfFileInfo]
        Requested subset of the remote index.
    previous_index : dict[str, OsfFileInfo]
        Index cached before the refresh, used as the local md5 baseline.
    refresh : bool
        Whether to re-download cached files whose remote md5 differs from the
        cached md5.

    Returns
    -------
    dict[str, OsfFileInfo]
        Subset of `files` to download, preserving the input order.
    """
    selected: dict[str, OsfFileInfo] = {}

    for rel_path, info in files.items():
        dest = bids_dir / rel_path
        if not dest.exists():
            selected[rel_path] = info
            continue
        if not refresh:
            continue
        remote_md5 = info.get("md5")
        local_md5 = previous_index.get(rel_path, {}).get("md5")
        # Re-download when the remote md5 is known and the cached md5 either
        # differs or is unavailable (a null hash, or a path not previously
        # catalogued): it cannot be verified, so refresh rather than assume current.
        if remote_md5 and local_md5 != remote_md5:
            selected[rel_path] = info

    return selected


def download_osf_files(
    bids_dir: Path,
    files: dict[str, OsfFileInfo],
    previous_index: dict[str, OsfFileInfo] | None = None,
    refresh: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """Download OSF files that are missing or whose upstream md5 changed.

    Files absent from `bids_dir` are always downloaded. When `refresh` is
    `True`, cached files whose remote md5 differs from — or is missing next to —
    the md5 recorded in `previous_index` are re-downloaded as well; see
    [`_select_downloads`][confusius.datasets._osf._select_downloads].

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local BIDS root directory where files are cached.
    files : dict[str, OsfFileInfo]
        Requested subset of the remote index, mapping BIDS-relative paths to
        OSF download metadata.
    previous_index : dict[str, OsfFileInfo], optional
        Index cached before the refresh, used as the local md5 baseline. Only
        consulted when `refresh` is `True`; if not provided, an empty baseline
        is used and refresh reduces to downloading missing files.
    refresh : bool, default: False
        Whether to re-download cached files whose remote md5 differs from the
        cached md5.
    progress_callback : Callable[[int, int, str], None], optional
        Callback receiving cumulative downloaded bytes, total bytes to download,
        and a user-facing description. Intended for GUI progress bars.
    """
    to_download = _select_downloads(bids_dir, files, previous_index or {}, refresh)
    if not to_download:
        return

    total_bytes = sum(info["size"] for info in to_download.values())

    with quiet_pooch_logger():
        pooch_logger = pooch.get_logger()
        if progress_callback is not None:
            completed_bytes = 0
            progress_callback(0, total_bytes, "Preparing download...")
            for rel_path, file_info in to_download.items():
                dest = bids_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                description = f"Downloading {Path(rel_path).name}"
                progress_callback(completed_bytes, total_bytes, description)
                adapter = _CallbackProgressAdapter(
                    progress_callback,
                    total_bytes,
                    completed_bytes,
                    description,
                )
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
                completed_bytes += adapter.current_bytes
            progress_callback(total_bytes, total_bytes, "Download complete.")
            return

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
