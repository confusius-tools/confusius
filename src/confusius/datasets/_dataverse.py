"""Shared Dataverse stored-zip helpers for dataset fetchers.

The Dataverse Data Access API redirects `/api/access/datafile/{id}` to a presigned
object-store URL that honours HTTP Range requests, so individual members can be
extracted without downloading the whole archive (if dataset is a zipped file). These
helpers wrap [`remotezip.RemoteZip`][remotezip.RemoteZip] to read such an archive's
central directory and pull selected members by range.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pooch
import requests
import urllib3.exceptions
from remotezip import RemoteIOError, RemoteZip
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ._pooch import _MAX_DOWNLOAD_RETRIES, _RETRY_BACKOFF_BASE, quiet_pooch_logger

if TYPE_CHECKING:
    from remotezip import zipfile

_ACCESS_URL = "https://edmond.mpg.de/api/access/datafile/{datafile_id}"
"""URL template for direct download of an Edmond Dataverse datafile by id."""

_INDEX_FILENAME = "dataverse_zip_index.json"
"""Filename of the cached member index for a Dataverse stored-zip dataset."""

_CHUNK_SIZE = 8 * 1024 * 1024
"""Streaming chunk size (bytes) used when extracting a member to disk."""

_RETRIABLE_ERRORS = (
    requests.exceptions.RequestException,
    RemoteIOError,
    urllib3.exceptions.HTTPError,
)
"""Exceptions treated as transient and retried during member extraction.

`remotezip` only wraps the *initial* range request in `RemoteIOError`; bytes
streamed afterwards flow straight through `urllib3`, so a connection dropped
mid-member surfaces as a `urllib3.exceptions.HTTPError` subclass
(`ProtocolError`, `ReadTimeoutError`) that is neither a `requests` error nor a
`RemoteIOError`. All three families are retried.
"""


class ZipMemberInfo(TypedDict):
    """Per-member metadata entry in a Dataverse stored-zip index.

    `crc` is the member's CRC-32 as recorded in the zip central directory. It
    changes whenever the member's content is re-uploaded, so it is the
    change-detection token used on refresh — the Dataverse analog of the md5 an
    OSF-backed index carries.
    """

    size: int
    crc: int


def access_url(datafile_id: int) -> str:
    """Return the Edmond data-access URL for a datafile id.

    Parameters
    ----------
    datafile_id : int
        Numeric Dataverse `dataFile` id (the `fileId` in a `file.xhtml` URL).

    Returns
    -------
    str
        Direct download URL for the datafile.
    """
    return _ACCESS_URL.format(datafile_id=datafile_id)


def get_zip_index(
    data_dir: Path,
    url: str,
    root_prefix: str,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, ZipMemberInfo]:
    """Return the stored-zip member index, preferring a locally cached copy.

    When `refresh` is `False` and a cached index exists in `data_dir`, it is
    decoded and returned directly (offline-friendly). Otherwise the archive's
    central directory is read over the network with a single range-backed
    [`remotezip.RemoteZip`][remotezip.RemoteZip] session.

    The freshly read index is only persisted here on the very first fetch (when
    no cache exists yet). On a refresh of an existing cache it is returned but
    not written: the caller merges it via
    [`update_cached_zip_index`][confusius.datasets._dataverse.update_cached_zip_index]
    so members it did not reconcile keep the crc they were downloaded with.

    Only members whose names start with `root_prefix` are indexed; directory
    entries are skipped. Index keys are the member paths with `root_prefix`
    stripped, so they double as cache-relative paths.

    Parameters
    ----------
    data_dir : pathlib.Path
        Local directory in which the index is cached.
    url : str
        Data-access URL of the zip datafile (see
        [`access_url`][confusius.datasets._dataverse.access_url]).
    root_prefix : str
        Common in-zip path prefix shared by every member of interest, e.g.
        `"naked_mole_rat_fusi_dataset/"`. It is stripped from index keys.
    refresh : bool, default: False
        Whether to re-read the central directory from the network even if a
        cached index exists.
    session : requests.Session, optional
        Session reused for the range requests so the central-directory read
        shares pooled keep-alive connections with the subsequent download. If
        not provided, `remotezip` opens unpooled connections.

    Returns
    -------
    dict[str, ZipMemberInfo]
        Mapping from prefix-stripped member paths to member metadata.

    Raises
    ------
    RuntimeError
        If no archive member sits under `root_prefix` (the archive layout
        changed or `root_prefix` is wrong), which would otherwise cache an
        empty index and turn every fetch into a silent no-op.
    """
    index_path = data_dir / _INDEX_FILENAME
    cache_exists = index_path.exists()
    if not refresh and cache_exists:
        return read_cached_zip_index(data_dir)

    def _filter_index(info: zipfile.ZipInfo) -> bool:
        return not info.is_dir() and info.filename.startswith(root_prefix)

    def _map_index(info: zipfile.ZipInfo) -> tuple[str, ZipMemberInfo]:
        return info.filename[len(root_prefix) :], {
            "size": info.file_size,
            "crc": info.CRC,
        }

    index = dict(
        map(
            _map_index,
            filter(_filter_index, RemoteZip(url, session=session).infolist()),
        )
    )

    if not index:
        raise RuntimeError(
            f"No archive members found under {root_prefix!r} in {url}. "
            "The archive layout may have changed."
        )

    # First fetch: no cached crcs to preserve, so persist the full index as the
    # baseline. On a refresh of an existing cache the caller merges instead (see
    # update_cached_zip_index), so leave the cache untouched here.
    if not cache_exists:
        _write_zip_index(index_path, index)
    return index


def _write_zip_index(index_path: Path, index: dict[str, ZipMemberInfo]) -> None:
    """Write a stored-zip member index to disk as sorted, indented JSON.

    Parameters
    ----------
    index_path : pathlib.Path
        Destination path for the cached index.
    index : dict[str, ZipMemberInfo]
        Index mapping to serialise.
    """
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_cached_zip_index(data_dir: Path) -> dict[str, ZipMemberInfo]:
    """Return the locally cached stored-zip index, or `{}` if none exists.

    Unlike [`get_zip_index`][confusius.datasets._dataverse.get_zip_index] this
    never touches the network: it only decodes the on-disk index. Callers read
    it before a refresh to capture the crcs the cached members were downloaded
    with, then compare those against the freshly read central directory to
    decide which members changed upstream.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory holding the cached index.

    Returns
    -------
    dict[str, ZipMemberInfo]
        Cached index mapping, or an empty mapping when no cache exists.
    """
    index_path = data_dir / _INDEX_FILENAME
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def update_cached_zip_index(
    data_dir: Path,
    remote_index: dict[str, ZipMemberInfo],
    previous_index: dict[str, ZipMemberInfo],
    members: dict[str, ZipMemberInfo],
) -> None:
    """Persist a refreshed index by merging, not replacing, the cached one.

    The cached index is the record of what is on disk, so it is not overwritten
    with the freshly read central directory. Instead the persisted index is:

    - the remote entry for every reconciled member (those in `members`), whose
      crc baseline is advanced to the current upstream value;
    - the cached entry for any member the call did not reconcile, so that a
      later refresh can still detect its crc changing upstream;
    - the remote entry for members present neither in the cache nor in the
      request (newly added upstream), so they are catalogued for next time.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory holding the cached index.
    remote_index : dict[str, ZipMemberInfo]
        Full index freshly read from the archive central directory.
    previous_index : dict[str, ZipMemberInfo]
        Index cached before the refresh.
    members : dict[str, ZipMemberInfo]
        Requested subset of `remote_index` that was reconciled with disk.
    """
    merged: dict[str, ZipMemberInfo] = {**remote_index, **previous_index, **members}
    _write_zip_index(data_dir / _INDEX_FILENAME, merged)


def _select_zip_downloads(
    bids_dir: Path,
    members: dict[str, ZipMemberInfo],
    previous_index: dict[str, ZipMemberInfo],
    refresh: bool,
) -> dict[str, ZipMemberInfo]:
    """Return the subset of `members` that must be (re)downloaded.

    A member is selected when it is absent from `bids_dir`. When `refresh` is
    `True`, a cached member is also selected when its crc in `previous_index`
    (the index it was last downloaded with) differs from its crc in `members`
    (the freshly read central directory). The cached index is trusted as the
    record of what is on disk, so members are never re-hashed; a member whose
    cached entry has no crc (a path not previously catalogued) cannot be
    vouched for, so it is re-downloaded rather than left in an unverifiable
    state.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local cache root where members are written.
    members : dict[str, ZipMemberInfo]
        Requested subset of the remote index.
    previous_index : dict[str, ZipMemberInfo]
        Index cached before the refresh, used as the local crc baseline.
    refresh : bool
        Whether to re-download cached members whose remote crc differs from the
        cached crc.

    Returns
    -------
    dict[str, ZipMemberInfo]
        Subset of `members` to download, preserving the input order.
    """
    selected: dict[str, ZipMemberInfo] = {}

    for rel, info in members.items():
        if not (bids_dir / rel).exists():
            selected[rel] = info
            continue
        if not refresh:
            continue
        if previous_index.get(rel, {}).get("crc") != info["crc"]:
            selected[rel] = info

    return selected


def download_zip_members(
    bids_dir: Path,
    url: str,
    members: dict[str, ZipMemberInfo],
    root_prefix: str,
    previous_index: dict[str, ZipMemberInfo] | None = None,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> None:
    """Extract stored-zip members that are missing or changed upstream.

    Members absent from `bids_dir` are always downloaded. When `refresh` is
    `True`, cached members whose remote crc differs from the crc recorded in
    `previous_index` are re-downloaded as well; see
    [`_select_zip_downloads`][confusius.datasets._dataverse._select_zip_downloads].
    Selected members are streamed out of the remote archive by range request
    through a single [`remotezip.RemoteZip`][remotezip.RemoteZip] session, with
    a shared progress bar tracking cumulative (uncompressed) bytes. Each member
    is written to a `.part` file and atomically renamed on success.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Local cache root where members are written (using their
        prefix-stripped relative paths).
    url : str
        Data-access URL of the zip datafile (see
        [`access_url`][confusius.datasets._dataverse.access_url]).
    members : dict[str, ZipMemberInfo]
        Mapping from prefix-stripped member paths to member metadata, as
        returned by [`get_zip_index`][confusius.datasets._dataverse.get_zip_index].
    root_prefix : str
        Common in-zip path prefix to prepend when reading a member by its
        stripped key.
    previous_index : dict[str, ZipMemberInfo], optional
        Index cached before the refresh, used as the local crc baseline. Only
        consulted when `refresh` is `True`; if not provided, an empty baseline
        is used and refresh reduces to downloading missing members.
    refresh : bool, default: False
        Whether to re-download cached members whose remote crc differs from the
        cached crc.
    session : requests.Session, optional
        Session reused for every member's range requests so the hundreds of
        per-member reads share pooled keep-alive connections instead of
        re-walking the access redirect and TLS handshake each time. If not
        provided, `remotezip` opens unpooled connections.

    Raises
    ------
    ValueError
        If a member's relative path escapes `bids_dir` (zip-slip), e.g. a
        member name containing `..`.
    """
    to_download = _select_zip_downloads(
        bids_dir, members, previous_index or {}, refresh
    )
    if not to_download:
        return

    bids_root = bids_dir.resolve()
    total_bytes = sum(info["size"] for info in to_download.values())

    with quiet_pooch_logger():
        logger = pooch.get_logger()
        with (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress,
            RemoteZip(url, session=session) as archive,
        ):
            task = progress.add_task("Downloading dataset...", total=total_bytes)
            for rel in to_download:
                dest = bids_dir / rel
                # Guard against zip-slip: a member path with `..` could resolve
                # outside the cache root. `RemoteZip.open` is the unsanitised
                # `zipfile.ZipFile.open`, so we check before creating anything.
                if not dest.resolve().is_relative_to(bids_root):
                    raise ValueError(
                        f"Refusing to extract member outside the cache "
                        f"directory: {rel!r}."
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                progress.update(
                    task, description=f"Downloading [bold]{Path(rel).name}[/bold]"
                )
                _extract_member_with_retries(
                    archive,
                    root_prefix + rel,
                    dest,
                    progress,
                    task,
                    logger,
                )
            progress.update(task, description="Download complete.")


def _extract_member_with_retries(
    archive: RemoteZip,
    member: str,
    dest: Path,
    progress: Progress,
    task_id: TaskID,
    logger: logging.Logger,
) -> None:
    """Stream one member to disk, retrying on transient network errors.

    The member is read in [`_CHUNK_SIZE`][confusius.datasets._dataverse._CHUNK_SIZE]
    blocks into a sibling `.part` file and renamed onto `dest` on success.
    Transient range-read failures
    ([`_RETRIABLE_ERRORS`][confusius.datasets._dataverse._RETRIABLE_ERRORS]) are
    retried up to
    [`_MAX_DOWNLOAD_RETRIES`][confusius.datasets._pooch._MAX_DOWNLOAD_RETRIES]
    times with exponential backoff, rewinding the shared progress bar by the
    bytes advanced in the failed attempt. The `.part` file is removed on any
    failure (retriable or not) so partial writes never linger in the cache.

    Parameters
    ----------
    archive : remotezip.RemoteZip
        Open remote archive to read the member from.
    member : str
        Full in-zip member name (including the root prefix).
    dest : pathlib.Path
        Destination path for the extracted member.
    progress : rich.progress.Progress
        Progress instance owning the shared download task.
    task_id : rich.progress.TaskID
        Identifier of the shared download task to advance.
    logger : logging.Logger
        Logger used to surface retry warnings.

    Raises
    ------
    requests.exceptions.RequestException
        If a network error persists after the retry budget is exhausted.
    remotezip.RemoteIOError
        If a range-read error persists after the retry budget is exhausted.
    urllib3.exceptions.HTTPError
        If a mid-stream connection error persists after the retry budget is
        exhausted.
    """
    part = dest.with_name(dest.name + ".part")
    try:
        for attempt in range(1, _MAX_DOWNLOAD_RETRIES + 1):
            advanced = 0
            try:
                with archive.open(member) as src, part.open("wb") as out:
                    while chunk := src.read(_CHUNK_SIZE):
                        out.write(chunk)
                        advanced += len(chunk)
                        progress.update(task_id, advance=len(chunk))
                part.replace(dest)
                return
            except _RETRIABLE_ERRORS as exc:
                progress.update(task_id, advance=-advanced)
                if attempt == _MAX_DOWNLOAD_RETRIES:
                    raise
                wait = _RETRY_BACKOFF_BASE**attempt
                logger.warning(
                    f"Download of {dest.name!r} failed "
                    f"(attempt {attempt}/{_MAX_DOWNLOAD_RETRIES}): {exc}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
    finally:
        # Clean up the partial file on any failure path (retriable exhausted,
        # or a non-retriable error such as an OSError mid-write). On success
        # `part` was already renamed onto `dest`, so this is a no-op.
        part.unlink(missing_ok=True)
