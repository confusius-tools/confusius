"""Anonymous downloads from the ConfUSIus dataset collection."""

from __future__ import annotations

import configparser
import json
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import TYPE_CHECKING, TypedDict, cast
from urllib.parse import quote, unquote, urlsplit

import boto3
import pooch
import requests
from boto3.s3.transfer import S3Transfer, TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from rich.progress import Progress

from confusius.datasets._pooch import quiet_pooch_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from pooch.typing import Downloader

BASE_URL = "https://confusius-datasets.s3.us-west-2.amazonaws.com"
"""Public HTTPS endpoint; downloads do not require AWS credentials."""

_BUCKET = "confusius-datasets"
"""S3 bucket containing the public release collection."""

_MAX_CONCURRENT_REQUESTS = 10
"""Shared transfer-request worker limit and maximum concurrent file jobs."""

_INDEX_FILENAME = "s3_index.json"
"""Local inventory recording the selected immutable release."""


class S3FileInfo(TypedDict):
    """Release file URL, size in bytes, and SHA-256 digest."""

    url: str
    size: int
    sha256: str


def read_cached_index(data_dir: Path) -> dict[str, S3FileInfo]:
    """Read locally recorded download metadata without network access.

    Parameters
    ----------
    data_dir : pathlib.Path
        Cache directory.

    Returns
    -------
    dict[str, S3FileInfo]
        Cached entries, or an empty mapping when no cache exists.

    Raises
    ------
    ValueError
        If the cached inventory is malformed or contains unsafe paths.
    """
    path = data_dir / _INDEX_FILENAME
    if not path.exists():
        return {}
    index = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(index)
    if any(not isinstance(info.get("url"), str) for info in index.values()):
        raise ValueError("Cached inventory must define a URL for each file.")
    return index


def get_index(
    data_dir: Path, category: str, name: str, refresh: bool = False
) -> dict[str, S3FileInfo]:
    """Resolve the latest published release and validate its manifest.

    Parameters
    ----------
    data_dir : pathlib.Path
        Cache directory.
    category : str
        Catalog section, `datasets` or `templates`.
    name : str
        Recipe name in the release catalog.
    refresh : bool, default: False
        Whether to query the catalog instead of using local metadata.

    Returns
    -------
    dict[str, S3FileInfo]
        Relative paths mapped to anonymous download URLs, sizes, and hashes.

    Raises
    ------
    RuntimeError
        If the recipe has no published release.
    ValueError
        If the catalog version or manifest is unsafe or malformed.
    requests.HTTPError
        If downloading the catalog or manifest fails.
    """
    cached = read_cached_index(data_dir) if not refresh else {}
    if cached and not refresh:
        versions = set()
        for relative, info in cached.items():
            match = re.fullmatch(
                re.escape(f"{BASE_URL}/{category}/{name}/")
                + r"([A-Za-z0-9][A-Za-z0-9._+-]*)/"
                + re.escape(quote(relative, safe="/")),
                info["url"],
            )
            if match is None:
                raise ValueError("Cached inventory contains an invalid release URL.")
            versions.add(match.group(1))
        if len(versions) != 1:
            raise ValueError("Cached inventory mixes release versions.")
        return cached
    response = requests.get(f"{BASE_URL}/last_versions.conf", timeout=30)
    response.raise_for_status()
    catalog = configparser.ConfigParser(interpolation=None)
    catalog.read_string(response.text)
    if not catalog.has_option(category, name):
        raise RuntimeError(f"No published S3 release for {category}/{name}.")
    version = catalog.get(category, name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version):
        raise ValueError(f"Invalid release version: {version!r}.")
    base = f"{BASE_URL}/{category}/{name}/{version}"
    response = requests.get(f"{base}/manifest.json", timeout=30)
    response.raise_for_status()
    manifest = response.json()
    _validate_manifest(manifest)
    return {
        relative: S3FileInfo(
            url=f"{base}/{quote(relative, safe='/')}",
            size=info["size"],
            sha256=info["sha256"],
        )
        for relative, info in manifest.items()
    }


def _validate_manifest(manifest: object) -> None:
    """Reject unsafe paths and malformed release file metadata.

    Parameters
    ----------
    manifest : object
        Decoded remote manifest or cached inventory.

    Raises
    ------
    ValueError
        If paths, sizes, or SHA-256 hashes are invalid.
    """
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Release manifest must be a nonempty mapping.")
    for relative, info in manifest.items():
        if (
            not isinstance(relative, str)
            or not relative
            or relative in {".", "manifest.json"}
            or PurePosixPath(relative).as_posix() != relative
            or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive
            or ".." in relative.split("/")
            or "\\" in relative
            or not isinstance(info, dict)
            or type(info.get("size")) is not int
            or info["size"] < 0
            or not isinstance(info.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
        ):
            raise ValueError(f"Invalid release manifest entry: {relative!r}.")


def get_release_dir(data_dir: Path, index: dict[str, S3FileInfo]) -> Path:
    """Create a version-isolated local release directory.

    Parameters
    ----------
    data_dir : pathlib.Path
        Cache root for one recipe.
    index : dict[str, S3FileInfo]
        Resolved release inventory.

    Returns
    -------
    pathlib.Path
        Directory for this immutable release version.
    """
    version = urlsplit(next(iter(index.values()))["url"]).path.split("/")[3]
    destination = data_dir / version
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def update_cached_index(
    data_dir: Path,
    remote_index: dict[str, S3FileInfo],
) -> None:
    """Record the release inventory after successful downloads.

    Parameters
    ----------
    data_dir : pathlib.Path
        Cache directory.
    remote_index : dict[str, S3FileInfo]
        Full resolved release inventory.
    """
    path = data_dir / _INDEX_FILENAME
    text = json.dumps(remote_index, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=data_dir, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_s3_files(
    bids_dir: Path,
    files: dict[str, S3FileInfo],
    progress_callback: Callable[[int, int, str], None] | None = None,
    refresh: bool = False,
) -> None:
    """Download missing or upstream-changed files, verifying their SHA-256.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Destination directory.
    files : dict[str, S3FileInfo]
        Selected release entries.
    progress_callback : Callable[[int, int, str], None], optional
        Callback receiving completed bytes, total bytes, and a description.
    refresh : bool, default: False
        Whether to compare the remote SHA-256 with the recorded cached hash.
        Existing files are trusted without re-reading their contents when the
        hashes match. Invalid cached metadata is treated as an unknown hash.

    Raises
    ------
    ValueError
        If a downloaded SHA-256 does not match the manifest.
    botocore.exceptions.BotoCoreError
        If a transfer fails.
    botocore.exceptions.ClientError
        If S3 rejects a request.
    boto3.exceptions.RetriesExceededError
        If streaming failures exhaust the download retry budget.
    """
    previous_index: dict[str, S3FileInfo] = {}
    if refresh:
        try:
            previous_index = read_cached_index(bids_dir.parent)
        except ValueError:
            # Explicit refresh can repair malformed cached metadata.
            pass
    pending = {
        relative: info
        for relative, info in files.items()
        if not (bids_dir / relative).is_file()
        or (
            refresh and previous_index.get(relative, {}).get("sha256") != info["sha256"]
        )
    }
    if not pending:
        return
    total = sum(info["size"] for info in pending.values())
    downloaded = dict.fromkeys(pending, 0)
    lock = Lock()

    def advance(relative: str, byte_count: int) -> None:
        """Record transfer bytes, including negative retry adjustments.

        Parameters
        ----------
        relative : str
            File path within the release.
        byte_count : int
            Bytes transferred, or a negative value when a range is retried.
        """
        with lock:
            downloaded[relative] = min(
                pending[relative]["size"],
                max(0, downloaded[relative] + byte_count),
            )

    if progress_callback is not None:
        progress_callback(0, total, "Preparing download...")
    # Boto3 expects a service endpoint, not the bucket-prefixed HTTPS endpoint.
    endpoint = BASE_URL.replace(f"//{_BUCKET}.", "//", 1)
    client = boto3.session.Session().client(
        "s3",
        region_name="us-west-2",
        endpoint_url=endpoint,
        config=Config(
            signature_version=UNSIGNED,
            s3={"addressing_style": "path"},
            max_pool_connections=_MAX_CONCURRENT_REQUESTS,
            connect_timeout=10,
            read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 3},
        ),
    )
    with (
        quiet_pooch_logger(),
        Progress(disable=progress_callback is not None) as progress,
        closing(client),
        S3Transfer(
            client,
            TransferConfig(
                max_concurrency=_MAX_CONCURRENT_REQUESTS,
                multipart_threshold=8 * 1024**2,
                multipart_chunksize=8 * 1024**2,
                num_download_attempts=3,
                # Keep the shared request-worker limit predictable across machines.
                preferred_transfer_client="classic",
            ),
        ) as transfer,
        ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_REQUESTS) as executor,
    ):
        task = progress.add_task("Downloading dataset...", total=total)
        futures = {
            executor.submit(
                _retrieve_s3_file,
                bids_dir / relative,
                info,
                transfer,
                partial(advance, relative),
            ): relative
            for relative, info in pending.items()
        }
        try:
            while futures:
                done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    relative = futures.pop(future)
                    with lock:
                        downloaded[relative] = pending[relative]["size"]
                with lock:
                    completed = sum(downloaded.values())
                progress.update(task, completed=completed)
                if progress_callback is not None:
                    progress_callback(completed, total, "Downloading dataset...")
        finally:
            # Queued jobs must not keep downloading after another file fails.
            for future in futures:
                future.cancel()
        if progress_callback is not None:
            progress_callback(total, total, "Download complete.")


def _retrieve_s3_file(
    dest: Path,
    info: S3FileInfo,
    transfer: S3Transfer,
    advance: Callable[[int], None],
) -> None:
    """Verify and stage one file using the shared AWS transfer manager.

    Parameters
    ----------
    dest : pathlib.Path
        Final local file path.
    info : S3FileInfo
        Remote URL, size, and SHA-256 digest.
    transfer : boto3.s3.transfer.S3Transfer
        Batch-scoped managed S3 transfer engine.
    advance : Callable[[int], None]
        Thread-safe byte-progress recorder for this file.

    Raises
    ------
    ValueError
        If the downloaded SHA-256 does not match the manifest.
    botocore.exceptions.BotoCoreError
        If the S3 transfer fails.
    botocore.exceptions.ClientError
        If S3 rejects the request.
    boto3.exceptions.RetriesExceededError
        If streaming failures exhaust the download retry budget.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def download(url: str, output_file: str, pooch: pooch.Pooch | None) -> None:
        """Download into Pooch's temporary path without accepting cached bytes.

        Parameters
        ----------
        url : str
            Anonymous object URL from the release manifest.
        output_file : str
            Temporary download destination supplied by Pooch.
        pooch : pooch.Pooch, optional
            Calling Pooch instance; unused by this downloader.

        Raises
        ------
        botocore.exceptions.BotoCoreError
            If the S3 transfer fails.
        botocore.exceptions.ClientError
            If S3 rejects the request.
        boto3.exceptions.RetriesExceededError
            If streaming failures exhaust the download retry budget.
        """
        transfer.download_file(
            _BUCKET,
            unquote(urlsplit(url).path.lstrip("/")),
            output_file,
            callback=advance,
        )

    pooch.retrieve(
        url=info["url"],
        fname=dest.name,
        path=dest.parent,
        known_hash=f"sha256:{info['sha256']}",
        progressbar=False,
        # Pooch's protocol names differ from its documented downloader signature.
        downloader=cast("Downloader", download),
    )
