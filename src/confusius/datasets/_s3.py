"""Anonymous downloads from the AWS Open Data–sponsored dataset collection."""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import quote, urlsplit

import pooch
import requests
from rich.progress import Progress

from confusius.datasets._pooch import (
    _CallbackProgressAdapter,
    _RichProgressAdapter,
    quiet_pooch_logger,
    retrieve_with_retries,
)

if TYPE_CHECKING:
    from collections.abc import Callable

BASE_URL = "https://confusius-datasets.s3.us-west-2.amazonaws.com"
"""Public HTTPS endpoint; downloads do not require AWS credentials."""

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
) -> None:
    """Download selected files, verifying SHA-256 even for existing cache files.

    Parameters
    ----------
    bids_dir : pathlib.Path
        Destination directory.
    files : dict[str, S3FileInfo]
        Selected release entries.
    progress_callback : Callable[[int, int, str], None], optional
        Callback receiving completed bytes, total bytes, and a description.
    """
    # Verify cached bytes too, so interrupted or modified files are repaired.
    pending = {
        relative: info
        for relative, info in files.items()
        if not (bids_dir / relative).is_file()
        or pooch.file_hash(bids_dir / relative, alg="sha256") != info["sha256"]
    }
    if not pending:
        return
    total = sum(info["size"] for info in pending.values())
    completed = 0
    with (
        quiet_pooch_logger(),
        Progress(disable=progress_callback is not None) as progress,
    ):
        task = progress.add_task("Downloading dataset...", total=total)
        for relative, info in pending.items():
            dest = bids_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            description = f"Downloading {dest.name}"
            progress.update(task, description=description)
            adapter: _CallbackProgressAdapter | _RichProgressAdapter
            if progress_callback is not None:
                progress_callback(completed, total, description)
                adapter = _CallbackProgressAdapter(
                    progress_callback, total, completed, description
                )
            else:
                adapter = _RichProgressAdapter(progress, task)
            retrieve_with_retries(
                url=info["url"],
                dest=dest,
                logger=pooch.get_logger(),
                progressbar=adapter,
                on_retry=adapter.rewind,
                known_hash=f"sha256:{info['sha256']}",
            )
            completed += info["size"]
        if progress_callback is not None:
            progress_callback(total, total, "Download complete.")
