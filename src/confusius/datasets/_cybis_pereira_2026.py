"""Fetcher for the Cybis Pereira et al. (2026) fUSI-BIDS dataset."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pooch
import requests
from rich.logging import RichHandler
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

from ._utils import get_datasets_dir

_OSF_PROJECT_ID = "2v6f7"
_BIDS_ROOT = "cybis-pereira-2026-bids"
_INDEX_FILENAME = "dataset_index.json"
_OSF_DOWNLOAD_BASE = "https://osf.io/download/{}/"
_TOTAL_SIZE_BYTES = 12_883_924_421

_MAX_DOWNLOAD_RETRIES = 3
"""Maximum number of attempts per file when OSF returns a transient error."""

_RETRY_BACKOFF_BASE = 2.0
"""Base of the exponential backoff (seconds) between retry attempts."""

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


class _RichProgressAdapter:
    """Adapt a rich Progress task to the tqdm-like interface pooch expects.

    Pooch's `HTTPDownloader` calls `.update(chunk_size)` during streaming,
    then `.reset()` / `.update(total)` / `.close()` after completion.
    This adapter forwards chunk updates to a shared rich task so a single
    progress bar tracks cumulative bytes across multiple file downloads.
    """

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id
        self._file_total = 0
        self._advanced = 0
        self._finalized = False

    @property
    def total(self) -> int:
        return self._file_total

    @total.setter
    def total(self, value: int) -> None:
        self._file_total = value

    def update(self, n: int) -> None:
        if self._finalized:
            return
        # Clamp so cumulative advance never exceeds the file size.
        clamped = min(n, self._file_total - self._advanced)
        if clamped > 0:
            self._advanced += clamped
            self._progress.update(self._task_id, advance=clamped)

    def reset(self) -> None:
        # Pooch calls reset() then update(total) to force 100%. Ignore both
        # by marking the adapter as finalized; close() handles any remainder.
        self._finalized = True

    def close(self) -> None:
        # On success, pooch calls reset() (setting _finalized=True) before
        # close(), so we add any rounding remainder here. On failure, close()
        # runs from pooch's `finally` block without reset() being called first,
        # and we must not advance the shared task for bytes that never arrived.
        if not self._finalized:
            return
        remaining = self._file_total - self._advanced
        if remaining > 0:
            self._progress.update(self._task_id, advance=remaining)
            self._advanced = self._file_total

    def rewind(self) -> None:
        """Undo progress reported so far so a retry doesn't double-count.

        After a failed download, the shared task may have been advanced by
        partial bytes. Call this before retrying so the cumulative bar
        stays aligned with what is actually on disk.
        """
        if self._advanced > 0:
            self._progress.update(self._task_id, advance=-self._advanced)
        self._advanced = 0
        self._finalized = False


def _retrieve_with_retries(
    url: str,
    dest: Path,
    adapter: _RichProgressAdapter,
    logger: logging.Logger,
) -> None:
    """Call `pooch.retrieve`, retrying on transient network errors.

    OSF occasionally closes long-running connections mid-stream, surfacing
    as `requests.exceptions.RequestException` subclasses (ReadTimeout,
    ConnectionError, ChunkedEncodingError). Retry up to
    `_MAX_DOWNLOAD_RETRIES` times with exponential backoff before
    propagating the failure.

    Parameters
    ----------
    url : str
        Direct download URL for the file.
    dest : pathlib.Path
        Target path on disk; used for the filename, parent directory, and
        user-facing log messages.
    adapter : _RichProgressAdapter
        Progress adapter bound to the shared cumulative task. Rewound on
        each failed attempt so byte accounting stays correct.
    logger : logging.Logger
        Logger used to surface retry warnings.
    """
    for attempt in range(1, _MAX_DOWNLOAD_RETRIES + 1):
        try:
            pooch.retrieve(
                url=url,
                known_hash=None,
                fname=dest.name,
                path=dest.parent,
                progressbar=adapter,
            )
            return
        except requests.exceptions.RequestException as exc:
            adapter.rewind()
            if attempt == _MAX_DOWNLOAD_RETRIES:
                raise
            wait = _RETRY_BACKOFF_BASE**attempt
            logger.warning(
                f"Download of {dest.name!r} failed "
                f"(attempt {attempt}/{_MAX_DOWNLOAD_RETRIES}): {exc}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)


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
        If the BIDS root folder or the index file is not found on OSF.
    """
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

    resp = requests.get(folder_url)
    resp.raise_for_status()

    for item in resp.json()["data"]:
        if item["attributes"]["name"] == _INDEX_FILENAME:
            return item["links"]["download"]

    raise RuntimeError(
        f"{_INDEX_FILENAME!r} was not found on OSF (project {_OSF_PROJECT_ID})."
    )


def _get_index(
    data_dir: Path, refresh: bool = False
) -> dict[str, dict[str, str | int]]:
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
    dict[str, dict[str, str | int]]
        Mapping from BIDS-relative file paths to dicts containing
        `"osf_path"` (str) and `"size"` (int, bytes).
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
    index: dict[str, dict[str, str | int]],
    datasets: list[str] | None,
    subjects: list[str] | None,
) -> dict[str, dict[str, str | int]]:
    """Filter the index to files matching the requested datasets and subjects.

    Top-level BIDS metadata files (dataset_description.json, participants.*,
    etc.) are always included.

    Parameters
    ----------
    index : dict[str, dict[str, str | int]]
        Full dataset index as returned by `_get_index`.
    datasets : list[str] or None
        Datasets to include. Use `"rawdata"` for the raw subject data and
        any derivative name (e.g. `"glm-speed"`, `"decode-speed"`) for
        specific derivatives. If `None`, all datasets are included.
    subjects : list[str] or None
        Subject IDs to include (without "sub-" prefix), e.g. `["rat83"]`.
        If `None`, all subjects are included.

    Returns
    -------
    dict[str, dict[str, str | int]]
        Subset of the index matching the filters.
    """
    filtered: dict[str, dict[str, str | int]] = {}

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

        filtered[path] = file_info

    return filtered


def fetch_cybis_pereira_2026(
    data_dir: str | Path | None = None,
    datasets: str | list[str] | None = None,
    subjects: str | list[str] | None = None,
    refresh: bool = False,
) -> Path:
    """Fetch the Cybis Pereira 2026 fUSI-BIDS dataset.

    Downloads functional ultrasound imaging data from freely-moving mice
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
        Subject IDs to download (without "sub-" prefix), e.g. `"rat83"` or
        `["rat83", "rat84"]`. If not provided, all subjects are downloaded.
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
        Cybis Pereira, F. et al. (2026). A Vascular Code for Speed in the
        Spatial Navigation System.

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

    if datasets is not None:
        invalid = set(datasets) - _VALID_DATASETS
        if invalid:
            raise ValueError(
                f"Unknown dataset(s): {invalid}. "
                f"Valid options: {sorted(_VALID_DATASETS)}"
            )

    index = _get_index(bids_dir, refresh=refresh)
    files = _filter_files(index, datasets, subjects)

    missing = {p: info for p, info in files.items() if not (bids_dir / p).exists()}
    if not missing:
        return bids_dir

    # Suppress pooch's INFO-level messages and route warnings/errors through
    # rich so they don't break the progress bar layout.
    pooch_logger = pooch.get_logger()
    original_handlers = pooch_logger.handlers[:]
    original_level = pooch_logger.level

    for handler in original_handlers:
        pooch_logger.removeHandler(handler)
    pooch_logger.addHandler(
        RichHandler(level=logging.WARNING, show_time=False, show_path=False)
    )
    pooch_logger.setLevel(logging.WARNING)

    total_bytes = sum(info["size"] for info in missing.values())

    try:
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
                    description=(f"Downloading [bold]{Path(rel_path).name}[/bold]"),
                )
                adapter = _RichProgressAdapter(progress, task)
                osf_path = file_info["osf_path"]
                _retrieve_with_retries(
                    url=_OSF_DOWNLOAD_BASE.format(osf_path.lstrip("/")),
                    dest=dest,
                    adapter=adapter,
                    logger=pooch_logger,
                )

            progress.update(task, description="Download complete.")
    finally:
        for handler in pooch_logger.handlers[:]:
            pooch_logger.removeHandler(handler)
        for handler in original_handlers:
            pooch_logger.addHandler(handler)
        pooch_logger.setLevel(original_level)

    return bids_dir
