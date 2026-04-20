"""Shared `pooch`-based download helpers for dataset fetchers."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

import pooch
import requests
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from rich.progress import Progress, TaskID

_MAX_DOWNLOAD_RETRIES = 3
"""Maximum number of attempts per file when a transient network error occurs."""

_RETRY_BACKOFF_BASE = 2.0
"""Base of the exponential backoff (seconds) between retry attempts."""


@contextlib.contextmanager
def quiet_pooch_logger() -> Iterator[None]:
    """Redirect `pooch`'s logger through `rich` at WARNING level.

    Suppresses pooch's INFO-level messages (SHA256 suggestions, download
    URLs) and routes warnings/errors through a
    [`rich.logging.RichHandler`][rich.logging.RichHandler] so they don't
    break any active progress-bar layout. Pre-existing handlers and the
    log level are restored on exit.

    Notes
    -----
    Pooch uses `logging.Logger("pooch")` directly rather than
    `logging.getLogger("pooch")`, so we must go through `pooch.get_logger()`.
    """
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
        yield
    finally:
        for handler in pooch_logger.handlers[:]:
            pooch_logger.removeHandler(handler)
        for handler in original_handlers:
            pooch_logger.addHandler(handler)
        pooch_logger.setLevel(original_level)


def retrieve_with_retries(
    url: str,
    dest: Path,
    logger: logging.Logger,
    progressbar: Any = False,
    on_retry: Callable[[], None] | None = None,
) -> None:
    """Download a file with `pooch`, retrying on transient network errors.

    OSF occasionally closes long-running connections mid-stream, surfacing
    as [`requests.exceptions.RequestException`][requests.exceptions.RequestException]
    subclasses (e.g. `ReadTimeout`, `ConnectionError`,
    `ChunkedEncodingError`). Retry up to `_MAX_DOWNLOAD_RETRIES` times
    with exponential backoff before propagating the failure.

    Parameters
    ----------
    url : str
        Direct download URL for the file.
    dest : pathlib.Path
        Target path on disk; used for the filename, parent directory, and
        user-facing log messages.
    logger : logging.Logger
        Logger used to surface retry warnings.
    progressbar : Any, default: False
        Forwarded to `pooch.retrieve`. Accepts `False` or any tqdm-like
        object (e.g. a rich-backed adapter).
    on_retry : Callable[[], None], optional
        Called with no arguments whenever a retry is about to occur.
        Useful for callers that need to rewind shared progress state
        that was advanced by the failed attempt.
    """
    for attempt in range(1, _MAX_DOWNLOAD_RETRIES + 1):
        try:
            pooch.retrieve(
                url=url,
                known_hash=None,
                fname=dest.name,
                path=dest.parent,
                progressbar=progressbar,
            )
            return
        except requests.exceptions.RequestException as exc:
            if on_retry is not None:
                on_retry()
            if attempt == _MAX_DOWNLOAD_RETRIES:
                raise
            wait = _RETRY_BACKOFF_BASE**attempt
            logger.warning(
                f"Download of {dest.name!r} failed "
                f"(attempt {attempt}/{_MAX_DOWNLOAD_RETRIES}): {exc}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)


class _RichProgressAdapter:  # pragma: no cover
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
