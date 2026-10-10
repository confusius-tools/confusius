"""Shared `pooch` logging helpers for dataset fetchers."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import pooch
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Generator


@contextlib.contextmanager
def quiet_pooch_logger() -> Generator[None, None, None]:
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
