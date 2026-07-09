"""Utilities for managing the confusius datasets cache directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pooch
from rich.console import Console
from rich.text import Text
from rich.theme import Theme

_ENV_VAR = "CONFUSIUS_DATA"

CONFUSIUS_TURQUOISE = "#3ad9a4"
"""Confusius brand turquoise, matching the docs theme."""

CONFUSIUS_RED = "#e94b5f"
"""Confusius brand red, matching the docs theme."""

_CITATION_THEME = Theme(
    {"citation.title": CONFUSIUS_RED, "citation.doi": CONFUSIUS_TURQUOISE}
)
"""Maps the citation style names used in `_CITATION` markup to brand colors."""

_console = Console(theme=_CITATION_THEME)


def print_citation_message(citation: str, kind: Literal["dataset", "template"]) -> None:
    """Print a citation prompt for a fetched dataset or template.

    Parameters
    ----------
    citation : str
        Citation text to print. Rich markup (e.g. `[italic]`, `[citation.title]`)
        is rendered.
    kind : {"dataset", "template"}
        Resource kind used in the prompt.
    """
    _console.print(
        f"If you use this {kind} in your work, please cite the following source:\n"
    )
    # Render as a Text renderable rather than a str so the markup styles apply but
    # rich's auto-highlighter does not colorize numbers, URLs, etc.
    text = Text.from_markup(citation)
    # Bold the whole citation; the rest of the markup is in each _CITATION string.
    text.stylize("bold")
    _console.print(text)


def plain_citation(citation: str) -> str:
    """Strip rich markup tags from a citation.

    Parameters
    ----------
    citation : str
        Citation text, possibly containing rich markup tags (e.g. `[italic]`).

    Returns
    -------
    str
        The citation with all markup tags removed, leaving only the visible text.
    """
    return Text.from_markup(citation).plain


def get_datasets_dir(data_dir: str | Path | None = None) -> Path:
    """Return the confusius data directory.

    Priority order:

    1. The `data_dir` argument.
    2. The `CONFUSIUS_DATA` environment variable.
    3. The platform cache directory (e.g. `~/.cache/confusius` on Linux).

    Parameters
    ----------
    data_dir : str or pathlib.Path, optional
        Custom data directory. If not provided, falls back to the environment variable
        or the platform cache.

    Returns
    -------
    pathlib.Path
        Resolved data directory (created if it does not exist).
    """
    if data_dir is not None:
        path = Path(data_dir)
    elif _ENV_VAR in os.environ:
        path = Path(os.environ[_ENV_VAR])
    else:
        path = Path(pooch.os_cache("confusius"))

    path.mkdir(parents=True, exist_ok=True)
    return path
