"""Utilities for managing the confusius datasets cache directory."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import pooch
from rich import print as rich_print
from rich.text import Text

_ENV_VAR = "CONFUSIUS_DATA"

_DOI_URL_RE = re.compile(r"https://doi\.org/\S+")
"""Pattern matching a DOI URL, rendered as a clickable link in the citation."""


def print_citation_message(citation: str, kind: Literal["dataset", "template"]) -> None:
    """Print a citation prompt for a fetched dataset or template.

    Parameters
    ----------
    citation : str
        Citation text to print. Rich markup (e.g. `[italic]`) is rendered.
    kind : {"dataset", "template"}
        Resource kind used in the prompt.
    """
    print(f"If you use this {kind} in your work, please cite the following source:\n")
    # Render as a Text renderable rather than a str so the markup styles apply but
    # rich's auto-highlighter does not colorize numbers, URLs, etc.
    text = Text.from_markup(citation)
    # Turn the DOI into an OSC 8 hyperlink where the terminal supports it.
    match = _DOI_URL_RE.search(text.plain)
    if match:
        text.stylize(f"link {match.group()}", match.start(), match.end())
    rich_print(text)


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
