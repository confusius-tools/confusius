"""Utilities for managing the confusius datasets cache directory."""

from __future__ import annotations

import os
from pathlib import Path

import pooch

_ENV_VAR = "CONFUSIUS_DATA"


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
