"""Command line interface: `zallery [--no-run] [--no-cache] [EXAMPLE ...]`.

Run from anywhere inside a project whose `pyproject.toml` has a `[tool.zallery]`
table (see `zallery.config`). With no arguments every example is executed; name
example scripts to run only those. The whole gallery is rendered either way: the
examples not run are restored from cache if present, or rendered without outputs
(code and Markdown only).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from zallery.config import ConfigError, deps_fingerprint, load_config
from zallery.pipeline import build_gallery

BINDER_REF_ENV = "ZALLERY_BINDER_REF"
"""Environment variable giving the git ref stamped into Binder URLs (default `main`)."""


def main(argv: Sequence[str] | None = None) -> int:
    """Build the gallery of the project the working directory belongs to.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command line arguments. If not provided, `sys.argv[1:]` is used.

    Returns
    -------
    int
        Exit status: 0 on success, 1 when the examples directory is missing or an
        argument names no example, 2 when no configuration is found.
    """
    parser = argparse.ArgumentParser(prog="zallery", description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        type=Path,
        help=(
            "Example scripts to run. The whole gallery is still rendered; unnamed "
            "examples are taken from cache if present, or rendered without outputs. "
            "If not provided, all examples are run."
        ),
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help=(
            "Run no example at all: every page is taken from cache if present, or "
            "rendered without outputs. Cannot be combined with example paths."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Execute the selected examples even if a cached result exists, and "
            "replace it. Examples not selected still come from cache."
        ),
    )
    args = parser.parse_args(argv)
    if args.no_run and args.examples:
        parser.error("--no-run cannot be combined with example paths.")

    try:
        config = load_config(Path.cwd())
    except ConfigError as exc:
        print(f"zallery: {exc}", file=sys.stderr)
        return 2
    if not config.examples.is_dir():
        print(f"No examples directory at {config.examples}", file=sys.stderr)
        return 1

    try:
        build_gallery(
            examples_root=config.examples,
            built_dir=config.built,
            cache_root=config.cache,
            deps_fingerprint=deps_fingerprint(config),
            repo_root=config.root,
            binder_repo=config.binder_repo,
            binder_ref=os.environ.get(BINDER_REF_ENV, "main"),
            only=[] if args.no_run else (args.examples or None),
            use_cache=not args.no_cache,
            hook_module=config.hook,
            hook_root=config.root,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0
