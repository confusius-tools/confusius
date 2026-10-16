"""CLI entry point for the examples-gallery builder.

Run as::

    uv run python tools/build_gallery.py

to run every example, or pass specific example scripts to run only those::

    uv run python tools/build_gallery.py docs/examples/01_io/01_confusius_xarray_101.py

The whole gallery is still rendered either way; the examples you do not name are
restored from cache if present, or rendered without outputs (code only). To run
nothing at all and only render, pass ``--no-run``::

    uv run python tools/build_gallery.py --no-run

To re-execute examples that already have a cached result, pass ``--no-cache``::

    uv run python tools/build_gallery.py --no-cache docs/examples/01_io/01_confusius_xarray_101.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import zallery
from zallery import build_gallery

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "docs" / "examples"
BUILT_DIR = EXAMPLES_ROOT / "_built"
CACHE_ROOT = REPO_ROOT / ".gallery-cache"
BINDER_REPO = "confusius-tools/confusius"
HOOK_MODULE = "tools.gallery_hooks"
"""Module with this project's `gallery_retheme` hook; `REPO_ROOT` makes it importable."""


def _deps_fingerprint() -> str:
    """Return a string identifying gallery execution/render inputs.

    Uses `uv.lock`, the `zallery` sources, the project hook and this script. Any
    change to the locked dependencies or the gallery pipeline forces a cache miss.

    Binder branch/ref is intentionally excluded so expensive gallery execution can be
    reused across branches.
    """
    # Source text must not depend on the platform's locale (e.g. Windows cp1252).
    parts: list[str] = []
    lockfile = REPO_ROOT / "uv.lock"
    if lockfile.is_file():
        parts.append(lockfile.read_text(encoding="utf-8"))

    zallery_root = Path(zallery.__file__).resolve().parent
    for path in sorted(zallery_root.glob("*.py")):
        parts.append(f"\n# zallery/{path.name}\n")
        parts.append(path.read_text(encoding="utf-8"))

    parts.append("\n# tools/gallery_hooks.py\n")
    parts.append((REPO_ROOT / "tools" / "gallery_hooks.py").read_text(encoding="utf-8"))

    parts.append("\n# tools/build_gallery.py\n")
    parts.append(Path(__file__).read_text(encoding="utf-8"))
    return "".join(parts)


def main() -> int:
    """Run the gallery builder end-to-end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        type=Path,
        help=(
            "Example scripts to run (paths under docs/examples/). The whole gallery "
            "is still rendered; unnamed examples are taken from cache if present, or "
            "rendered without outputs. If not provided, all examples are run."
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
    args = parser.parse_args()
    if args.no_run and args.examples:
        parser.error("--no-run cannot be combined with example paths.")

    if not EXAMPLES_ROOT.is_dir():
        print(f"No examples directory at {EXAMPLES_ROOT}", file=sys.stderr)
        return 1

    binder_ref = os.environ.get("CONFUSIUS_BINDER_REF", "main")
    try:
        build_gallery(
            examples_root=EXAMPLES_ROOT,
            built_dir=BUILT_DIR,
            cache_root=CACHE_ROOT,
            deps_fingerprint=_deps_fingerprint(),
            repo_root=REPO_ROOT,
            binder_repo=BINDER_REPO,
            binder_ref=binder_ref,
            only=[] if args.no_run else (args.examples or None),
            use_cache=not args.no_cache,
            hook_module=HOOK_MODULE,
            hook_root=REPO_ROOT,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
