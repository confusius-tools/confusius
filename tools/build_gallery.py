"""CLI entry point for the examples-gallery builder.

Run as::

    uv run python tools/build_gallery.py

to run every example, or pass specific example scripts to run only those::

    uv run python tools/build_gallery.py docs/examples/01_io/01_confusius_xarray_101.py

The whole gallery is still rendered either way; the examples you do not name are
restored from cache if present, or rendered without outputs (code only).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the repo root is importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gallery._pipeline import build_gallery  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "docs" / "examples"
BUILT_DIR = EXAMPLES_ROOT / "_built"
CACHE_ROOT = REPO_ROOT / ".gallery-cache"
BINDER_REPO = "confusius-tools/confusius"


def _deps_fingerprint() -> str:
    """Return a string identifying gallery execution/render inputs.

    Uses ``uv.lock`` plus the gallery-builder source files directly. Any change to the
    locked dependencies or gallery pipeline forces a cache miss.

    Binder branch/ref is intentionally excluded so expensive gallery execution can be
    reused across branches.
    """
    parts: list[str] = []
    lockfile = REPO_ROOT / "uv.lock"
    if lockfile.is_file():
        parts.append(lockfile.read_text())

    gallery_root = REPO_ROOT / "tools" / "gallery"
    if gallery_root.is_dir():
        for path in sorted(gallery_root.glob("*.py")):
            parts.append(f"\n# {path.relative_to(REPO_ROOT)}\n")
            parts.append(path.read_text())

    parts.append("\n# tools/build_gallery.py\n")
    parts.append(Path(__file__).read_text())
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
    args = parser.parse_args()

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
            only=args.examples or None,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
