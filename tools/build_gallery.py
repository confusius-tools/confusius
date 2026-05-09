"""CLI entry point for the examples-gallery builder.

Run as::

    uv run python tools/build_gallery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gallery._pipeline import build_gallery  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = REPO_ROOT / "docs" / "examples"
BUILT_DIR = EXAMPLES_ROOT / "_built"
CACHE_ROOT = REPO_ROOT / ".cache" / "gallery"


def _deps_fingerprint() -> str:
    """Return a string identifying gallery build inputs.

    Uses ``uv.lock`` plus the gallery-builder source files directly. Any change to the
    locked dependencies or to the gallery pipeline forces a cache miss.
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
    if not EXAMPLES_ROOT.is_dir():
        print(f"No examples directory at {EXAMPLES_ROOT}", file=sys.stderr)
        return 1

    build_gallery(
        examples_root=EXAMPLES_ROOT,
        built_dir=BUILT_DIR,
        cache_root=CACHE_ROOT,
        deps_fingerprint=_deps_fingerprint(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
