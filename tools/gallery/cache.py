"""Cache key + cache directory helpers for the gallery builder."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._types import ExampleSpec


def cache_key(
    spec: ExampleSpec,
    *,
    deps_fingerprint: str,
    python_version: str,
) -> str:
    """Compute a deterministic cache key for one example."""
    digest = hashlib.sha256()
    digest.update(spec.source.read_bytes())
    digest.update(b"\x00")
    digest.update(deps_fingerprint.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(python_version.encode("utf-8"))
    return digest.hexdigest()


def cache_dir(root: Path, key: str) -> Path:
    """Return the cache directory for ``key`` rooted at ``root``."""
    return root / key
