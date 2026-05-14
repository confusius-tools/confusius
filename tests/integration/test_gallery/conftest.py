"""Shared fixtures for gallery integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest


class GalleryPaths(NamedTuple):
    """Paths used by the gallery builder in tests."""

    examples_root: Path
    built_dir: Path
    cache_root: Path


@pytest.fixture
def gallery_paths(tmp_path: Path) -> GalleryPaths:
    """Set up a minimal ``docs/examples`` tree with default thumb assets."""
    examples_root = tmp_path / "docs" / "examples"
    built_dir = examples_root / "_built"
    cache_root = tmp_path / ".cache" / "gallery"
    examples_root.mkdir(parents=True)
    assets = examples_root / "_assets"
    assets.mkdir()
    (assets / "default_thumb.svg").write_text("<svg/>")
    (assets / "default_thumb_dark.svg").write_text("<svg/>")
    return GalleryPaths(examples_root, built_dir, cache_root)
