"""Tests for tools.gallery.cache."""

from __future__ import annotations

from pathlib import Path

from tools.gallery._types import ExampleSpec
from tools.gallery.cache import cache_key


def _spec(source: Path) -> ExampleSpec:
    return ExampleSpec(source=source, section="io", section_intro="")


def test_cache_key_is_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "ex.py"
    src.write_text("print('hi')\n")
    deps = "confusius==0.2.0\nmatplotlib==3.10.7\n"

    k1 = cache_key(_spec(src), deps_fingerprint=deps, python_version="3.13.1")
    k2 = cache_key(_spec(src), deps_fingerprint=deps, python_version="3.13.1")

    assert k1 == k2
    assert len(k1) == 64  # sha256 hex digest


def test_cache_key_changes_when_source_changes(tmp_path: Path) -> None:
    src = tmp_path / "ex.py"
    src.write_text("print('hi')\n")
    k1 = cache_key(_spec(src), deps_fingerprint="x", python_version="3.13.1")

    src.write_text("print('bye')\n")
    k2 = cache_key(_spec(src), deps_fingerprint="x", python_version="3.13.1")

    assert k1 != k2


def test_cache_key_changes_when_deps_change(tmp_path: Path) -> None:
    src = tmp_path / "ex.py"
    src.write_text("print('hi')\n")

    k1 = cache_key(_spec(src), deps_fingerprint="a", python_version="3.13.1")
    k2 = cache_key(_spec(src), deps_fingerprint="b", python_version="3.13.1")

    assert k1 != k2


def test_cache_key_changes_when_python_version_changes(tmp_path: Path) -> None:
    src = tmp_path / "ex.py"
    src.write_text("print('hi')\n")

    k1 = cache_key(_spec(src), deps_fingerprint="a", python_version="3.13.1")
    k2 = cache_key(_spec(src), deps_fingerprint="a", python_version="3.13.2")

    assert k1 != k2
