"""Tests for tools.gallery.discover."""

from __future__ import annotations

from pathlib import Path

from tools.gallery.discover import discover


def test_discover_returns_examples_grouped_by_section(gallery_tree: Path) -> None:
    specs = discover(gallery_tree)

    paths = sorted(spec.source.relative_to(gallery_tree).as_posix() for spec in specs)
    assert paths == ["glm/first_level.py", "io/load_autc.py"]


def test_discover_skips_underscore_prefixed_files(gallery_tree: Path) -> None:
    specs = discover(gallery_tree)
    assert all(not spec.source.name.startswith("_") for spec in specs)


def test_discover_attaches_section_metadata(gallery_tree: Path) -> None:
    specs = {spec.source.name: spec for spec in discover(gallery_tree)}

    assert specs["load_autc.py"].section == "io"
    assert specs["load_autc.py"].section_intro.startswith("# Input/Output")
    assert specs["first_level.py"].section == "glm"


def test_discover_no_section_md_uses_empty_intro(gallery_tree: Path) -> None:
    specs = {spec.source.name: spec for spec in discover(gallery_tree)}

    assert specs["first_level.py"].section == "glm"
    assert specs["first_level.py"].section_intro == ""
