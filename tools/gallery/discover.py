"""Discover example source files under ``docs/examples/``."""

from __future__ import annotations

from pathlib import Path

from ._types import ExampleSpec


def discover(root: Path) -> list[ExampleSpec]:
    """Return one example spec per example file."""
    specs: list[ExampleSpec] = []
    if not root.is_dir():
        return specs

    for section_dir in sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
    ):
        intro_path = section_dir / "_section.md"
        intro = intro_path.read_text(encoding="utf-8") if intro_path.is_file() else ""

        for source in sorted(section_dir.glob("*.py")):
            if source.name.startswith("_"):
                continue
            specs.append(
                ExampleSpec(
                    source=source,
                    section=section_dir.name,
                    section_intro=intro,
                )
            )

    return specs
