"""Discover example source files under ``docs/examples/``."""

from __future__ import annotations

import re
from pathlib import Path

from ._types import ExampleSpec

# Strip leading "01_", "02_" etc. from section directory names and example file
# names so the built output uses clean names ("io", "decomposition",
# "pca_single_recording") while the source paths can be prefixed for explicit
# ordering.
_NUMERIC_PREFIX_RE = re.compile(r"^\d+_")


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
        section_name = _NUMERIC_PREFIX_RE.sub("", section_dir.name)

        for source in sorted(section_dir.glob("*.py")):
            if source.name.startswith("_"):
                continue
            specs.append(
                ExampleSpec(
                    source=source,
                    base_name=_NUMERIC_PREFIX_RE.sub("", source.stem),
                    section=section_name,
                    section_intro=intro,
                )
            )

    return specs
