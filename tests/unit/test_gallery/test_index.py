"""Tests for tools.gallery.index."""

from __future__ import annotations

from pathlib import Path

from tools.gallery._types import ExampleSpec, RenderedExample
from tools.gallery.index import build_index


def _spec(source: Path, section: str) -> ExampleSpec:
    return ExampleSpec(
        source=source,
        section=section,
        section_intro=f"# {section.upper()}\n\nIntro for {section}.\n",
    )


def test_build_index_groups_cards_by_section(tmp_path: Path) -> None:
    src1 = tmp_path / "io" / "load_autc.py"
    src2 = tmp_path / "io" / "load_nifti.py"
    src3 = tmp_path / "glm" / "first_level.py"
    for src in (src1, src2, src3):
        src.parent.mkdir(parents=True, exist_ok=True)
        src.touch()

    rendered = [
        RenderedExample(
            spec=_spec(src1, "io"),
            title="Load AUTC",
            summary="Quick AUTC demo.",
            md_path=src1.with_suffix(".md"),
            thumbnail_light=src1.parent / "load_autc_thumb_light.png",
            thumbnail_dark=src1.parent / "load_autc_thumb_dark.png",
        ),
        RenderedExample(
            spec=_spec(src2, "io"),
            title="Load NIfTI",
            summary="",
            md_path=src2.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
        RenderedExample(
            spec=_spec(src3, "glm"),
            title="First-level GLM",
            summary="Subject-level GLM.",
            md_path=src3.with_suffix(".md"),
            thumbnail_light=src3.parent / "first_level_thumb_light.png",
            thumbnail_dark=src3.parent / "first_level_thumb_dark.png",
        ),
    ]

    index_md = build_index(rendered, root=tmp_path)

    # Section headers in document order.
    assert "## GLM" in index_md
    assert "## IO" in index_md
    assert "Load AUTC" in index_md
    assert "Load NIfTI" in index_md
    assert "First-level GLM" in index_md
    # Cards include thumbnail and summary where present.
    assert "load_autc_thumb_light.png#only-light" in index_md
    assert "load_autc_thumb_dark.png#only-dark" in index_md
    assert "Quick AUTC demo." in index_md
    # Falls back to default thumb for examples without one.
    assert "_assets/default_thumb.svg#only-light" in index_md
    assert "_assets/default_thumb_dark.svg#only-dark" in index_md


def test_build_index_uses_grid_cards_block(tmp_path: Path) -> None:
    src = tmp_path / "io" / "ex.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    rendered = [
        RenderedExample(
            spec=_spec(src, "io"),
            title="Ex",
            summary="",
            md_path=src.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
    ]
    index_md = build_index(rendered, root=tmp_path)
    assert '<div class="grid cards examples-cards" markdown>' in index_md
    assert "</div>" in index_md


def test_build_index_demotes_h1_section_intros(tmp_path: Path) -> None:
    """Section intros starting with H1 are demoted to H2 so there's a single page title."""
    src = tmp_path / "io" / "ex.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    spec = ExampleSpec(
        source=src,
        section="io",
        section_intro="# Input/Output\n\nIntro paragraph.\n",
    )
    rendered = [
        RenderedExample(
            spec=spec,
            title="Ex",
            summary="",
            md_path=src.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
    ]
    md = build_index(rendered, root=tmp_path)
    # Exactly one H1 (the page title); the section uses H2.
    assert md.count("\n# ") + (1 if md.startswith("# ") else 0) == 1
    assert "## Input/Output" in md
