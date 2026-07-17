"""Tests for tools.gallery.index."""

from __future__ import annotations

from pathlib import Path

from tools.gallery._types import ExampleSpec, RenderedExample
from tools.gallery.index import build_index


def _spec(source: Path, section: str) -> ExampleSpec:
    return ExampleSpec(
        source=source,
        base_name=source.stem,
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
    # Cards include thumbnail and title links.
    assert "load_autc_thumb_light.png#only-light" in index_md
    assert "load_autc_thumb_dark.png#only-dark" in index_md
    # Falls back to default thumb for examples without one.
    assert "_assets/default_thumb.svg#only-light" in index_md
    assert "_assets/default_thumb_dark.svg#only-dark" in index_md
    # The summary rides along as a hover overlay, only where there is one.
    assert (
        '<span class="examples-card-summary" aria-hidden="true">Quick AUTC demo.</span>'
        in index_md
    )
    assert index_md.count("examples-card-summary") == 2


def test_build_index_preserves_section_order_from_input(tmp_path: Path) -> None:
    """Sections appear in the order they are first seen in the rendered list.

    Discovery sorts section folders by their prefixed name (e.g. ``01_io`` before
    ``02_decomposition``), so the index must preserve that order rather than
    re-sorting alphabetically by the stripped section name.
    """
    src_io = tmp_path / "io" / "load.py"
    src_decomp = tmp_path / "decomposition" / "fastica.py"
    for src in (src_io, src_decomp):
        src.parent.mkdir(parents=True, exist_ok=True)
        src.touch()

    rendered = [
        RenderedExample(
            spec=_spec(src_io, "io"),
            title="Load",
            summary="",
            md_path=src_io.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
        RenderedExample(
            spec=_spec(src_decomp, "decomposition"),
            title="FastICA",
            summary="",
            md_path=src_decomp.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
    ]

    md = build_index(rendered, root=tmp_path)
    assert md.index("## IO") < md.index("## DECOMPOSITION")


def test_build_index_flattens_links_in_card_overlays(tmp_path: Path) -> None:
    """Both link forms in a summary are flattened to their text.

    Overlays are copied verbatim onto the index page at the examples root, where an
    example's relative cross-links (written against its own built page) would not
    resolve and would fail ``zensical build --strict``. A mkdocstrings cross-reference
    resolves, but would render as a link inside an overlay that is inert by design.
    """
    src = tmp_path / "decomposition" / "nmf.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    rendered = [
        RenderedExample(
            spec=_spec(src, "decomposition"),
            title="NMF",
            summary=(
                "Uses [FastICA][confusius.decomposition.FastICA]. Complements the "
                "[PCA](pca_single_recording.md) example."
            ),
            md_path=src.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
    ]
    md = build_index(rendered, root=tmp_path)
    assert "](pca_single_recording.md)" not in md
    assert "confusius.decomposition.FastICA" not in md
    assert "Uses FastICA. Complements the PCA example.</span>" in md


def test_build_index_escapes_html_in_card_overlays(tmp_path: Path) -> None:
    """Summary text is escaped so it cannot inject markup into the overlay span.

    Quotes are left alone: they need no escaping in text content, and an escaped one
    survives a code span, which escapes the entity's ``&`` in turn and shows it
    verbatim.
    """
    src = tmp_path / "io" / "ex.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    rendered = [
        RenderedExample(
            spec=_spec(src, "io"),
            title="Ex",
            summary='Uses <div> & `mode="temporal"`.',
            md_path=src.with_suffix(".md"),
            thumbnail_light=None,
            thumbnail_dark=None,
        ),
    ]
    md = build_index(rendered, root=tmp_path)
    assert 'Uses &lt;div&gt; &amp; `mode="temporal"`.</span>' in md


def test_build_index_demotes_h1_section_intros(tmp_path: Path) -> None:
    """Section intros starting with H1 are demoted to H2 so there's a single page title."""
    src = tmp_path / "io" / "ex.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    spec = ExampleSpec(
        source=src,
        base_name=src.stem,
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
