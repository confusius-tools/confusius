"""Generate the gallery index page."""

from __future__ import annotations

import html as html_lib
import re
from collections import defaultdict
from pathlib import Path

from ._types import RenderedExample

_DEFAULT_THUMB_LIGHT = "_assets/default_thumb.svg"
_DEFAULT_THUMB_DARK = "_assets/default_thumb_dark.svg"

# Markdown inline link: [text](url). Card overlays are copied verbatim onto the gallery
# index page (at the examples root), where an example's relative links — written against
# its own built page — would not resolve and fail `zensical build --strict`. The full
# links still render on the example's own page.
_INLINE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _flatten_links(text: str) -> str:
    """Replace Markdown inline links with their visible text.

    Parameters
    ----------
    text : str
        Markdown text that may contain inline links.

    Returns
    -------
    str
        The text with every `[label](target)` reduced to `label`.
    """
    return _INLINE_LINK.sub(r"\1", text)


def _demote_h1(text: str) -> str:
    """Demote a leading H1 heading to H2 so the index has a single page title."""
    lines = text.split("\n", 1)
    first = lines[0]
    # "# " matches only true H1s; "## " starts with "#" not "# ", so no extra guard needed.
    if first.startswith("# "):
        first = "#" + first
    return first + ("\n" + lines[1] if len(lines) > 1 else "")


def _card_image_markdown(rex: RenderedExample, *, root: Path, href: str) -> str:
    """Return theme-aware thumbnail markup for one example card."""
    if rex.thumbnail_light is None or rex.thumbnail_dark is None:
        return (
            f'[<img class="skip-lightbox" src="{_DEFAULT_THUMB_LIGHT}#only-light" alt="Example thumbnail">'
            f'<img class="skip-lightbox" src="{_DEFAULT_THUMB_DARK}#only-dark" alt="Example thumbnail">]({href})'
        )

    light = rex.thumbnail_light.relative_to(root).as_posix()
    dark = rex.thumbnail_dark.relative_to(root).as_posix()
    alt = html_lib.escape(rex.title, quote=True)
    return (
        f'[<img class="skip-lightbox" src="{light}#only-light" alt="{alt}">'
        f'<img class="skip-lightbox" src="{dark}#only-dark" alt="{alt}">]({href})'
    )


def _card_overlay_markdown(rex: RenderedExample) -> str:
    """Return the hover-overlay markup carrying an example's summary.

    The span is absolutely positioned over the whole card by `extra.css` and revealed on
    hover, mirroring sphinx-gallery's thumbnail tooltip. It stays in the thumbnail's
    paragraph so it adds no vertical space to the card.

    Parameters
    ----------
    rex : RenderedExample
        The example whose summary is shown on hover.

    Returns
    -------
    str
        The overlay markup, or an empty string for an example without a summary.
    """
    if not rex.summary:
        return ""
    summary = html_lib.escape(_flatten_links(rex.summary))
    return f'<span class="examples-card-summary" aria-hidden="true">{summary}</span>'


def build_index(rendered: list[RenderedExample], *, root: Path) -> str:
    """Return the Markdown text of the gallery index page."""
    by_section: dict[str, list[RenderedExample]] = defaultdict(list)
    for rendered_example in rendered:
        by_section[rendered_example.spec.section].append(rendered_example)

    parts: list[str] = [
        "# Examples\n\n"
        "These examples show how to use ConfUSIus on real data, with an emphasis on\n"
        "workflows you can run and adapt in your own analyses.\n\n"
        "Each example starts from a plain Python script and is rendered as a notebook-style\n"
        "page with code, outputs, and downloadable source files.\n\n"
    ]

    # Iterate in insertion order; discover() yields specs sorted by the section
    # folder name (e.g. "01_io" before "02_registration"), so the index reflects
    # that explicit ordering rather than alphabetical order of the stripped name.
    for section, items in by_section.items():
        intro = _demote_h1(items[0].spec.section_intro.strip())
        parts.append((intro if intro else f"## {section}") + "\n\n")
        parts.append('<div class="grid cards examples-cards" markdown>\n\n')
        for rendered_example in sorted(items, key=lambda item: item.spec.source.name):
            href = rendered_example.md_path.relative_to(root).as_posix()
            image = _card_image_markdown(rendered_example, root=root, href=href)
            overlay = _card_overlay_markdown(rendered_example)
            card = (
                f"-   {image}{overlay}\n\n    ---\n\n"
                f"    **[{rendered_example.title}]({href})**"
            )
            parts.append(card + "\n\n")
        parts.append("</div>\n\n")

    return "".join(parts)
