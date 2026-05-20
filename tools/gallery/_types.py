"""Shared dataclasses for the gallery builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleSpec:
    """A single discovered example file.

    Parameters
    ----------
    source : pathlib.Path
        Absolute path to the percent-format ``.py`` file.
    base_name : str
        Output stem used for the rendered Markdown, downloads, and cache entry.
        Derived from ``source.stem`` with any leading numeric ordering prefix
        (e.g. ``01_``) stripped, so the source filename can control card order
        without leaking the prefix into URLs.
    section : str
        Section folder name (e.g. ``"io"``).
    section_intro : str
        Contents of the sibling ``_section.md`` file, or empty string if none.
    """

    source: Path
    base_name: str
    section: str
    section_intro: str


@dataclass(frozen=True)
class RenderedExample:
    """An example after the renderer has produced its artifacts."""

    spec: ExampleSpec
    title: str
    summary: str
    md_path: Path
    thumbnail_light: Path | None
    thumbnail_dark: Path | None
