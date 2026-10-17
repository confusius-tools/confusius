"""Build a light-and-dark example gallery for Zensical docs from jupytext scripts.

Examples are percent-format `.py` files executed once in an ipykernel; each
displayed matplotlib figure is rendered for both themes from that single run
(see `zallery.theme`). [`build_gallery`][zallery.build_gallery] runs the whole
pipeline: discover, cache, execute, render, index.
"""

from zallery.pipeline import build_gallery

__all__ = ["build_gallery"]
