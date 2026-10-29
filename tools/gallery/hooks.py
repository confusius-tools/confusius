"""Project-level adjustments to the gallery's light/dark re-rendering.

`tools.gallery.theme` turns each drawn figure dark by swapping the theme colors
(figure and axes backgrounds, black foreground) and the default diverging
colormap. Anything else an example draws is kept as is. When that is not enough,
edit `gallery_retheme` below: it runs inside the kernel, after the built-in swap,
once with `"dark"` before the dark rendering is printed and once with `"light"`
after the figure is swapped back. Whatever it changes for `"dark"` it must undo
for `"light"`, since the same figure object is returned to the example.

This file is the project's side of the gallery tooling: a project using the
gallery owns it, the rest of `tools/gallery` is the tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK
from tools.gallery.theme import swap_cmaps, swap_colors

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SEQUENTIAL_CMAP_SWAPS: dict[str, str] = {"inferno_r": "inferno"}
"""Sequential colormaps examples draw with on light pages, and their dark-page twins.

A sequential map reads best when its low end fades into the page: bright on a light
page (`inferno_r`), dark on a dark page (`inferno`). Examples pick the light-page
name; the hook flips it for the dark rendering and back again.
"""

BRAND_COLOR_SWAPS: dict[str, str] = {RED_DARK: RED, TURQUOISE_DARK: TURQUOISE}
"""Brand colors examples draw with on light pages, and their dark-page twins.

The darker variants keep contrast against the light page; the dark page gets the
brighter brand colors, as in the docs theme.
"""


def gallery_retheme(fig: Figure, theme: Literal["light", "dark"]) -> None:
    """Adjust `fig` after the gallery's built-in theme swap.

    Flips the sequential colormaps in `SEQUENTIAL_CMAP_SWAPS` and the brand colors
    in `BRAND_COLOR_SWAPS` to their dark-page twins for `"dark"`, and back for
    `"light"`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure about to be printed, already swapped to `theme`.
    theme : {"light", "dark"}
        Theme the figure is now drawn with.
    """
    if theme == "dark":
        swap_cmaps(fig, SEQUENTIAL_CMAP_SWAPS)
        swap_colors(fig, BRAND_COLOR_SWAPS)
    else:
        swap_cmaps(fig, {v: k for k, v in SEQUENTIAL_CMAP_SWAPS.items()})
        swap_colors(fig, {v: k for k, v in BRAND_COLOR_SWAPS.items()})
