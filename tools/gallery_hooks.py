"""Project-level adjustments to the gallery's light/dark re-rendering.

`zallery.theme` turns each drawn figure dark by swapping the theme colors
(figure and axes backgrounds, black foreground) and the default diverging
colormap. Anything else an example draws is kept as is. When that is not enough,
edit `gallery_retheme` below: it runs inside the kernel, after the built-in swap,
once with `"dark"` before the dark rendering is printed and once with `"light"`
after the figure is swapped back. Whatever it changes for `"dark"` it must undo
for `"light"`, since the same figure object is returned to the example.

This file is the project's side of the gallery tooling: the project owns it, the
`zallery` package (`tools/zallery`) is the tool. `tools/build_gallery.py` names it
through `HOOK_MODULE`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zallery.theme import swap_cmaps

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SEQUENTIAL_CMAP_SWAPS: dict[str, str] = {"inferno_r": "inferno"}
"""Sequential colormaps examples draw with on light pages, and their dark-page twins.

A sequential map reads best when its low end fades into the page: bright on a light
page (`inferno_r`), dark on a dark page (`inferno`). Examples pick the light-page
name; the hook flips it for the dark rendering and back again.
"""


def gallery_retheme(fig: Figure, theme: Literal["light", "dark"]) -> None:
    """Adjust `fig` after the gallery's built-in theme swap.

    Flips the sequential colormaps in `SEQUENTIAL_CMAP_SWAPS` to their dark-page
    twins for `"dark"`, and back for `"light"`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure about to be printed, already swapped to `theme`.
    theme : {"light", "dark"}
        Theme the figure is now drawn with.
    """
    mapping = (
        SEQUENTIAL_CMAP_SWAPS
        if theme == "dark"
        else {dark: light for light, dark in SEQUENTIAL_CMAP_SWAPS.items()}
    )
    swap_cmaps(fig, mapping)
