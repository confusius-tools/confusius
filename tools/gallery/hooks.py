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

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def gallery_retheme(fig: Figure, theme: Literal["light", "dark"]) -> None:
    """Adjust `fig` after the gallery's built-in theme swap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure about to be printed, already swapped to `theme`.
    theme : {"light", "dark"}
        Theme the figure is now drawn with.
    """
