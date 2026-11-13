"""Project palette swaps, restored exactly after each gallery rendering.

`gallery_retheme` runs after the built-in swap. It returns an `ExitStack` that
restores only the properties it changed, rather than reversing the whole palette.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.style
from matplotlib.lines import Line2D

from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK
from tools.gallery.theme import swap_cmaps, swap_colors

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SEQUENTIAL_CMAP_SWAPS: dict[str, str] = {"inferno_r": "inferno"}
"""Light-page sequential colormaps and their dark-page counterparts."""

BRAND_COLOR_SWAPS: dict[str, str] = {RED_DARK: RED, TURQUOISE_DARK: TURQUOISE}
"""Light-page brand colors and their brighter dark-page counterparts."""

LINE_COLOR_SWAPS: dict[str, str] = dict(
    zip(
        map(mcolors.to_hex, mpl.rcParamsDefault["axes.prop_cycle"].by_key()["color"]),
        mpl.style.library["dark_background"]["axes.prop_cycle"].by_key()["color"],
        strict=True,
    )
)
"""Default light and dark line cycles, matching the former two-execution gallery."""


def gallery_retheme(fig: Figure, theme: Literal["light", "dark"]) -> ExitStack:
    """Apply the project's palette for `theme`, returning its restoration context.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to adjust, already swapped to `theme`.
    theme : {"light", "dark"}
        Target theme. Light conversion is used for figures originally drawn on black.

    Returns
    -------
    contextlib.ExitStack
        Context manager restoring the original colors and colormaps.
    """
    cmaps, colors, cycle = (
        (mapping if theme == "dark" else {v: k for k, v in mapping.items()})
        for mapping in (SEQUENTIAL_CMAP_SWAPS, BRAND_COLOR_SWAPS, LINE_COLOR_SWAPS)
    )
    artists = fig.findobj()
    with ExitStack() as restore:
        restore.enter_context(swap_cmaps(artists, cmaps))
        restore.enter_context(swap_colors(artists, colors))
        for line in artists:
            if not isinstance(line, Line2D):
                continue
            color = line.get_color()
            # Named colors are explicit; default-cycle colors are RGB tuples or hex.
            if isinstance(color, str) and not color.startswith(("#", "C")):
                continue
            replacement = cycle.get(mcolors.to_hex(color))
            if replacement is not None:
                restore.callback(line.set_color, color)
                line.set_color(
                    mcolors.to_rgba(replacement, alpha=mcolors.to_rgba(color)[3])
                )
        return restore.pop_all()
