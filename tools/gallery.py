"""The integration classes of the example gallery.

The gallery is built by [zallery](https://github.com/confusius-tools/zallery), which
runs each example once and draws every figure again for the dark page. `[tool.zallery]`
in `pyproject.toml` names the classes here under `integrations`; zallery imports them
from the root of the repository, in the builder and in the kernel of each example.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.style
from zallery.integration.matplotlib import Matplotlib

from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK


class Figures(Matplotlib):
    """The matplotlib look of the ConfUSIus site.

    A pair is `(light, dark)`; a mapping is `{light: dark}`.

    Attributes
    ----------
    background : tuple[str, str]
        Figure background: the page colors `docs/stylesheets/extra.css` sets, so that
        figures blend into the page.
    axes_background : tuple[str, str]
        Axes background.
    foreground : tuple[str, str]
        Text, ticks, spines and default lines.
    palette : tuple[None, tuple[str, ...]]
        Matplotlib's default color cycle on the light page, its `dark_background`
        cycle on the dark one.
    cmaps : Mapping[str, str]
        The default diverging colormap of the library on each background, and the
        sequential colormap that reads well on each.
    colors : Mapping[str, str]
        The brand colors, brighter on the dark page.
    """

    background = ("#f8f9fb", "#090c10")
    axes_background = ("#fefefe", "#111720")
    foreground = ("black", "#e6edf3")
    palette = (
        None,
        tuple(
            mpl.style.library["dark_background"]["axes.prop_cycle"].by_key()["color"]
        ),
    )
    cmaps: Mapping[str, str] = {"coolwarm": "berlin", "inferno_r": "inferno"}
    colors: Mapping[str, str] = {RED_DARK: RED, TURQUOISE_DARK: TURQUOISE}
