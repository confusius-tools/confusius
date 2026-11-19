"""The integration classes of the example gallery.

The gallery is built by [zallery](https://github.com/confusius-tools/zallery), which
runs each example once and draws every figure again for the dark page. `[tool.zallery]`
in `pyproject.toml` names the classes here under `integrations`; zallery imports them
from the root of the repository, in the builder and in the kernel of each example.
"""

from __future__ import annotations

from collections.abc import Mapping

from zallery.integration.matplotlib import Matplotlib

from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK


class Figures(Matplotlib):
    """What ConfUSIus adds to zallery's matplotlib look.

    Everything else is zallery's default. A mapping is `{light: dark}`.

    Attributes
    ----------
    cmaps : Mapping[str, str]
        zallery's colormap pairs, and the sequential colormap that reads well on each
        background.
    colors : Mapping[str, str]
        The brand colors, brighter on the dark page.
    """

    cmaps: Mapping[str, str] = {**Matplotlib.cmaps, "inferno_r": "inferno"}
    colors: Mapping[str, str] = {RED_DARK: RED, TURQUOISE_DARK: TURQUOISE}
