"""Kernel-side theming: render each figure once for light and once for dark.

This module runs inside the gallery's ipykernel (see `execute.py`), not in the
builder process. `setup_kernel` applies the light rcParams and wraps
matplotlib_inline's PNG printer so every displayed figure also carries its dark
rendering in the output metadata, under `metadata["image/png"]["dark"]`, which
`render.py` picks up. Nothing is re-executed: the dark rendering comes from
palette-swapping the drawn artists and printing again.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.style
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from confusius.plotting import ThresholdedColormap

Printer = Callable[[Figure], tuple[str, dict[str, Any]] | None]
"""A matplotlib_inline figure printer: base64 PNG plus metadata, or `None` if empty."""

Hook = Callable[[Figure, Literal["light", "dark"]], None]
"""Project hook run after the built-in swap, see `tools.gallery.hooks`."""


@dataclass(frozen=True)
class Theme:
    """Colors a rendering is built with, by role.

    Parameters
    ----------
    bg : str
        Figure background.
    axes_bg : str
        Axes background.
    fg : str
        Text, ticks, spines and default line color.
    diverging_cmap : str
        Name of the diverging colormap confusius defaults to on `bg`.
    """

    bg: str
    axes_bg: str
    fg: str
    diverging_cmap: str


LIGHT = Theme(bg="#f8f9fb", axes_bg="#ffffff", fg="black", diverging_cmap="coolwarm")
"""Theme every example is executed under."""

DARK = Theme(bg="#090c10", axes_bg="#111720", fg="#e6edf3", diverging_cmap="berlin")
"""Dark counterpart of `LIGHT`, role by role.

No value may equal a `LIGHT` value (`fg` is off-white, not `white`, because `white`
is `LIGHT.axes_bg`), or swapping back would be ambiguous.
"""

_COLOR_SLOTS = (
    ("get_facecolor", "set_facecolor"),
    ("get_edgecolor", "set_edgecolor"),
    ("get_color", "set_color"),
)
"""Getter/setter pairs through which artists expose a single color."""


def _as_rgba(value: Any) -> tuple[float, float, float, float] | None:
    """Return `value` as an RGBA tuple, or `None` if it is not a single color.

    Parameters
    ----------
    value : Any
        Whatever an artist's color getter returned.

    Returns
    -------
    tuple[float, float, float, float] or None
        The color, or `None` for per-element color arrays (collections, images)
        and anything matplotlib cannot parse as a color.
    """
    if isinstance(value, np.ndarray) and value.ndim != 1:
        return None
    try:
        return mcolors.to_rgba(value)
    except (ValueError, TypeError):
        return None


def _swap_colors(fig: Figure, src: Theme, dst: Theme) -> None:
    """Replace every artist color equal to a `src` role color by `dst`'s.

    Colors are read for all artists before any is written: an Axes and its
    background patch expose the same facecolor, so a read-modify-write loop would
    apply the mapping twice. Matching is on RGB; the artist's alpha is kept.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to recolor in place.
    src : Theme
        Theme the figure is currently drawn with.
    dst : Theme
        Theme to recolor to.
    """
    table = {
        mcolors.to_rgb(getattr(src, role)): getattr(dst, role)
        for role in ("bg", "axes_bg", "fg")
    }
    writes: list[tuple[Callable[[Any], Any], tuple[float, float, float, float]]] = []
    for artist in fig.findobj():
        for getter, setter in _COLOR_SLOTS:
            if not hasattr(artist, getter):
                continue
            rgba = _as_rgba(getattr(artist, getter)())
            if rgba is not None and rgba[:3] in table:
                new = mcolors.to_rgba(table[rgba[:3]], alpha=rgba[3])
                writes.append((getattr(artist, setter), new))
    for setter, color in writes:
        setter(color)


def _swap_cmaps(fig: Figure, src: Theme, dst: Theme) -> None:
    """Switch every mappable drawn with `src`'s diverging colormap to `dst`'s.

    A `ThresholdedColormap` is matched on its source and rebuilt with the same
    gray band, so thresholded stat maps keep their threshold in both themes.
    Colorbars follow their mappable automatically.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to update in place.
    src : Theme
        Theme the figure is currently drawn with.
    dst : Theme
        Theme to switch to.
    """
    mappables: list[Any] = fig.findobj(lambda a: isinstance(a, ScalarMappable))
    for mappable in mappables:
        cmap = mappable.get_cmap()
        if isinstance(cmap, ThresholdedColormap):
            if cmap.source.name == src.diverging_cmap:
                mappable.set_cmap(cmap.with_source(dst.diverging_cmap))
        elif cmap.name == src.diverging_cmap:
            mappable.set_cmap(dst.diverging_cmap)


def retheme(fig: Figure, src: Theme, dst: Theme) -> None:
    """Recolor a drawn figure from `src` to `dst` in place.

    Only colors and colormaps that exactly match a `src` role are touched;
    anything else the author chose on purpose is kept.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to recolor.
    src : Theme
        Theme the figure is currently drawn with.
    dst : Theme
        Theme to recolor to.
    """
    _swap_colors(fig, src, dst)
    _swap_cmaps(fig, src, dst)


def render_both(
    fig: Figure, *, printer: Printer, hook: Hook | None = None
) -> tuple[str, dict[str, Any]] | None:
    """Print `fig` under `LIGHT`, then under `DARK`, without re-executing anything.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A figure drawn under the `LIGHT` theme.
    printer : Callable[[matplotlib.figure.Figure], tuple[str, dict] or None]
        The figure printer matplotlib_inline registered for `Figure`.
    hook : Callable[[matplotlib.figure.Figure, str], None], optional
        Project hook called with `"dark"` after the swap to dark and with
        `"light"` after the swap back, so it can adjust what the built-in swap
        does not cover. It must undo in `"light"` what it did in `"dark"`. If
        not provided, nothing else is adjusted.

    Returns
    -------
    tuple[str, dict[str, Any]] or None
        The light rendering and its metadata, with the dark rendering added under
        `"dark"`. `None` when `printer` returns `None` (an empty figure).
    """
    light = printer(fig)
    if light is None:
        return None
    data, metadata = light
    # Layout engines (constrained, tight) re-solve on every draw and are not
    # idempotent, so the dark render would come out shifted by a few pixels.
    # Freezing the layout after the light render keeps the geometry identical.
    fig.set_layout_engine("none")
    retheme(fig, LIGHT, DARK)
    try:
        if hook is not None:
            hook(fig, "dark")
        dark = printer(fig)
    finally:
        retheme(fig, DARK, LIGHT)
        if hook is not None:
            hook(fig, "light")
    if dark is None:
        return light
    return data, {**metadata, "dark": dark[0]}


def setup_kernel() -> None:
    """Configure the running kernel: light rcParams, retina PNGs, dark hook.

    Must run before any example cell. `savefig.facecolor` is deliberately left at
    `"auto"` so the swapped figure facecolor is what gets printed.
    """
    from IPython import get_ipython
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    from tools.gallery.hooks import gallery_retheme

    warnings.filterwarnings("ignore", message="IProgress not found.*")
    # rich.live.Live.refresh() warns once per process the first time it detects a
    # Jupyter kernel without ipywidgets installed; it would land in the rendered
    # output of whichever cell happens to trigger it.
    warnings.filterwarnings(
        "ignore", message='install "ipywidgets" for Jupyter support'
    )
    mpl.style.use("default")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "figure.figsize": (7, 4),
            "figure.facecolor": LIGHT.bg,
            "axes.facecolor": LIGHT.axes_bg,
        }
    )
    set_matplotlib_formats("retina")
    shell = get_ipython()
    png_formatter = shell.display_formatter.formatters["image/png"]
    printer = png_formatter.lookup_by_type(Figure)
    png_formatter.for_type(
        Figure, partial(render_both, printer=printer, hook=gallery_retheme)
    )
