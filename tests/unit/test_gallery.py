"""Tests for the gallery integration class in `tools/gallery.py`.

The gallery builder is the `zallery` package, tested in its own repository. These
tests cover what ConfUSIus adds to its matplotlib look: a colormap pair and the brand
colors, and that `ThresholdedColormap` has the shape zallery's colormap swap relies on.
"""

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps

from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK
from confusius.plotting import ThresholdedColormap
from tools.gallery import Figures


def _retheme(fig, scheme):
    """Give a figure the look of a scheme; the figure must need a change for it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to recolor in place.
    scheme : {"light", "dark"}
        The color scheme to recolor for.

    Returns
    -------
    contextlib.ExitStack
        Context manager restoring the figure.
    """
    restore = Figures().retheme(fig, scheme)
    assert restore is not None
    return restore


def test_dark_page_rebuilds_thresholded_colormaps_on_their_source(matplotlib_pyplot):
    """A `ThresholdedColormap` keeps its threshold around the dark-page colormap."""
    fig, axes = matplotlib_pyplot.subplots(1, 3)
    banded = {
        name: ThresholdedColormap(
            colormaps[name], mcolors.Normalize(-1.0, 1.0), threshold=0.2
        )
        for name in ("coolwarm", "inferno_r", "viridis")
    }
    data = np.linspace(-1.0, 1.0, 12).reshape(3, 4)
    images = {
        name: ax.imshow(data, cmap=cmap, norm=cmap.norm)
        for ax, (name, cmap) in zip(axes, banded.items(), strict=True)
    }

    with _retheme(fig, "dark"):
        dark = {name: image.get_cmap() for name, image in images.items()}
        assert all(isinstance(cmap, ThresholdedColormap) for cmap in dark.values())
        # `coolwarm` is zallery's own pair, which the class must keep.
        assert {name: cmap.source.name for name, cmap in dark.items()} == {
            "coolwarm": "berlin",
            "inferno_r": "inferno",
            "viridis": "viridis",
        }
        assert dark["coolwarm"].threshold == 0.2

    assert all(images[name].get_cmap() is cmap for name, cmap in banded.items())


def test_dark_page_brightens_the_brand_colors(matplotlib_pyplot):
    """The light-page brand colors take their dark values, and come back."""
    fig, ax = matplotlib_pyplot.subplots()
    (red,) = ax.plot([0, 1], color=RED_DARK)
    label = ax.set_ylabel("iterations", color=TURQUOISE_DARK)
    (kept,) = ax.plot([1, 0], color="tab:red")

    with _retheme(fig, "dark"):
        assert mcolors.to_hex(red.get_color()) == RED
        assert mcolors.to_hex(label.get_color()) == TURQUOISE
        assert kept.get_color() == "tab:red"

    assert red.get_color() == RED_DARK
    assert label.get_color() == TURQUOISE_DARK


def test_figure_on_black_gets_the_light_brand_colors_on_the_light_page(
    matplotlib_pyplot,
):
    """A figure drawn on black with the bright brand red prints darker on light."""
    fig, ax = matplotlib_pyplot.subplots(facecolor="black")
    ax.set_facecolor("black")
    (line,) = ax.plot([0, 1], color=RED)

    with _retheme(fig, "light"):
        assert mcolors.to_hex(line.get_color()) == RED_DARK

    with _retheme(fig, "dark"):
        assert mcolors.to_hex(line.get_color()) == RED

    assert line.get_color() == RED
