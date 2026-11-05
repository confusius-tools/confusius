"""Tests for the project's gallery theme hook in `tools/gallery_hooks.py`.

The gallery builder is the `zallery` package, tested in its own repository. These
tests cover what ConfUSIus adds to it: the palette swaps of the hook, and that
`ThresholdedColormap` has the shape zallery's colormap swap relies on.
"""

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest
from IPython.core.pylabtools import retina_figure
from matplotlib import colormaps
from zallery.theme import DARK, LIGHT, render_both, retheme

from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK
from confusius.plotting import ThresholdedColormap
from tools.gallery_hooks import gallery_retheme


def test_zallery_swaps_thresholded_colormaps_on_their_source(matplotlib_pyplot):
    """The built-in swap rebuilds a `ThresholdedColormap` around the dark colormap."""
    fig, ax = matplotlib_pyplot.subplots(facecolor=LIGHT.bg)
    banded = ThresholdedColormap(
        colormaps[LIGHT.diverging_cmap], mcolors.Normalize(-1.0, 1.0), threshold=0.2
    )
    image = ax.imshow(
        np.linspace(-1.0, 1.0, 12).reshape(3, 4), cmap=banded, norm=banded.norm
    )

    with retheme(fig, LIGHT, DARK):
        dark = image.get_cmap()
        assert isinstance(dark, ThresholdedColormap)
        assert dark.source.name == DARK.diverging_cmap
        assert dark.threshold == 0.2

    assert image.get_cmap() is banded


def test_gallery_retheme_flips_sequential_cmaps_and_back(matplotlib_pyplot):
    """inferno_r on the light page becomes inferno on the dark one, plain or banded."""
    fig, ax = matplotlib_pyplot.subplots()
    plain = ax.scatter([0, 1], [1, 0], c=[0, 1], cmap="inferno_r")
    banded = ax.imshow(
        np.linspace(0.0, 1.0, 4).reshape(2, 2),
        cmap=ThresholdedColormap(colormaps["inferno_r"], mcolors.Normalize(0.0, 1.0)),
    )
    kept = ax.imshow(np.linspace(-1.0, 1.0, 4).reshape(2, 2), cmap="coolwarm")
    before = fig.get_facecolor()

    with gallery_retheme(fig, "dark"):
        assert plain.get_cmap().name == "inferno"
        assert banded.get_cmap().source.name == "inferno"
        assert kept.get_cmap().name == "coolwarm"
        assert fig.get_facecolor() == before

    assert plain.get_cmap().name == "inferno_r"
    assert banded.get_cmap().source.name == "inferno_r"


def test_gallery_retheme_brightens_brand_colors_and_back(matplotlib_pyplot):
    """The light-page brand colors become the dark-page ones; named colors stay."""
    fig, ax = matplotlib_pyplot.subplots()
    (red,) = ax.plot([0, 1], color=RED_DARK)
    (named,) = ax.plot([1, 0], color="tab:red")
    label = ax.set_ylabel("iterations", color=TURQUOISE_DARK)

    with gallery_retheme(fig, "dark"):
        assert mcolors.to_hex(red.get_color()) == RED
        assert mcolors.to_hex(label.get_color()) == TURQUOISE
        assert named.get_color() == "tab:red"

    assert mcolors.to_hex(red.get_color()) == RED_DARK
    assert mcolors.to_hex(label.get_color()) == TURQUOISE_DARK


def test_gallery_retheme_on_black_figures_uses_light_brand_colors(matplotlib_pyplot):
    """A black figure's bright brand red prints darker on the light page only."""
    fig, ax = matplotlib_pyplot.subplots(facecolor="black")
    (line,) = ax.plot([0, 1], color=RED)
    printed: list[str] = []

    def printer(figure):
        printed.append(mcolors.to_hex(line.get_color()))
        return retina_figure(figure, base64=True)

    render_both(fig, printer=printer, hook=gallery_retheme)

    assert printed == [RED_DARK, RED]
    assert mcolors.to_hex(line.get_color()) == RED


def test_gallery_retheme_uses_dark_style_line_cycle(matplotlib_pyplot):
    """Default traces follow dark_background; explicitly named colors stay fixed."""
    with matplotlib.rc_context():
        matplotlib_pyplot.style.use("default")
        fig, ax = matplotlib_pyplot.subplots(facecolor=LIGHT.bg)
        lines = [ax.plot([0, 1], marker="o")[0] for _ in range(10)]
        (explicit,) = ax.plot([1, 0], color="tab:blue")
        original = [line.get_color() for line in lines]
        printed = []

        def printer(figure):
            printed.append([mcolors.to_hex(line.get_color()) for line in lines])
            assert explicit.get_color() == "tab:blue"
            return "png", {}

        render_both(fig, printer=printer, hook=gallery_retheme)

        dark_cycle = matplotlib_pyplot.style.library["dark_background"][
            "axes.prop_cycle"
        ].by_key()["color"]
        assert printed[0] == [mcolors.to_hex(color) for color in original]
        assert printed[1] == dark_cycle
        assert [line.get_color() for line in lines] == original


def test_gallery_retheme_is_undone_when_printing_fails(matplotlib_pyplot):
    """A failed dark print leaves the hook's colors and automatic markers as drawn."""
    fig, ax = matplotlib_pyplot.subplots(facecolor=LIGHT.bg)
    image = ax.imshow([[0, 1]], cmap="inferno_r")
    cmap = image.get_cmap()
    (line,) = ax.plot([0, 1], color=RED_DARK, marker="o")

    def printer(figure):
        if mcolors.to_hex(figure.get_facecolor()) == DARK.bg:
            raise RuntimeError("dark print failed")
        return "png", {}

    with pytest.raises(RuntimeError, match="dark print failed"):
        render_both(fig, printer=printer, hook=gallery_retheme)

    assert image.get_cmap() is cmap
    assert mcolors.to_hex(fig.get_facecolor()) == LIGHT.bg
    assert line.get_color() == RED_DARK
    line.set_color("red")
    assert line.get_markerfacecolor() == "red"
