"""Tests for tools.gallery.theme."""

from __future__ import annotations

import base64
import io
from contextlib import ExitStack
from functools import partial

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.core.pylabtools import retina_figure
from matplotlib import colormaps

from confusius.plotting import ThresholdedColormap
from tools.gallery.theme import DARK, LIGHT, render_both, retheme


@pytest.fixture
def figure():
    """A light-themed figure exercising every kind of artist the swap touches."""
    with matplotlib.rc_context():
        plt.style.use("default")
        plt.rcParams.update(
            {"figure.facecolor": LIGHT.bg, "axes.facecolor": LIGHT.axes_bg}
        )
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3), layout="constrained")
        banded = ThresholdedColormap(
            colormaps[LIGHT.diverging_cmap], mcolors.Normalize(-1.0, 1.0), threshold=0.2
        )
        image = ax0.imshow(
            np.linspace(-1.0, 1.0, 12).reshape(3, 4), cmap=banded, norm=banded.norm
        )
        fig.colorbar(image, ax=ax0)
        scatter = ax1.scatter([0, 1], [0, 1], c=[0, 1], cmap=LIGHT.diverging_cmap)
        (theme_line,) = ax1.plot([0, 1], color="black")
        (red_line,) = ax1.plot([1, 0], color="tab:red")
        legend = ax1.legend([theme_line, red_line], ["theme", "kept"], framealpha=0.8)
        ax1.set_title("title")
        artists = {
            "fig": fig,
            "ax1": ax1,
            "image": image,
            "scatter": scatter,
            "theme_line": theme_line,
            "red_line": red_line,
            "legend": legend,
        }
        yield artists
        plt.close(fig)


def _corner_rgb(png_b64: str) -> tuple[float, float, float]:
    pixels = mimg.imread(io.BytesIO(base64.b64decode(png_b64)))
    r, g, b = (float(v) for v in pixels[0, 0, :3])
    return r, g, b


def test_retheme_swaps_theme_colors(figure) -> None:
    retheme(figure["fig"], LIGHT, DARK)

    assert figure["fig"].get_facecolor() == pytest.approx(mcolors.to_rgba(DARK.bg))
    assert figure["ax1"].get_facecolor() == pytest.approx(mcolors.to_rgba(DARK.axes_bg))
    assert mcolors.to_rgba(figure["ax1"].title.get_color()) == pytest.approx(
        mcolors.to_rgba(DARK.fg)
    )
    assert mcolors.to_rgba(figure["theme_line"].get_color()) == pytest.approx(
        mcolors.to_rgba(DARK.fg)
    )


def test_retheme_recolors_tick_marks(figure) -> None:
    retheme(figure["fig"], LIGHT, DARK)

    tick = figure["ax1"].xaxis.get_major_ticks()[0].tick1line
    assert mcolors.to_rgba(tick.get_markeredgecolor()) == pytest.approx(
        mcolors.to_rgba(DARK.fg)
    )


def test_retheme_keeps_intentional_colors(figure) -> None:
    figure["fig"].set_facecolor("#123456")
    (white_line,) = figure["ax1"].plot([0, 1], color="white")

    retheme(figure["fig"], LIGHT, DARK)

    assert figure["fig"].get_facecolor() == pytest.approx(mcolors.to_rgba("#123456"))
    assert figure["red_line"].get_color() == "tab:red"
    assert white_line.get_color() == "white"


def test_retheme_preserves_alpha(figure) -> None:
    retheme(figure["fig"], LIGHT, DARK)

    frame = figure["legend"].get_frame().get_facecolor()
    assert frame == pytest.approx(mcolors.to_rgba(DARK.axes_bg, alpha=0.8))


def test_retheme_swaps_diverging_colormaps_including_thresholded(figure) -> None:
    retheme(figure["fig"], LIGHT, DARK)

    banded = figure["image"].get_cmap()
    assert isinstance(banded, ThresholdedColormap)
    assert banded.source.name == DARK.diverging_cmap
    assert banded.threshold == 0.2
    assert figure["scatter"].get_cmap().name == DARK.diverging_cmap

    retheme(figure["fig"], DARK, LIGHT)

    assert figure["image"].get_cmap().source.name == LIGHT.diverging_cmap
    assert figure["scatter"].get_cmap().name == LIGHT.diverging_cmap


def test_render_both_returns_light_data_and_dark_metadata(figure) -> None:
    printer = partial(retina_figure, base64=True)

    result = render_both(figure["fig"], printer=printer)

    assert result is not None
    data, metadata = result
    assert set(metadata) >= {"width", "height", "dark"}
    assert _corner_rgb(data) == pytest.approx(mcolors.to_rgb(LIGHT.bg), abs=1 / 255)
    assert _corner_rgb(metadata["dark"]) == pytest.approx(
        mcolors.to_rgb(DARK.bg), abs=1 / 255
    )
    # The figure is handed back in its light state, with identical geometry.
    again = printer(figure["fig"])
    assert again is not None
    assert _corner_rgb(again[0]) == pytest.approx(mcolors.to_rgb(LIGHT.bg), abs=1 / 255)
    assert (
        figure["fig"].get_layout_engine().__class__.__name__
        == "ConstrainedLayoutEngine"
    )


@pytest.mark.parametrize("cmap", ["inferno", "berlin"])
def test_render_both_preserves_existing_destination_colormap(cmap) -> None:
    """Displaying twice must not change an explicitly chosen destination palette."""
    from tools.gallery.hooks import gallery_retheme

    fig, ax = plt.subplots(facecolor=LIGHT.bg)
    image = ax.imshow([[0, 1]], cmap=cmap)
    original = image.get_cmap()
    printed = []

    def printer(fig):
        printed.append(image.get_cmap().name)
        return "png", {}

    try:
        for _ in range(2):
            render_both(fig, printer=printer, hook=gallery_retheme)
            assert image.get_cmap() is original
        assert printed == [cmap] * 4
    finally:
        plt.close(fig)


def test_render_both_preserves_automatic_marker_colors() -> None:
    """Automatic marker colors must still follow later line-color changes."""
    fig, ax = plt.subplots(facecolor=LIGHT.bg)
    (line,) = ax.plot([0, 1], color="black", marker="o")
    try:
        render_both(fig, printer=lambda fig: ("png", {}))
        line.set_color("red")
        assert line.get_markerfacecolor() == "red"
        assert line.get_markeredgecolor() == "red"
    finally:
        plt.close(fig)


def test_render_both_uses_dark_style_line_cycle() -> None:
    """Default traces follow dark_background; explicitly named colors stay fixed."""
    from tools.gallery.hooks import gallery_retheme

    with matplotlib.rc_context():
        plt.style.use("default")
        fig, ax = plt.subplots(facecolor=LIGHT.bg)
        lines = [ax.plot([0, 1], marker="o")[0] for _ in range(10)]
        (explicit,) = ax.plot([1, 0], color="tab:blue")
        original = [line.get_color() for line in lines]
        printed = []

        def printer(fig):
            printed.append([mcolors.to_hex(line.get_color()) for line in lines])
            assert explicit.get_color() == "tab:blue"
            return "png", {}

        try:
            render_both(fig, printer=printer, hook=gallery_retheme)
            assert printed[0] == [mcolors.to_hex(color) for color in original]
            assert (
                printed[1]
                == plt.style.library["dark_background"]["axes.prop_cycle"].by_key()[
                    "color"
                ]
            )
            assert [line.get_color() for line in lines] == original
        finally:
            plt.close(fig)


def test_render_both_restores_original_state_when_printing_fails(figure) -> None:
    """A failed dark print must restore colormap objects, colors and layout."""
    from tools.gallery.hooks import gallery_retheme

    fig = figure["fig"]
    image = figure["image"]
    cmap = image.get_cmap()
    engine = fig.get_layout_engine()
    (line,) = figure["ax1"].plot([0, 1], color="#d93a54", marker="o")

    def printer(fig):
        if mcolors.to_hex(fig.get_facecolor()) == DARK.bg:
            raise RuntimeError("dark print failed")
        return "png", {}

    with pytest.raises(RuntimeError, match="dark print failed"):
        render_both(fig, printer=printer, hook=gallery_retheme)

    assert image.get_cmap() is cmap
    assert fig.get_layout_engine() is engine
    assert mcolors.to_hex(fig.get_facecolor()) == LIGHT.bg
    assert line.get_color() == "#d93a54"
    line.set_color("red")
    assert line.get_markerfacecolor() == "red"


def test_render_both_passes_through_empty_figure() -> None:
    fig = plt.figure()
    try:
        assert render_both(fig, printer=partial(retina_figure, base64=True)) is None
    finally:
        plt.close(fig)


def test_render_both_themes_figures_drawn_on_black() -> None:
    """A figure on confusius's black default renders for both pages and is restored."""
    fig, ax = plt.subplots(facecolor="black")
    ax.set_facecolor("black")
    title = ax.set_title("title", color="white")
    try:
        result = render_both(fig, printer=partial(retina_figure, base64=True))

        assert result is not None
        data, metadata = result
        assert _corner_rgb(data) == pytest.approx(mcolors.to_rgb(LIGHT.bg), abs=1 / 255)
        assert _corner_rgb(metadata["dark"]) == pytest.approx(
            mcolors.to_rgb(DARK.bg), abs=1 / 255
        )
        assert mcolors.to_hex(fig.get_facecolor()) == "#000000"
        assert mcolors.to_hex(ax.get_facecolor()) == "#000000"
        assert mcolors.to_hex(title.get_color()) == "#ffffff"
    finally:
        plt.close(fig)


def test_render_both_restores_hook_changes(figure) -> None:
    """The hook sees the dark figure and registers its exact restoration."""
    seen: list[tuple[str, tuple[float, ...]]] = []

    def hook(fig, theme):
        seen.append((theme, tuple(fig.get_facecolor())))
        restore = ExitStack()
        restore.callback(fig.axes[1].set_title, fig.axes[1].get_title())
        fig.axes[1].set_title("dark title")
        return restore

    render_both(figure["fig"], printer=partial(retina_figure, base64=True), hook=hook)

    assert [theme for theme, _ in seen] == ["dark"]
    assert seen[0][1] == pytest.approx(mcolors.to_rgba(DARK.bg))
    assert figure["ax1"].get_title() == "title"


def test_render_both_restores_light_when_hook_raises(figure) -> None:
    def hook(fig, theme):
        raise RuntimeError("hook broke")

    with pytest.raises(RuntimeError, match="hook broke"):
        render_both(
            figure["fig"], printer=partial(retina_figure, base64=True), hook=hook
        )

    assert figure["fig"].get_facecolor() == pytest.approx(mcolors.to_rgba(LIGHT.bg))


def test_default_project_hook_flips_sequential_cmaps_and_back(figure) -> None:
    """The shipped hook swaps inferno_r for inferno on dark pages, plain or banded."""
    from tools.gallery.hooks import gallery_retheme

    ax = figure["ax1"]
    plain = ax.scatter([0, 1], [1, 0], c=[0, 1], cmap="inferno_r")
    banded = ax.imshow(
        np.linspace(0.0, 1.0, 4).reshape(2, 2),
        cmap=ThresholdedColormap(colormaps["inferno_r"], mcolors.Normalize(0.0, 1.0)),
    )
    before = figure["fig"].get_facecolor()

    with gallery_retheme(figure["fig"], "dark"):
        assert plain.get_cmap().name == "inferno"
        assert banded.get_cmap().source.name == "inferno"
        assert figure["scatter"].get_cmap().name == LIGHT.diverging_cmap
        assert figure["fig"].get_facecolor() == before

    assert plain.get_cmap().name == "inferno_r"
    assert banded.get_cmap().source.name == "inferno_r"


def test_default_project_hook_brightens_brand_colors_and_back(figure) -> None:
    """The shipped hook swaps the light-page brand colors for the dark-page ones."""
    from confusius._utils.colors import RED, RED_DARK, TURQUOISE, TURQUOISE_DARK
    from tools.gallery.hooks import gallery_retheme

    ax = figure["ax1"]
    (red,) = ax.plot([0, 1], color=RED_DARK)
    label = ax.set_ylabel("iterations", color=TURQUOISE_DARK)

    with gallery_retheme(figure["fig"], "dark"):
        assert mcolors.to_hex(red.get_color()) == RED
        assert mcolors.to_hex(label.get_color()) == TURQUOISE
        assert figure["red_line"].get_color() == "tab:red"

    assert mcolors.to_hex(red.get_color()) == RED_DARK
    assert mcolors.to_hex(label.get_color()) == TURQUOISE_DARK


def test_default_project_hook_on_black_figures_uses_light_brand_colors() -> None:
    """A black figure's bright brand red prints darker on the light page only."""
    from confusius._utils.colors import RED, RED_DARK
    from tools.gallery.hooks import gallery_retheme

    fig, ax = plt.subplots(facecolor="black")
    (line,) = ax.plot([0, 1], color=RED)
    printed: list[str] = []

    def printer(f):
        printed.append(mcolors.to_hex(line.get_color()))
        return retina_figure(f, base64=True)

    try:
        render_both(fig, printer=printer, hook=gallery_retheme)

        assert printed == [RED_DARK, RED]
        assert mcolors.to_hex(line.get_color()) == RED
    finally:
        plt.close(fig)
