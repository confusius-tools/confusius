"""Tests for tools.gallery.theme."""

from __future__ import annotations

import base64
import io
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
    plt.style.use("default")
    plt.rcParams.update({"figure.facecolor": LIGHT.bg, "axes.facecolor": LIGHT.axes_bg})
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


def test_retheme_keeps_intentional_colors(figure) -> None:
    figure["fig"].set_facecolor("#123456")

    retheme(figure["fig"], LIGHT, DARK)

    assert figure["fig"].get_facecolor() == pytest.approx(mcolors.to_rgba("#123456"))
    assert figure["red_line"].get_color() == "tab:red"


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
    assert again[0] == data


def test_render_both_passes_through_empty_figure() -> None:
    fig = plt.figure()
    try:
        assert render_both(fig, printer=partial(retina_figure, base64=True)) is None
    finally:
        plt.close(fig)


def test_render_both_calls_hook_after_each_swap(figure) -> None:
    """The project hook sees the figure in dark before printing and in light after."""
    seen: list[tuple[str, tuple[float, ...]]] = []

    def hook(fig, theme):
        seen.append((theme, tuple(fig.get_facecolor())))
        fig.axes[1].set_title("dark title" if theme == "dark" else "title")

    render_both(figure["fig"], printer=partial(retina_figure, base64=True), hook=hook)

    assert [theme for theme, _ in seen] == ["dark", "light"]
    assert seen[0][1] == pytest.approx(mcolors.to_rgba(DARK.bg))
    assert seen[1][1] == pytest.approx(mcolors.to_rgba(LIGHT.bg))
    assert figure["ax1"].get_title() == "title"


def test_render_both_restores_light_when_hook_raises(figure) -> None:
    def hook(fig, theme):
        raise RuntimeError("hook broke")

    with pytest.raises(RuntimeError, match="hook broke"):
        render_both(
            figure["fig"], printer=partial(retina_figure, base64=True), hook=hook
        )

    assert figure["fig"].get_facecolor() == pytest.approx(mcolors.to_rgba(LIGHT.bg))


def test_default_project_hook_is_a_noop(figure) -> None:
    from tools.gallery.hooks import gallery_retheme

    before = figure["fig"].get_facecolor()
    gallery_retheme(figure["fig"], "dark")
    assert figure["fig"].get_facecolor() == before
