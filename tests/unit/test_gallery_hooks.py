"""Tests for the project's gallery theme hook in `tools/gallery_hooks.py`."""

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps

from confusius.plotting import ThresholdedColormap
from tools.gallery_hooks import gallery_retheme


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

    gallery_retheme(fig, "dark")

    assert plain.get_cmap().name == "inferno"
    assert banded.get_cmap().source.name == "inferno"
    assert kept.get_cmap().name == "coolwarm"
    assert fig.get_facecolor() == before

    gallery_retheme(fig, "light")

    assert plain.get_cmap().name == "inferno_r"
    assert banded.get_cmap().source.name == "inferno_r"
