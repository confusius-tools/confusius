"""Colormaps built by the plotting functions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.colors import Colormap, Normalize

ColorStop = tuple[float, str | tuple[float, ...]]
"""A `(position, color)` pair as accepted by `LinearSegmentedColormap.from_list`."""


def _gray_band_colors(
    cmap: Colormap,
    norm: Normalize,
    threshold: float,
    threshold_mode: Literal["lower", "upper"],
) -> list[ColorStop]:
    """Return `cmap`'s color stops with a gray band over the thresholded range.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap to take the non-gray stops from.
    norm : matplotlib.colors.Normalize
        Norm positioning the band: `norm(-threshold)` to `norm(threshold)`.
    threshold : float
        Threshold on the absolute data values, already made non-negative.
    threshold_mode : {"lower", "upper"}
        `"lower"` grays out values inside `[-threshold, threshold]`, `"upper"`
        grays out values outside it.

    Returns
    -------
    list[tuple[float, str or tuple[float, ...]]]
        Color stops with strictly increasing positions.
    """
    cmap_colors: list[ColorStop] = [
        (i / (cmap.N - 1), cmap(i / (cmap.N - 1))) for i in range(cmap.N)
    ]

    gray_low = norm(-threshold)
    gray_high = norm(threshold)
    if threshold_mode == "lower":
        colors_before = [c for c in cmap_colors if c[0] <= gray_low]
        colors_after = [c for c in cmap_colors if c[0] >= gray_high]

        if gray_low < gray_high:
            # We clip the gray low/high points to [0, 1] because Matplotlib expects
            # colormap values in that range.
            gray_band: list[ColorStop] = [
                (max(0, gray_low), "gray"),
                (min(1.0, gray_high), "gray"),
            ]
        else:
            gray_band = []
        new_colors = colors_before + gray_band + colors_after
    else:
        colors_middle = [c for c in cmap_colors if gray_low <= c[0] <= gray_high]

        gray_band_low: list[ColorStop] = (
            [(0.0, "gray"), (gray_low, "gray")] if gray_low > 0.0 else []
        )
        gray_band_high: list[ColorStop] = (
            [(gray_high, "gray"), (1.0, "gray")] if gray_high < 1.0 else []
        )
        new_colors = gray_band_low + colors_middle + gray_band_high

    # Matplotlib 3.11+ requires (value, color) pairs passed to `from_list` to be
    # strictly monotonically increasing in value. Our boundary points can collide
    # with neighboring cmap entries (e.g. when `gray_low == 0` the first
    # `colors_before` entry shares its value with the start of `gray_band`).
    # Collapse duplicates by value, keeping the later entry so gray-band
    # boundaries take precedence over the underlying cmap at the same value.
    deduped: dict[float, str | tuple[float, ...]] = {}
    for value, color in new_colors:
        deduped[value] = color
    return list(deduped.items())


class ThresholdedColormap(mcolors.LinearSegmentedColormap):
    """Colormap with a gray band over the thresholded range of a norm.

    [`VolumePlotter.add_volume`][confusius.plotting.VolumePlotter.add_volume] wraps
    every colormap it draws with in one of these, so that the part of the data
    masked by `threshold` reads as gray in the colorbar. The source colormap and
    the threshold are kept, so the same band can be rebuilt on another colormap
    with `with_source`; the docs gallery uses this to re-theme a drawn figure.

    Parameters
    ----------
    source : matplotlib.colors.Colormap
        Colormap the gray band is applied to.
    norm : matplotlib.colors.Normalize
        Norm positioning the band: it covers `norm(-threshold)` to
        `norm(threshold)` for `threshold_mode="lower"`, and everything outside
        that range for `threshold_mode="upper"`.
    threshold : float, optional
        Threshold on the absolute data values. If not provided, no band is added
        and the colormap matches `source`.
    threshold_mode : {"lower", "upper"}, default: "lower"
        Whether values below (`"lower"`) or above (`"upper"`) the threshold are
        grayed out.

    Raises
    ------
    ValueError
        If `norm.vmin` or `norm.vmax` is not finite.
    """

    def __init__(
        self,
        source: Colormap,
        norm: Normalize,
        threshold: float | None = None,
        threshold_mode: Literal["lower", "upper"] = "lower",
    ) -> None:
        if (
            norm.vmin is None
            or norm.vmax is None
            or not math.isfinite(norm.vmin)
            or not math.isfinite(norm.vmax)
        ):
            raise ValueError(
                "norm.vmin and norm.vmax must be finite, got "
                f"vmin={norm.vmin!r}, vmax={norm.vmax!r}."
            )

        if threshold is None:
            # No band in either mode: pass the source's stops through untouched.
            colors: list[ColorStop] = [
                (i / (source.N - 1), source(i / (source.N - 1)))
                for i in range(source.N)
            ]
        else:
            colors = _gray_band_colors(source, norm, abs(threshold), threshold_mode)
        # Preserve the source colormap's resolution. The default N=256 of
        # `LinearSegmentedColormap.from_list` collapses larger discrete cmaps such as
        # the atlas `ListedColormap` (N == number of regions, often >256), aliasing
        # high indices to wrong (or out-of-range) colours. Propagate under/over/bad
        # colours so the atlas's transparent under-colour for label 0 (background)
        # survives the rebuild. Cast to tuple because the getters return numpy arrays
        # but the kwargs expect a color-like.
        extremes: dict[str, Any] = {
            "under": tuple(source.get_under()),
            "over": tuple(source.get_over()),
            "bad": tuple(source.get_bad()),
        }
        base = mcolors.LinearSegmentedColormap.from_list(
            f"{source.name}_thresholded", colors, N=source.N, **extremes
        )
        # `from_list` is a staticmethod that always builds the base class, so the
        # subclass is initialised from the segment data it computed.
        super().__init__(
            base.name,
            base._segmentdata,  # ty: ignore[unresolved-attribute]
            N=base.N,
            **extremes,
        )
        self.source = source
        self.norm = norm
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def with_source(self, source: str | Colormap) -> ThresholdedColormap:
        """Return the same gray band applied to another colormap.

        Parameters
        ----------
        source : str or matplotlib.colors.Colormap
            Colormap to apply the band to, by name or instance.

        Returns
        -------
        ThresholdedColormap
            A new colormap with this one's `norm`, `threshold` and
            `threshold_mode`.
        """
        if isinstance(source, str):
            source = plt.get_cmap(source)
        return ThresholdedColormap(
            source, self.norm, self.threshold, self.threshold_mode
        )
