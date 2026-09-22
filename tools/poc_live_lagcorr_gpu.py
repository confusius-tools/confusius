"""PoC: live GPU lag-correlation maps, seed = voxel under the mouse.

Same idea as tools/poc_live_seedmap_gpu.py (Shift + move to pick a seed voxel from
this recording's pre-cleaned, GPU-resident signal), but instead of a single zero-lag
correlation map, this computes cross-correlation at a range of lags around the seed
and shows two maps side by side in napari grid mode:

- max correlation: the signed correlation coefficient of largest magnitude across all
  tested lags (so anti-correlated regions show up negative, not just positive peaks).
- lag at max: the lag (in seconds) at which that peak occurs, lightly spatially
  smoothed (see `_LAG_SMOOTH_FWHM`) since per-voxel argmax-over-lags is a hard,
  noise-amplifying selection. Positive lag means the seed leads the voxel (the
  voxel's activity follows the seed's).

Cross-correlation reuses the global (already cosine/aCompCor-cleaned) centering and
per-voxel norm computed once for the whole recording, rather than re-centering on
each lag's truncated overlap window -- the standard simplification for fast lag maps
(e.g. Mitra et al. 2014) since it only under/over-shoots ambiguously near the edges of
a small lag range.

See tools/benchmark_seed_map_gpu.py for the smoothing/cleaning pipeline this reuses.
"""

from __future__ import annotations

import cupy as cp
import napari
import numpy as np
from benchmark_seed_map_gpu import (
    gpu_clean_cosine,
    gpu_compcor,
    gpu_smooth,
    prepare_example,
)
from cupyx.scipy import ndimage as cp_ndimage

from confusius.plotting import plot_napari

_MAX_LAG_SECONDS = 4.0
_LAG_SMOOTH_FWHM = 0.15
"""Spatial FWHM (mm) applied to the lag map only. Per-voxel argmax-over-lags is a hard,
noise-amplifying selection even where the underlying correlation surface is smooth, so
lag maps are conventionally smoothed post-hoc (e.g. Mitra et al. 2014)."""


def main() -> None:
    data, _seed_masks, white_matter = prepare_example()
    moving = data.mean(dim="time").fusi.scale.db().compute()

    dt = float(np.median(np.diff(data.time.values.astype(float))))
    smoothed = gpu_smooth(data.astype(np.float32), 0.1)
    acompcor, _explained = gpu_compcor(smoothed, np.asarray(white_matter.data))
    cleaned = gpu_clean_cosine(smoothed, acompcor, dt, 0.01)

    t, k, j, i = cleaned.shape
    flat = cleaned.reshape(t, -1)
    x_centered = flat - flat.mean(axis=0)
    denom_x = cp.sqrt(cp.sum(x_centered * x_centered, axis=0))

    max_lag = max(1, round(_MAX_LAG_SECONDS / dt))
    lags = np.arange(-max_lag, max_lag + 1)
    lags_gpu = cp.asarray(lags, dtype=cp.float32)

    # Same fwhm-to-sigma conversion as gpu_smooth, but computed once here and applied
    # spatially only (k is already a singleton slice dim, so its sigma is moot).
    sigma_factor = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spacing = moving.fusi.spacing
    lag_sigma = [_LAG_SMOOTH_FWHM * sigma_factor / float(spacing[dim]) for dim in ("k", "j", "i")]

    def lag_corr_maps(vk: int, vj: int, vi: int) -> tuple[np.ndarray, np.ndarray]:
        index = (vk * j + vj) * i + vi
        seed = x_centered[:, index]
        denom_seed = cp.sqrt(cp.sum(seed * seed))

        # Column L of `shifted` holds the seed shifted so that shifted[:, L][t] ==
        # seed[t - lags[L]], zero-padded outside the valid overlap. Multiplying by
        # `x_centered` (zero outside the overlap contributes nothing) is exactly the
        # truncated dot product for that lag, computed for every lag in one matmul.
        shifted = cp.zeros((t, len(lags)), dtype=x_centered.dtype)
        for idx, lag in enumerate(lags):
            lag = int(lag)
            if lag >= 0:
                shifted[lag:, idx] = seed[: t - lag]
            else:
                shifted[: t + lag, idx] = seed[-lag:]

        corr = (x_centered.T @ shifted) / (denom_x[:, None] * denom_seed)  # (n_voxels, n_lags)
        # argmax on the signed value would only ever find positive peaks, silently
        # missing voxels whose strongest relationship to the seed is an anti-correlation
        # at some lag. Pick the lag with the largest magnitude, but keep the signed
        # correlation for display so anti-correlated regions show up as negative.
        lag_index = cp.argmax(cp.abs(corr), axis=1)
        max_corr = cp.take_along_axis(corr, lag_index[:, None], axis=1)[:, 0]
        lag_at_max = lags_gpu[lag_index].reshape(k, j, i) * dt
        lag_at_max = cp_ndimage.gaussian_filter(lag_at_max, sigma=lag_sigma)
        return max_corr.reshape(k, j, i).get(), lag_at_max.get()

    # plot_napari carries `moving`'s VoxelToWorldIndex (scale/translate/units) onto the
    # layers, so the anatomical image and both maps line up in world coordinates
    # instead of assuming unit voxel spacing.
    viewer, anatomical_layer = plot_napari(moving, name="anatomical", colormap="gray")
    max_corr_map = moving.copy(data=np.zeros_like(moving.data))
    max_corr_map.name = "max correlation"
    _viewer, max_corr_layer = plot_napari(
        max_corr_map,
        viewer=viewer,
        show_colorbar=True,
        show_scale_bar=False,
        colormap="twilight",
        contrast_limits=(-1, 1),
    )
    lag_map = moving.copy(data=np.zeros_like(moving.data))
    lag_map.name = "lag at max (s)"
    _viewer, lag_layer = plot_napari(
        lag_map,
        viewer=viewer,
        show_colorbar=True,
        show_scale_bar=False,
        colormap="coolwarm",
        contrast_limits=(-_MAX_LAG_SECONDS, _MAX_LAG_SECONDS),
    )

    viewer.grid.enabled = True
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "Hold Shift and move the mouse to set the seed voxel."

    last_voxel = None

    def on_mouse_move(viewer, event) -> None:
        nonlocal last_voxel
        if "Shift" not in event.modifiers:
            return
        vk, vj, vi = (round(c) for c in anatomical_layer.world_to_data(viewer.cursor.position))
        if not (0 <= vk < k and 0 <= vj < j and 0 <= vi < i):
            return
        if (vk, vj, vi) == last_voxel:
            return
        last_voxel = (vk, vj, vi)
        max_corr_layer.data, lag_layer.data = lag_corr_maps(vk, vj, vi)

    viewer.mouse_move_callbacks.append(on_mouse_move)
    napari.run()


if __name__ == "__main__":
    main()
