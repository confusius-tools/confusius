"""PoC: live CPU seed-based connectivity map, seed = voxel under the mouse.

Same interaction as tools/poc_live_seedmap_gpu.py, but the smooth/aCompCor/cosine-clean
pipeline and the per-mousemove correlation map both run on CPU (confusius public API +
plain NumPy), to compare against the GPU version's latency.
"""

from __future__ import annotations

import time

import napari
import numpy as np

import confusius as cf
from benchmark_seed_map_gpu import prepare_example
from confusius.plotting import plot_napari


def main() -> None:
    data, _seed_masks, white_matter = prepare_example()
    moving = data.mean(dim="time").fusi.scale.db().compute()

    smoothed = cf.spatial.smooth_volume(data, fwhm=0.1).compute()
    acompcor = cf.signal.compute_compcor_confounds(
        smoothed, noise_mask=white_matter, n_components=1, variance_threshold=0.95
    )
    cleaned = cf.signal.clean(
        smoothed, low_cutoff=0.01, filter_method="cosine", confounds=acompcor
    ).compute()

    _t, k, j, i = cleaned.shape
    flat = cleaned.values.reshape(cleaned.shape[0], -1)
    # Cleaned data is static once computed, so center it and cache the per-voxel
    # norm once instead of redoing it on every mouse move.
    x_centered = flat - flat.mean(axis=0)
    denom_x = np.sqrt(np.sum(x_centered * x_centered, axis=0))

    def seed_corr_map(vk: int, vj: int, vi: int) -> np.ndarray:
        index = (vk * j + vj) * i + vi
        seed = x_centered[:, index]
        numerator = x_centered.T @ seed
        denom = denom_x * np.sqrt(np.sum(seed * seed))
        corr = np.where(denom == 0, 0.0, numerator / denom)
        return corr.reshape(k, j, i)

    viewer, anatomical_layer = plot_napari(moving, name="anatomical", colormap="gray")
    seed_map = moving.copy(data=np.zeros_like(moving.data))
    seed_map.name = "live seed map (CPU)"
    _viewer, seed_layer = plot_napari(
        seed_map,
        viewer=viewer,
        show_colorbar=False,
        show_scale_bar=False,
        colormap="twilight",
        contrast_limits=(-1, 1),
        blending="translucent",
        opacity=0.7,
    )
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "Hold Shift and move the mouse to set the seed voxel. (CPU)"

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
        t0 = time.perf_counter()
        seed_layer.data = seed_corr_map(vk, vj, vi)
        viewer.text_overlay.text = f"CPU seed map: {(time.perf_counter() - t0) * 1000:.1f} ms/call"

    viewer.mouse_move_callbacks.append(on_mouse_move)
    napari.run()


if __name__ == "__main__":
    main()
