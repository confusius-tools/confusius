"""PoC: live GPU seed-based connectivity map, seed = voxel under the mouse.

Standalone demo, independent of the confu napari plugin. Loads one recording, runs
the smooth/aCompCor/cosine-clean steps once on GPU (the "pre-corrected/denoised"
recording from the seed-map benchmark), then keeps the cleaned volume resident on
GPU. Hold Shift and move the mouse over the anatomical image to recompute and
display the seed-based correlation map in real time.

See tools/benchmark_seed_map_gpu.py for the pipeline this reuses and its timings.
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

from confusius.plotting import plot_napari


def main() -> None:
    data, _seed_masks, white_matter = prepare_example()
    moving = data.mean(dim="time").fusi.scale.db().compute()

    dt = float(np.median(np.diff(data.time.values.astype(float))))
    smoothed = gpu_smooth(data.astype(np.float32), 0.1)
    acompcor, _explained = gpu_compcor(smoothed, np.asarray(white_matter.data))
    cleaned = gpu_clean_cosine(smoothed, acompcor, dt, 0.01)

    _t, k, j, i = cleaned.shape
    flat = cleaned.reshape(cleaned.shape[0], -1)
    # Cleaned data is static once computed, so center it and cache the per-voxel
    # norm once instead of redoing it on every mouse move.
    x_centered = flat - flat.mean(axis=0)
    denom_x = cp.sqrt(cp.sum(x_centered * x_centered, axis=0))

    def seed_corr_map(vk: int, vj: int, vi: int) -> np.ndarray:
        index = (vk * j + vj) * i + vi
        seed = x_centered[:, index]
        numerator = x_centered.T @ seed
        denom = denom_x * cp.sqrt(cp.sum(seed * seed))
        corr = cp.where(denom == 0, 0.0, numerator / denom)
        return corr.reshape(k, j, i).get()

    # plot_napari carries `moving`'s VoxelToWorldIndex (scale/translate/units) onto the
    # layers, so the anatomical image and the overlay line up in world coordinates
    # instead of assuming unit voxel spacing.
    viewer, anatomical_layer = plot_napari(moving, name="anatomical", colormap="gray")
    seed_map = moving.copy(data=np.zeros_like(moving.data))
    seed_map.name = "live seed map"
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
        seed_layer.data = seed_corr_map(vk, vj, vi)

    viewer.mouse_move_callbacks.append(on_mouse_move)
    napari.run()


if __name__ == "__main__":
    main()
