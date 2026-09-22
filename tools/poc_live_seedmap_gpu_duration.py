"""PoC: live GPU seed map using only a selected time window.

Same interaction as tools/poc_live_seedmap_gpu.py, but docked start/end sliders limit
the correlation window to any cleaned recording period. This is for checking how seed
maps change with acquisition length and placement.
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
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from superqt import QRangeSlider

from confusius.plotting import plot_napari


def main() -> None:
    data, _seed_masks, white_matter = prepare_example()
    moving = data.mean(dim="time").fusi.scale.db().compute()

    dt = float(np.median(np.diff(data.time.values.astype(float))))
    smoothed = gpu_smooth(data.astype(np.float32), 0.1)
    acompcor, _explained = gpu_compcor(smoothed, np.asarray(white_matter.data))
    cleaned = gpu_clean_cosine(smoothed, acompcor, dt, 0.01)

    t, k, j, i = cleaned.shape
    flat = cleaned.reshape(t, -1)
    current_start = 0
    current_stop = t
    current_x_centered = flat - flat.mean(axis=0)
    current_denom_x = cp.sqrt(cp.sum(current_x_centered * current_x_centered, axis=0))

    def seed_corr_map(vk: int, vj: int, vi: int) -> np.ndarray:
        index = (vk * j + vj) * i + vi
        seed = current_x_centered[:, index]
        numerator = current_x_centered.T @ seed
        denom = current_denom_x * cp.sqrt(cp.sum(seed * seed))
        corr = cp.where(denom == 0, 0.0, numerator / denom)
        return corr.reshape(k, j, i).get()

    viewer, anatomical_layer = plot_napari(moving, name="anatomical", colormap="gray")
    seed_map = moving.copy(data=np.zeros_like(moving.data))
    seed_map.name = "live seed map (time window)"
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

    time_values = data.time.values.astype(float)
    duration_label = QLabel()
    time_slider = QRangeSlider(Qt.Horizontal)
    time_slider.setRange(0, t - 1)
    time_slider.setValue((0, t - 1))

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(duration_label)
    layout.addWidget(time_slider)
    viewer.window.add_dock_widget(panel, area="right", name="Seed-map time window")

    last_voxel = None

    def update_window(_value: tuple[int, int] | None = None) -> None:
        nonlocal current_start, current_stop, current_x_centered, current_denom_x
        start, end = (int(v) for v in time_slider.value())
        if start >= end:
            start, end = (end - 1, end) if end == t - 1 else (start, start + 1)
            time_slider.blockSignals(True)
            time_slider.setValue((start, end))
            time_slider.blockSignals(False)
        current_start = start
        current_stop = end + 1
        window = flat[current_start:current_stop]
        current_x_centered = window - window.mean(axis=0)
        current_denom_x = cp.sqrt(cp.sum(current_x_centered * current_x_centered, axis=0))
        duration = float(time_values[current_stop - 1]) - float(time_values[current_start])
        duration_label.setText(
            f"Using frames {current_start}..{current_stop - 1} "
            f"({current_stop - current_start}/{t}, {duration:.1f} s)"
        )
        if last_voxel is not None:
            seed_layer.data = seed_corr_map(*last_voxel)

    time_slider.valuesChanged.connect(update_window)
    update_window()

    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Drag the start/end sliders to choose the time window; "
        "hold Shift and move the mouse to set the seed voxel."
    )

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
