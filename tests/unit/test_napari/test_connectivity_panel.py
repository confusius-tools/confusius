"""Tests for the napari functional-connectivity panel."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from confusius._napari._signals._store import SignalStore


def _make_panel(make_napari_viewer, sample_voxeldata_3dt):
    """Create a panel with one plotted VoxelData layer."""
    from confusius._napari._connectivity._panel import FunctionalConnectivityPanel

    viewer = make_napari_viewer()
    viewer.add_image(
        sample_voxeldata_3dt.values,
        name="data",
        metadata={"xarray": sample_voxeldata_3dt},
    )
    store = SignalStore()
    panel = FunctionalConnectivityPanel(viewer, store)
    return viewer, store, panel


def test_functional_connectivity_panel_computes_seed_map(
    make_napari_viewer, sample_voxeldata_3dt
):
    """A stored seed signal produces the expected voxel-wise Pearson map."""
    viewer, store, panel = _make_panel(make_napari_viewer, sample_voxeldata_3dt)
    seed = sample_voxeldata_3dt.isel(k=0, j=0, i=0).values
    store.pin_signal(
        "test-seed",
        "seed",
        np.arange(seed.size, dtype=float),
        seed,
        "#ffffff",
        "test",
    )

    panel._seed_combo.setCurrentIndex(1)
    panel._compute_clicked()

    result = viewer.layers["Seed map: seed"].data
    flat = sample_voxeldata_3dt.values.reshape(sample_voxeldata_3dt.sizes["time"], -1)
    expected = np.corrcoef(seed, flat, rowvar=False)[0, 1:].reshape(
        sample_voxeldata_3dt.sizes["k"],
        sample_voxeldata_3dt.sizes["j"],
        sample_voxeldata_3dt.sizes["i"],
    )
    assert_allclose(result, expected)


def test_seed_map_auto_updates_for_selected_time_window(
    make_napari_viewer, sample_voxeldata_3dt
):
    """Moving the time-coordinate slider updates the visible seed map."""
    viewer, store, panel = _make_panel(make_napari_viewer, sample_voxeldata_3dt)
    seed = sample_voxeldata_3dt.isel(k=0, j=0, i=0).values
    store.pin_signal(
        "test-seed",
        "seed",
        np.arange(seed.size, dtype=float),
        seed,
        "#ffffff",
        "test",
    )

    panel._seed_combo.setCurrentIndex(1)
    panel._compute_clicked()
    panel._time_slider.setValue((10.5, 11.5))

    result = viewer.layers["Seed map: seed"].data
    data = sample_voxeldata_3dt.values[1:4]
    flat = data.reshape(data.shape[0], -1)
    expected = np.corrcoef(seed[1:4], flat, rowvar=False)[0, 1:].reshape(
        sample_voxeldata_3dt.sizes["k"],
        sample_voxeldata_3dt.sizes["j"],
        sample_voxeldata_3dt.sizes["i"],
    )
    assert_allclose(result, expected)
    assert panel._time_start_label.text() == "Start: 10.5 s"
    assert panel._time_end_label.text() == "End: 11.5 s"


def test_mouse_seed_radius_averages_voxels(make_napari_viewer, sample_voxeldata_3dt):
    """Mouse seeds average all voxels inside the selected sphere."""
    viewer, _store, panel = _make_panel(make_napari_viewer, sample_voxeldata_3dt)
    prepared = panel._prepare_layer()
    panel._last_voxel = (1, 1, 1)
    panel._show_map(
        panel._corr_map(panel._mean_seed_trace(prepared, panel._last_voxel), prepared),
        "Mouse seed map",
    )
    panel._radius_spin.setValue(0.11)

    result = panel._mean_seed_trace(prepared, (1, 1, 1))

    values = np.moveaxis(sample_voxeldata_3dt.values, 0, 0)
    grids = np.ogrid[: values.shape[1], : values.shape[2], : values.shape[3]]
    spacings = [sample_voxeldata_3dt.fusi.spacing[dim] for dim in ("k", "j", "i")]
    mask = sum(
        ((grid - 1) * spacing) ** 2 for grid, spacing in zip(grids, spacings)
    ) <= 0.11**2
    flat = values.reshape(values.shape[0], -1)
    centered = flat - flat.mean(axis=0)
    expected = centered[:, mask.ravel()].mean(axis=1)
    assert panel._radius_spin.singleStep() == 0.1
    assert_allclose(result, expected)
    assert_allclose(
        viewer.layers["Mouse seed map"].data,
        panel._corr_map(expected, prepared).data,
    )
