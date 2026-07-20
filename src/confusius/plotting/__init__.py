"""Plotting module for fUSI data."""

__all__ = [
    "draw_napari_labels",
    "labels_from_layer",
    "plot_atlas_mesh",
    "plot_carpet",
    "plot_composite",
    "plot_contours",
    "plot_matrix",
    "plot_motion_diagnostics",
    "plot_napari",
    "plot_stat_map",
    "plot_surface",
    "plot_volume",
    "VolumePlotter",
]

from confusius.plotting.image import (
    VolumePlotter,
    plot_carpet,
    plot_composite,
    plot_contours,
    plot_stat_map,
    plot_volume,
)
from confusius.plotting.matrix import plot_matrix
from confusius.plotting.motion import plot_motion_diagnostics
from confusius.plotting.napari import (
    draw_napari_labels,
    labels_from_layer,
    plot_atlas_mesh,
    plot_napari,
    plot_surface,
)
