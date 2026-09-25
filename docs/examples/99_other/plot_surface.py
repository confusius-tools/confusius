import numpy as np

import confusius as cf
from confusius._utils.napari import (
    get_napari_layer_geometry,
)
from confusius.xarray import create_voxeldata

atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_25um")

# Oblique grid: the atlas grid rotated 15 degrees about z (in the y/x plane), same sizes
# and spacing, re-centred on the volume so the rotated field of view still covers the
# brain. Content stays put in world space (identity pull transform), only the voxel
# lattice is oblique, so `get_meshes` goes through the oblique `.sel` path when clipping
# to one hemisphere.
reference = atlas.atlas.reference
spacing = np.array([reference.fusi.spacing[d] for d in ("k", "j", "i")])
theta = np.deg2rad(15)
rotation = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ]
)
center_mm = (np.array(reference.shape) - 1) / 2 * spacing
oblique = np.eye(4)
oblique[:3, :3] = rotation * spacing  # Columns: world direction of k/j/i times spacing.
oblique[:3, 3] = center_mm - rotation @ center_mm
target = create_voxeldata(
    np.zeros(reference.shape, dtype=np.float32),
    dims=("k", "j", "i"),
    voxel_to_world=oblique,
)
oblique_atlas = atlas.atlas.resample_like(target, np.eye(4))

rois = set(np.unique(oblique_atlas.annotation)) - {0, 997}

cf.atlas.get_atlas_meshes(oblique_atlas, 997, sides="right")


viewer, _ = cf.plotting.plot_napari(oblique_atlas.reference)

cf.plotting.plot_napari(atlas.atlas.reference, viewer=viewer, colormap="inferno")

cf.plotting.plot_atlas_mesh(atlas, 997, viewer=viewer, sides="right")
viewer, layer = cf.plotting.plot_surface(
    oblique_atlas.atlas.get_meshes("VISp")["VISp"], colormap="magenta", viewer=viewer
)
# The mesh, layer name, and color are all pulled from the atlas region.
oblique_atlas.atlas.plot.mesh("root", viewer=viewer)
