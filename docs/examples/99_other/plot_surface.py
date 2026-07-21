import numpy as np

import confusius as cf

atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_25um")


rois = set(np.unique(atlas.annotation)) - {0, 997}

viewer, _ = cf.plotting.plot_napari(atlas.atlas.reference)
cf.plotting.plot_atlas_mesh(atlas, rois, viewer=viewer)
viewer, layer = cf.plotting.plot_surface(
    atlas.atlas.get_mesh("VISp"), colormap="magenta", viewer=viewer
)
# The mesh, layer name, and color are all pulled from the atlas region.
atlas.atlas.plot.mesh("root", viewer=viewer)
