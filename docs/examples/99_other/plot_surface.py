import confusius as cf

atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_25um")
viewer, _ = cf.plotting.plot_napari(atlas.atlas.reference)
# The mesh, layer name, and color are all pulled from the atlas region.
cf.plotting.plot_surface(atlas, "root", viewer=viewer)
