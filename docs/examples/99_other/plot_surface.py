import confusius as cf

atlas = cf.atlas.Atlas.from_brainglobe("allen_mouse_25um")
viewer, _ = cf.plotting.plot_napari(atlas.reference)
cf.plotting.plot_surface(atlas.get_mesh("root"), colormap="magenta")
