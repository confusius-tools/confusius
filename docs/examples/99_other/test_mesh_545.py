import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

atlas = BrainGlobeAtlas("allen_mouse_100um")

np.sum(atlas.get_structure_mask(545))

atlas.structures[545]["mesh"]
