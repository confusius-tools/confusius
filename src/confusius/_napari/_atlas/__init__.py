"""Atlas panel subpackage: BrainGlobe atlases, region masks and fUSI templates."""

from confusius._napari._atlas._panel import AtlasPanel
from confusius._napari._atlas._tree import StructureTreeWidget

__all__ = ["AtlasPanel", "StructureTreeWidget"]
