"""Read-only structure hierarchy tree for a loaded atlas."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem, QWidget

if TYPE_CHECKING:
    import treelib
    import xarray as xr
    from brainglobe_atlasapi.structure_class import StructuresDict

SWATCH_SIZE_PX = 12
"""Side length of the colour swatch drawn next to each structure acronym."""


class StructureTreeWidget(QTreeWidget):
    """Tree view of an atlas structure hierarchy, for browsing only.

    One row per BrainGlobe structure, nested following
    `ds.atlas.structures.tree`, with columns `acronym`, `name` and `id`. The acronym
    cell carries a colour swatch from the structure's `rgb_triplet`. Rows are not
    editable and selecting one has no side effect.

    Parameters
    ----------
    parent : qtpy.QtWidgets.QWidget, optional
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHeaderLabels(["acronym", "name", "id"])
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setMinimumWidth(320)

    def set_atlas(self, ds: xr.Dataset) -> None:
        """Rebuild the tree from an atlas Dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Atlas Dataset whose `.atlas.structures` hierarchy is displayed. The root
            structure becomes the single top-level item; the tree is expanded two
            levels deep.
        """
        self.clear()
        structures = ds.atlas.structures
        tree = structures.tree
        root_item = self._make_item(structures[tree.root])
        self.addTopLevelItem(root_item)
        self._add_children(root_item, tree, structures, tree.root)
        self.expandToDepth(1)
        self.resizeColumnToContents(0)

    def _add_children(
        self,
        parent_item: QTreeWidgetItem,
        tree: treelib.Tree,
        structures: StructuresDict,
        node_id: int,
    ) -> None:
        """Append the children of `node_id` under `parent_item`, recursively.

        Parameters
        ----------
        parent_item : qtpy.QtWidgets.QTreeWidgetItem
            Tree item receiving the child rows.
        tree : treelib.Tree
            Structure hierarchy.
        structures : brainglobe_atlasapi.structure_class.StructuresDict
            Structure metadata keyed by id.
        node_id : int
            Identifier of the structure whose children are appended.
        """
        # treelib types node ids as str; BrainGlobe uses integer structure ids.
        for node in tree.children(node_id):  # ty: ignore[invalid-argument-type]
            item = self._make_item(structures[node.identifier])
            parent_item.addChild(item)
            self._add_children(item, tree, structures, int(node.identifier))

    @staticmethod
    def _make_item(info: dict) -> QTreeWidgetItem:
        """Build one tree row from a BrainGlobe structure record.

        Parameters
        ----------
        info : dict
            Structure record with `id`, `acronym`, `name` and `rgb_triplet` keys.

        Returns
        -------
        qtpy.QtWidgets.QTreeWidgetItem
            Row with acronym, name and id columns and a colour swatch icon.
        """
        item = QTreeWidgetItem([info["acronym"], info["name"], str(info["id"])])
        swatch = QPixmap(SWATCH_SIZE_PX, SWATCH_SIZE_PX)
        swatch.fill(QColor(*info["rgb_triplet"]))
        item.setIcon(0, QIcon(swatch))
        return item
