"""Unit tests for the StructureTreeWidget."""

from __future__ import annotations

import pytest

from confusius._napari._atlas._tree import StructureTreeWidget


@pytest.fixture
def tree(qtbot, atlas_ds):
    widget = StructureTreeWidget()
    qtbot.addWidget(widget)
    widget.set_atlas(atlas_ds)
    return widget


def test_hierarchy_follows_structure_tree(tree) -> None:
    assert tree.topLevelItemCount() == 1
    root = tree.topLevelItem(0)
    assert [root.text(col) for col in range(3)] == ["root", "whole brain", "997"]
    assert root.childCount() == 1

    child = root.child(0)
    assert [child.text(col) for col in range(3)] == ["ch", "child region", "10"]
    assert child.childCount() == 1

    grandchild = child.child(0)
    assert [grandchild.text(col) for col in range(3)] == [
        "gc",
        "grandchild region",
        "20",
    ]
    assert grandchild.childCount() == 0


def test_expanded_two_levels_deep(tree) -> None:
    root = tree.topLevelItem(0)
    assert root.isExpanded()
    assert root.child(0).isExpanded()


def test_rows_are_read_only(tree) -> None:
    from qtpy.QtWidgets import QAbstractItemView

    assert tree.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers


def test_swatch_matches_rgb_triplet(tree) -> None:
    child = tree.topLevelItem(0).child(0)
    pixmap = child.icon(0).pixmap(12, 12)
    color = pixmap.toImage().pixelColor(6, 6)
    assert (color.red(), color.green(), color.blue()) == (255, 0, 0)


def test_set_atlas_replaces_previous_tree(tree, atlas_ds) -> None:
    tree.set_atlas(atlas_ds)
    assert tree.topLevelItemCount() == 1


def test_only_acronym_column_stretches(tree) -> None:
    from qtpy.QtWidgets import QHeaderView

    header = tree.header()
    assert not header.stretchLastSection()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive
    assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.Interactive
    assert tree.indentation() == 12


def test_name_column_starts_at_half_its_content_width(tree) -> None:
    half_width = tree.columnWidth(1)
    tree.resizeColumnToContents(1)
    assert half_width == tree.columnWidth(1) // 2
