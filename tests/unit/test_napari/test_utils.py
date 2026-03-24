"""Unit tests for shared napari utilities."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import QToolBar

from confusius._napari._utils import (
    recolor_toolbar_icons,
    resolve_delimited_export_path,
    write_delimited_series,
)


def _solid_icon(color: str, size: QSize = QSize(16, 16)) -> QIcon:
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()
    return QIcon(pixmap)


def test_recolor_toolbar_icons_preserves_disabled_state(qtbot):
    toolbar = QToolBar()
    qtbot.addWidget(toolbar)

    action = toolbar.addAction("Back")
    action.setIcon(_solid_icon("#000000"))

    recolor_toolbar_icons(toolbar, "#ffffff")

    icon = action.icon()
    normal = icon.pixmap(QSize(16, 16), QIcon.Mode.Normal).toImage().pixelColor(8, 8)
    disabled = (
        icon.pixmap(QSize(16, 16), QIcon.Mode.Disabled).toImage().pixelColor(8, 8)
    )

    assert normal.red() == 255
    assert normal.green() == 255
    assert normal.blue() == 255
    assert disabled.alpha() < normal.alpha()


def test_recolor_toolbar_icons_uses_original_icon_on_retheme(qtbot):
    toolbar = QToolBar()
    qtbot.addWidget(toolbar)

    action = toolbar.addAction("Back")
    action.setIcon(_solid_icon("#000000"))

    recolor_toolbar_icons(toolbar, "#ffffff")
    recolor_toolbar_icons(toolbar, "#ff0000")

    pixel = (
        action.icon()
        .pixmap(QSize(16, 16), QIcon.Mode.Normal)
        .toImage()
        .pixelColor(8, 8)
    )

    assert pixel.red() == 255
    assert pixel.green() == 0
    assert pixel.blue() == 0


def test_resolve_delimited_export_path_uses_selected_filter():
    path, delimiter = resolve_delimited_export_path("/tmp/export", "CSV (*.csv)")
    assert path.suffix == ".csv"
    assert delimiter == ","

    path, delimiter = resolve_delimited_export_path("/tmp/export", "TSV (*.tsv)")
    assert path.suffix == ".tsv"
    assert delimiter == "\t"


def test_write_delimited_series_aligns_duplicate_time_values(tmp_path):
    out_path = tmp_path / "export.tsv"

    write_delimited_series(
        out_path,
        [
            ("series", [0, 0, 1], [10, 11, 12]),
            ("series", [0, 1, 2], [20, 21, 22]),
        ],
        delimiter="\t",
    )

    rows = [line.split("\t") for line in out_path.read_text().splitlines()]
    assert rows == [
        ["time", "series", "series_2"],
        ["0", "10", "20"],
        ["0", "11", ""],
        ["1", "12", "21"],
        ["2", "", "22"],
    ]


def test_write_delimited_series_rejects_mismatched_lengths(tmp_path):
    out_path = tmp_path / "broken.tsv"

    with pytest.raises(ValueError, match="different lengths"):
        write_delimited_series(
            out_path,
            [("series", [0, 1], [10])],
            delimiter="\t",
        )


def test_write_delimited_series_rejects_empty_input(tmp_path):
    out_path = tmp_path / "empty.tsv"

    with pytest.raises(ValueError, match="No plotted data"):
        write_delimited_series(out_path, [], delimiter="\t")
