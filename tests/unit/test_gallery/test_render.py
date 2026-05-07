"""Tests for tools.gallery.render."""

from __future__ import annotations

from pathlib import Path

import nbformat

from tools.gallery.render import render_notebook


def _make_nb() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# Hello\n\nIntro text."))
    code = nbformat.v4.new_code_cell("print('hi')")
    code.outputs = [
        nbformat.v4.new_output(
            output_type="stream",
            name="stdout",
            text="hi\n",
        ),
    ]
    nb.cells.append(code)

    # 1x1 transparent PNG.
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YA"
        "AAAASUVORK5CYII="
    )
    plot_cell = nbformat.v4.new_code_cell("plt.plot([1, 2, 3])")
    plot_cell.outputs = [
        nbformat.v4.new_output(
            output_type="display_data",
            data={"image/png": png_b64},
            metadata={},
        ),
    ]
    nb.cells.append(plot_cell)
    return nb


def test_render_writes_markdown_with_code_and_output(tmp_path: Path) -> None:
    nb = _make_nb()
    md_path, _ = render_notebook(
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=4.3,
    )

    md = md_path.read_text()
    assert md.startswith("# Hello")
    assert "```python\nprint('hi')\n```" in md
    assert "```\nhi\n```" in md
    assert "ex_output/cell_03_0.png" in md
    assert "**Total running time:** 4.3 s" in md
    assert "[Download .py](ex.py)" in md
    assert "[Download .ipynb](ex.ipynb)" in md


def test_render_writes_image_files_into_output_folder(tmp_path: Path) -> None:
    nb = _make_nb()
    render_notebook(nb, out_dir=tmp_path, base_name="ex", runtime_seconds=1.0)

    img = tmp_path / "ex_output" / "cell_03_0.png"
    assert img.is_file()
    assert img.stat().st_size > 0


def test_render_returns_thumbnail_path_pointing_at_last_image(tmp_path: Path) -> None:
    nb = _make_nb()
    _, thumbnail = render_notebook(
        nb, out_dir=tmp_path, base_name="ex", runtime_seconds=1.0
    )
    assert thumbnail == tmp_path / "ex_thumb.png"
    assert thumbnail.is_file()


def test_render_handles_notebook_without_images(tmp_path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("x = 1"))

    _, thumbnail = render_notebook(
        nb, out_dir=tmp_path, base_name="ex", runtime_seconds=0.1
    )
    assert thumbnail is None
