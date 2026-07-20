"""Tests for tools.gallery.render."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from tools.gallery.render import render_notebook


def _make_nb(*, thumbnail: bool = False) -> nbformat.NotebookNode:
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
    if thumbnail:
        plot_cell.metadata["tags"] = ["thumbnail"]

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
        nb,
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=4.3,
    )

    md = md_path.read_text()
    assert md.startswith("# Hello")
    # Input code stays a bare highlighted block; output is wrapped so docs CSS can
    # wrap long lines instead of scrolling.
    assert "```python\nprint('hi')\n```" in md
    assert '<div class="gallery-output" markdown>\n\n```\nhi\n```\n\n</div>' in md
    assert "ex_output_light/cell_03_0_light.png#only-light" in md
    assert "ex_output_dark/cell_03_0_dark.png#only-dark" in md
    assert "**Total running time:** 4.3 s" in md
    assert "[Download .py](ex.py)" in md
    assert "[Download .ipynb](ex.ipynb)" in md


def test_render_writes_image_files_into_output_folder(tmp_path: Path) -> None:
    nb = _make_nb()
    render_notebook(
        nb,
        nb,
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=1.0,
    )

    light_img = tmp_path / "ex_output_light" / "cell_03_0_light.png"
    dark_img = tmp_path / "ex_output_dark" / "cell_03_0_dark.png"
    assert light_img.is_file()
    assert light_img.stat().st_size > 0
    assert dark_img.is_file()
    assert dark_img.stat().st_size > 0


def test_render_returns_thumbnail_paths_for_tagged_output(tmp_path: Path) -> None:
    nb = _make_nb(thumbnail=True)
    _, thumbnail = render_notebook(
        nb,
        nb,
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=1.0,
    )
    assert thumbnail == (
        tmp_path / "ex_thumb_light.png",
        tmp_path / "ex_thumb_dark.png",
    )
    assert (tmp_path / "ex_thumb_light.png").is_file()
    assert (tmp_path / "ex_thumb_dark.png").is_file()


def test_render_collapses_code_cell_with_tag(tmp_path: Path) -> None:
    """A `collapse: <title>` tag hides the code behind a titled admonition."""
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# Hello"))
    code = nbformat.v4.new_code_cell("print('hi')")
    code.metadata["tags"] = ["collapse: Setup and registration"]
    code.outputs = [
        nbformat.v4.new_output(output_type="stream", name="stdout", text="hi\n"),
    ]
    nb.cells.append(code)

    md_path, _ = render_notebook(
        nb, nb, nb, out_dir=tmp_path, base_name="ex", runtime_seconds=1.0
    )
    md = md_path.read_text()
    # The code is wrapped in a collapsed admonition with the given title, indented as its
    # content; the output still renders (below, un-indented) so only the input is hidden.
    assert '??? example "Setup and registration"' in md
    assert "    ```python\n    print('hi')\n    ```" in md
    assert '<div class="gallery-output" markdown>\n\n```\nhi\n```' in md


def test_render_collapse_tag_without_title_uses_default(tmp_path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    code = nbformat.v4.new_code_cell("x = 1")
    code.metadata["tags"] = ["collapse"]
    nb.cells.append(code)

    md_path, _ = render_notebook(
        nb, nb, nb, out_dir=tmp_path, base_name="ex", runtime_seconds=1.0
    )
    md = md_path.read_text()
    assert '??? example "Show code"' in md
    assert "    ```python\n    x = 1\n    ```" in md


def test_render_collapse_tag_with_admonition_type(tmp_path: Path) -> None:
    """`collapse[<type>]` picks the admonition type, with or without a title."""
    nb = nbformat.v4.new_notebook()
    titled = nbformat.v4.new_code_cell("x = 1")
    titled.metadata["tags"] = ["collapse[warning]: Advanced setup"]
    nb.cells.append(titled)
    untitled = nbformat.v4.new_code_cell("y = 2")
    untitled.metadata["tags"] = ["collapse[tip]"]
    nb.cells.append(untitled)

    md_path, _ = render_notebook(
        nb, nb, nb, out_dir=tmp_path, base_name="ex", runtime_seconds=1.0
    )
    md = md_path.read_text()
    assert '??? warning "Advanced setup"' in md
    assert '??? tip "Show code"' in md

def test_render_handles_notebook_without_images(tmp_path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("x = 1"))

    _, thumbnail = render_notebook(
        nb,
        nb,
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=0.1,
    )
    assert thumbnail is None


def test_render_adds_binder_button_when_url_given(tmp_path: Path) -> None:
    nb = _make_nb()
    url = "https://mybinder.org/v2/gh/owner/repo/main?urlpath=lab/tree/foo/ex.py"
    md_path, _ = render_notebook(
        nb,
        nb,
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=1.0,
        binder_url=url,
    )

    md = md_path.read_text()
    assert f"[Launch in Binder]({url})" in md
    assert ".md-button--primary" in md


def test_render_omits_binder_button_when_url_missing(tmp_path: Path) -> None:
    nb = _make_nb()
    md_path, _ = render_notebook(
        nb,
        nb,
        nb,
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=1.0,
    )

    md = md_path.read_text()
    assert "Launch in Binder" not in md
    assert ".md-button--primary" not in md


def test_render_uses_outputs_from_light_and_dark_notebooks(tmp_path: Path) -> None:
    """Image bytes come from light_notebook / dark_notebook, not the un-executed source."""
    import base64

    light_b64 = base64.b64encode(b"light_pixel").decode()
    dark_b64 = base64.b64encode(b"dark_pixel").decode()

    source_nb = nbformat.v4.new_notebook()
    source_nb.cells.append(nbformat.v4.new_code_cell("plt.plot([1])"))

    def _nb_with_png(b64: str) -> nbformat.NotebookNode:
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell("plt.plot([1])")
        cell.outputs = [
            nbformat.v4.new_output(
                output_type="display_data",
                data={"image/png": b64},
                metadata={},
            )
        ]
        nb.cells.append(cell)
        return nb

    render_notebook(
        source_nb,
        _nb_with_png(light_b64),
        _nb_with_png(dark_b64),
        out_dir=tmp_path,
        base_name="ex",
        runtime_seconds=1.0,
    )

    light_bytes = (tmp_path / "ex_output_light" / "cell_01_0_light.png").read_bytes()
    dark_bytes = (tmp_path / "ex_output_dark" / "cell_01_0_dark.png").read_bytes()
    assert light_bytes == b"light_pixel"
    assert dark_bytes == b"dark_pixel"
    assert light_bytes != dark_bytes


def test_render_raises_when_light_and_dark_output_counts_differ(
    tmp_path: Path,
) -> None:
    """Light/dark output mismatch is a hard error.

    Independent kernels would occasionally emit a one-time stream output
    (download progress, first-import warning) in only one of the two builds.
    Silently dropping the extras hid real divergence between the two rendered
    notebooks, so the renderer now refuses to pair mismatched cells. The fix
    is to remove the non-determinism (pre-warming caches, etc.) before the
    gallery runs, not to relax the renderer.
    """
    source_nb = nbformat.v4.new_notebook()
    source_nb.cells.append(nbformat.v4.new_code_cell("print('a')\nx = 1\nx"))

    light_nb = nbformat.v4.new_notebook()
    light_cell = nbformat.v4.new_code_cell("print('a')\nx = 1\nx")
    light_cell.outputs = [
        nbformat.v4.new_output(output_type="stream", name="stdout", text="a\n"),
        nbformat.v4.new_output(
            output_type="execute_result",
            data={"text/plain": "1"},
            execution_count=1,
            metadata={},
        ),
    ]
    light_nb.cells.append(light_cell)

    dark_nb = nbformat.v4.new_notebook()
    dark_cell = nbformat.v4.new_code_cell("print('a')\nx = 1\nx")
    # Dark missed the stream output entirely.
    dark_cell.outputs = [
        nbformat.v4.new_output(
            output_type="execute_result",
            data={"text/plain": "1"},
            execution_count=1,
            metadata={},
        ),
    ]
    dark_nb.cells.append(dark_cell)

    with pytest.raises(ValueError, match="light outputs and"):
        render_notebook(
            source_nb,
            light_nb,
            dark_nb,
            out_dir=tmp_path,
            base_name="ex",
            runtime_seconds=1.0,
        )
