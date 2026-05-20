"""Tests for tools.gallery.execute."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.gallery.execute import execute_example


def test_execute_runs_jupytext_py_and_returns_executed_nb(tmp_path: Path) -> None:
    src = tmp_path / "ex.py"
    src.write_text("# %% [markdown]\n# # Hello\n\n# %%\nx = 6 * 7\nprint(x)\n")

    nb, seconds = execute_example(src, timeout=60)

    assert seconds > 0
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 1
    streams = [
        out
        for out in code_cells[0].outputs
        if out.get("output_type") == "stream" and out.get("name") == "stdout"
    ]
    assert any("42" in out.get("text", "") for out in streams)


def test_execute_uses_current_python_interpreter(tmp_path: Path) -> None:
    """The kernel must be the same Python that runs the script."""
    src = tmp_path / "ex.py"
    src.write_text(f"# %%\nimport sys\nassert sys.executable == {sys.executable!r}\n")
    # Will raise if the kernel is some other interpreter.
    execute_example(src, timeout=60)


def test_execute_captures_cell_error_as_output_and_continues(tmp_path: Path) -> None:
    """A failing cell records its traceback and execution continues to the next cell.

    The renderer (`tools.gallery.render`) already inlines `error` outputs into
    the rendered Markdown the same way a normal Jupyter notebook would, so the
    executor must not abort on the first failing cell.
    """
    src = tmp_path / "broken.py"
    src.write_text("# %%\nraise RuntimeError('boom')\n\n# %%\nprint('after')\n")

    nb, _ = execute_example(src, timeout=60)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 2

    error_outputs = [
        out for out in code_cells[0].outputs if out.get("output_type") == "error"
    ]
    assert error_outputs, "the failing cell did not produce an error output"
    assert error_outputs[0].get("ename") == "RuntimeError"
    assert "boom" in error_outputs[0].get("evalue", "")

    after_streams = [
        out
        for out in code_cells[1].outputs
        if out.get("output_type") == "stream" and out.get("name") == "stdout"
    ]
    assert any("after" in out.get("text", "") for out in after_streams)
