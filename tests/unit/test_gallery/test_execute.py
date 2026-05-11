"""Tests for tools.gallery.execute."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.gallery.execute import execute_example


def test_execute_runs_jupytext_py_and_returns_executed_nb(tmp_path: Path) -> None:
    src = tmp_path / "ex.py"
    src.write_text("# %% [markdown]\n# # Hello\n\n# %%\nx = 6 * 7\nprint(x)\n")

    nb, seconds = execute_example(src, timeout=60)

    assert seconds > 0
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 1
    streams = [
        o
        for o in code_cells[0].outputs
        if o.get("output_type") == "stream" and o.get("name") == "stdout"
    ]
    assert any("42" in o.get("text", "") for o in streams)


def test_execute_uses_current_python_interpreter(tmp_path: Path) -> None:
    """The kernel must be the same Python that runs the script."""
    src = tmp_path / "ex.py"
    src.write_text(f"# %%\nimport sys\nassert sys.executable == {sys.executable!r}\n")
    # Will raise if the kernel is some other interpreter.
    execute_example(src, timeout=60)


def test_execute_re_raises_with_example_path(tmp_path: Path) -> None:
    src = tmp_path / "broken.py"
    src.write_text("# %%\nraise RuntimeError('boom')\n")

    with pytest.raises(RuntimeError) as excinfo:
        execute_example(src, timeout=60)

    assert "broken.py" in str(excinfo.value)
