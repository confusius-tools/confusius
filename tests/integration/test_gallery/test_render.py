"""Unit tests for the gallery Markdown renderer."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from tools.gallery.render import render_notebook


def _notebook_with_outputs(outputs: list[dict[str, object]]) -> nbformat.NotebookNode:
    """Return a one-code-cell notebook whose single cell carries ``outputs``."""
    cell = nbformat.v4.new_code_cell("print('x')")
    cell.outputs = outputs
    nb = nbformat.v4.new_notebook()
    nb.cells = [cell]
    return nb


def test_render_notebook_shows_outputs_on_count_mismatch(tmp_path: Path) -> None:
    """A light/dark output-count mismatch reports the actual outputs, not just counts."""
    source = _notebook_with_outputs([])
    light = _notebook_with_outputs(
        [
            {"output_type": "stream", "name": "stdout", "text": "shared"},
            {
                "output_type": "stream",
                "name": "stderr",
                "text": "only-in-light warning",
            },
        ]
    )
    dark = _notebook_with_outputs(
        [{"output_type": "stream", "name": "stdout", "text": "shared"}]
    )

    with pytest.raises(ValueError, match="2 light outputs and 1 dark") as excinfo:
        render_notebook(
            source,
            light,
            dark,
            out_dir=tmp_path,
            base_name="demo",
            runtime_seconds=1.0,
        )

    message = str(excinfo.value)
    # The richer message exists so the offending extra output is visible; a
    # counts-only message would not contain the stray warning text.
    assert "only-in-light warning" in message
    assert "Light outputs (2):" in message
    assert "Dark outputs (1):" in message
