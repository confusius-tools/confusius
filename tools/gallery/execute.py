"""Execute percent-format examples with an explicit ipykernel."""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import jupytext
import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

OnCellExecuted = Callable[..., None]
"""Hook invoked after each cell finishes.

nbclient calls it as `hook(cell=..., cell_index=..., execute_reply=...)`; the
callable must accept those keyword arguments (typically via `**kwargs`). The
injected setup cell is `cell_index == 0`.
"""

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SETUP_SOURCE = "\n".join(
    [
        "import sys",
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})",
        "from tools.gallery.theme import setup_kernel",
        "setup_kernel()",
    ]
)
"""Code of the setup cell run before the example (see `tools.gallery.theme`)."""


def read_example(source: Path) -> nbformat.NotebookNode:
    """Read one percent-format example into a notebook without executing it."""
    return jupytext.read(source)


def execute_example(
    source: Path,
    *,
    timeout: int = 600,
    on_cell_executed: OnCellExecuted | None = None,
) -> tuple[nbformat.NotebookNode, float]:
    """Execute one percent-format `.py` example end-to-end.

    Parameters
    ----------
    source : pathlib.Path
        The example script.
    timeout : int, default: 600
        Per-cell timeout in seconds.
    on_cell_executed : OnCellExecuted, optional
        Hook called after each cell, including the setup cell at index 0.

    Returns
    -------
    notebook : nbformat.NotebookNode
        The executed notebook, with the same cells as `source`. Figure outputs
        carry their dark rendering under `metadata["image/png"]["dark"]`.
    seconds : float
        Wall-clock execution time.

    Raises
    ------
    RuntimeError
        If the kernel setup cell fails; theming would otherwise be silently off.
    """
    notebook = read_example(source)
    setup_cell = nbformat.v4.new_code_cell(
        _SETUP_SOURCE, metadata={"tags": ["_gallery_internal"]}
    )
    notebook.cells.insert(0, setup_cell)

    kernel_manager = KernelManager()
    kernel_manager.kernel_cmd = [  # ty: ignore[unresolved-attribute]
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]

    # allow_errors=True lets execution continue past failing cells, so the
    # traceback ends up as a regular `error` output that the renderer (see
    # tools/gallery/render.py) inlines into the rendered page, matching what
    # a normal Jupyter notebook would show after a failed run. A broken
    # example therefore degrades to "renders with a visible traceback"
    # instead of aborting the whole gallery build.
    with tempfile.TemporaryDirectory(prefix="confusius-gallery-") as working_dir:
        client = NotebookClient(
            notebook,
            km=kernel_manager,
            timeout=timeout,
            on_cell_executed=on_cell_executed,
            allow_errors=True,
            resources={"metadata": {"path": working_dir}},
        )
        client.owns_km = True

        start = time.perf_counter()
        client.execute()
        elapsed = time.perf_counter() - start

    # The setup cell is ours, not the example's: drop it so the notebook lines up
    # one to one with the source. Its own failure is a build error, not content.
    notebook.cells.pop(0)
    errors = [out for out in setup_cell.outputs if out.get("output_type") == "error"]
    if errors:
        traceback = "\n".join(errors[0].get("traceback", []))
        raise RuntimeError(f"Gallery kernel setup failed:\n{traceback}")
    return notebook, elapsed
