"""Execute a percent-format example with an explicit ipykernel."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jupytext
import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def execute_example(
    source: Path,
    *,
    timeout: int = 600,
) -> tuple[nbformat.NotebookNode, float]:
    """Execute one percent-format ``.py`` example end-to-end.

    Reads ``source`` with jupytext, runs all cells against a kernel launched
    from ``sys.executable`` (so the running interpreter matches the project's
    locked virtual environment), and returns the executed notebook plus the
    wall-clock time the execution took.

    Parameters
    ----------
    source : pathlib.Path
        Path to a jupytext percent-format ``.py`` file.
    timeout : int, default: 600
        Per-cell timeout in seconds.

    Returns
    -------
    nb : nbformat.NotebookNode
        The notebook with outputs filled in.
    seconds : float
        Wall-clock time spent in ``client.execute``.

    Raises
    ------
    RuntimeError
        If a cell raises. The original exception is wrapped to attach the
        example path.
    """
    nb = jupytext.read(source)

    km = KernelManager()
    km.kernel_cmd = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]

    client = NotebookClient(nb, km=km, timeout=timeout)
    # NotebookClient sets owns_km=False whenever a km is passed in, which
    # disables its cleanup_kc default and leaks zmq sockets. We do want the
    # client to own the lifecycle here, so flip the flag back on.
    client.owns_km = True

    start = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as exc:
        raise RuntimeError(f"Failed to execute {source}: {exc}") from exc
    seconds = time.perf_counter() - start

    return nb, seconds
