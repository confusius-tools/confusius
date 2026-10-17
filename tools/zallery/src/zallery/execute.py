"""Execute percent-format examples with an explicit ipykernel."""

from __future__ import annotations

import random
import socket
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


def _setup_source(hook_module: str | None, hook_root: Path | None) -> str:
    """Return the code of the setup cell run before the example.

    Parameters
    ----------
    hook_module : str, optional
        Dotted name of the project hook module passed to
        `zallery.theme.setup_kernel`. If not provided, only the built-in swap runs.
    hook_root : pathlib.Path, optional
        Directory inserted at the front of the kernel's `sys.path` so that
        `hook_module` is importable. If not provided, the kernel's default path is
        used as is.

    Returns
    -------
    str
        Python source for the setup cell.
    """
    lines = []
    if hook_root is not None:
        lines += ["import sys", f"sys.path.insert(0, {str(hook_root)!r})"]
    lines += [
        "from zallery.theme import setup_kernel",
        f"setup_kernel({hook_module!r})",
    ]
    return "\n".join(lines)


_KERNEL_IP = "127.0.0.1"
"""Address the kernel binds its sockets to."""

_EPHEMERAL_PORT_FLOOR = 32768
"""Lowest port the OS hands to outgoing connections, unless the system says otherwise.

This is Linux's default (`/proc/sys/net/ipv4/ip_local_port_range`, read when
available); macOS and Windows start at 49152, so the same floor is safe there too.
"""

_LOWEST_KERNEL_PORT = 10000
"""Lowest port considered for a kernel socket, clear of well-known service ports."""


def _ephemeral_port_floor() -> int:
    """Return the start of the OS's ephemeral port range.

    Returns
    -------
    int
        The lower bound of `/proc/sys/net/ipv4/ip_local_port_range` when it can be
        read, `_EPHEMERAL_PORT_FLOOR` otherwise.
    """
    try:
        text = Path("/proc/sys/net/ipv4/ip_local_port_range").read_text()
        return int(text.split()[0])
    except (OSError, ValueError, IndexError):
        return _EPHEMERAL_PORT_FLOOR


def find_kernel_ports(count: int = 5) -> list[int]:
    """Return `count` free localhost ports below the OS's ephemeral range.

    jupyter_client picks kernel ports by binding to port 0, which draws from the
    ephemeral range, and closes them again before the kernel binds. Outgoing
    connections made in that window, nbclient's own ZMQ connects included, draw from
    the same range and can take a port first, so the kernel's bind fails with
    "Address already in use" (seen in CI on the heartbeat socket). Ports below the
    range are never assigned to outgoing connections, so they cannot be lost that way.

    Parameters
    ----------
    count : int, default: 5
        Number of ports to return, one per kernel channel.

    Returns
    -------
    list[int]
        Distinct ports that were free on `_KERNEL_IP` when probed, in increasing
        order from a random starting point.

    Raises
    ------
    RuntimeError
        If fewer than `count` free ports are found.
    """
    floor = _ephemeral_port_floor()
    # A random start keeps two concurrent builds, or a stuck process, from
    # fighting over the same first ports every time.
    start = random.randrange(_LOWEST_KERNEL_PORT, floor - count)
    ports: list[int] = []
    held: list[socket.socket] = []
    for port in range(start, floor):
        sock = socket.socket()
        try:
            sock.bind((_KERNEL_IP, port))
        except OSError:
            sock.close()
            continue
        held.append(sock)
        ports.append(port)
        if len(ports) == count:
            break
    for sock in held:
        sock.close()
    if len(ports) < count:
        raise RuntimeError(
            f"Could not find {count} free ports in [{start}, {floor}) for the kernel."
        )
    return ports


def read_example(source: Path) -> nbformat.NotebookNode:
    """Read one percent-format example into a notebook without executing it."""
    return jupytext.read(source)


def execute_example(
    source: Path,
    *,
    timeout: int = 600,
    on_cell_executed: OnCellExecuted | None = None,
    hook_module: str | None = None,
    hook_root: Path | None = None,
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
    hook_module : str, optional
        Dotted name of the module defining the project's `gallery_retheme` hook
        (see `zallery.theme.setup_kernel`). If not provided, only the built-in
        theme swap runs.
    hook_root : pathlib.Path, optional
        Directory put on the kernel's `sys.path` so `hook_module` is importable.

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
        _setup_source(hook_module, hook_root), metadata={"tags": ["_gallery_internal"]}
    )
    notebook.cells.insert(0, setup_cell)

    kernel_manager = KernelManager()
    kernel_manager.ip = _KERNEL_IP
    # With port caching on (the TCP default) the provisioner replaces preset ports
    # with ephemeral ones of its own; off, the ports below are written as given.
    kernel_manager.cache_ports = False
    (
        kernel_manager.shell_port,
        kernel_manager.iopub_port,
        kernel_manager.stdin_port,
        kernel_manager.control_port,
        kernel_manager.hb_port,
    ) = find_kernel_ports()
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
