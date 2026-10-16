"""Tests for tools.gallery.execute."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from tools.gallery import execute
from tools.gallery.execute import execute_example, find_kernel_ports


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


def test_execute_attaches_dark_rendering_to_figure_outputs(tmp_path: Path) -> None:
    """One run yields both themes: the dark PNG rides along in the output metadata."""
    src = tmp_path / "ex.py"
    src.write_text("# %%\nimport matplotlib.pyplot as plt\n_ = plt.plot([1, 2, 3])\n")

    nb, _ = execute_example(src, timeout=120)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 1, "the injected setup cell must not leak out"
    images = [
        out
        for out in code_cells[0].outputs
        if out.get("output_type") == "display_data" and "image/png" in out["data"]
    ]
    assert len(images) == 1
    dark = images[0]["metadata"]["image/png"]["dark"]
    assert isinstance(dark, str)
    assert dark != images[0]["data"]["image/png"]


def test_execute_raises_when_kernel_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken setup cell must not silently produce light-only pages."""
    monkeypatch.setattr(execute, "_SETUP_SOURCE", "raise RuntimeError('no theme')")
    src = tmp_path / "ex.py"
    src.write_text("# %%\nprint('hi')\n")

    with pytest.raises(RuntimeError, match="no theme"):
        execute_example(src, timeout=60)


def test_execute_binds_kernel_ports_below_the_ephemeral_range(tmp_path: Path) -> None:
    """Kernel ports stay out of the range the OS hands to outgoing connections.

    Ports drawn from that range can be taken by a client connection between
    jupyter_client choosing them and the kernel binding them, which surfaced in CI
    as a heartbeat socket failing with "Address already in use".
    """
    src = tmp_path / "ex.py"
    src.write_text(
        "# %%\n"
        "from ipykernel.kernelapp import IPKernelApp\n"
        "app = IPKernelApp.instance()\n"
        "print(app.shell_port, app.iopub_port, app.stdin_port, "
        "app.control_port, app.hb_port)\n"
    )

    nb, _ = execute_example(src, timeout=60)

    stdout = "".join(
        out.get("text", "")
        for out in nb.cells[0].outputs
        if out.get("output_type") == "stream" and out.get("name") == "stdout"
    )
    ports = [int(token) for token in stdout.split()]
    assert len(ports) == 5
    assert len(set(ports)) == 5
    assert all(10000 <= port < 32768 for port in ports)


def test_find_kernel_ports_skips_ports_in_use(monkeypatch: pytest.MonkeyPatch) -> None:
    taken = find_kernel_ports(1)[0]
    held = socket.socket()
    held.bind(("127.0.0.1", taken))
    # Start the scan on the held port so the skip is exercised deterministically.
    monkeypatch.setattr(execute.random, "randrange", lambda *_: taken)
    try:
        ports = find_kernel_ports()
    finally:
        held.close()

    assert taken not in ports
    assert len(set(ports)) == 5
    assert all(taken < port < 32768 for port in ports)
