"""Tests for the build_gallery CLI entry point."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "tools" / "build_gallery.py"


@pytest.mark.slow
def test_cli_runs_to_completion_when_examples_dir_exists(tmp_path: Path) -> None:
    """Running the CLI directly via `python tools/build_gallery.py` succeeds."""
    # Build a sandboxed mini-project so we don't pollute the repo's cache.
    sandbox = tmp_path / "sandbox"
    examples = sandbox / "docs" / "examples"
    (examples / "_assets").mkdir(parents=True)
    (examples / "_assets" / "default_thumb.png").write_bytes(b"\x89PNG\r\n")
    (examples / "io").mkdir()
    (examples / "io" / "_section.md").write_text("## Input/Output\n\nIO.\n")
    (examples / "io" / "hello.py").write_text(
        "# %% [markdown]\n# # Hello\n\n# %%\nprint('hi')\n"
    )

    # Copy the CLI + tools/gallery into the sandbox so REPO_ROOT inside the
    # script resolves to the sandbox.
    shutil.copytree(REPO_ROOT / "tools", sandbox / "tools")
    # uv.lock is needed for the deps fingerprint helper but its content is
    # irrelevant to this test; an empty one would also work.
    (sandbox / "uv.lock").write_text("# placeholder\n")

    proc = subprocess.run(
        [sys.executable, str(sandbox / "tools" / "build_gallery.py")],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert (examples / "_built" / "io" / "hello.md").is_file()
    assert (examples / "index.md").is_file()


def test_cli_returns_1_when_examples_dir_missing(tmp_path: Path) -> None:
    """If the docs/examples/ tree doesn't exist, the CLI exits 1."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    shutil.copytree(REPO_ROOT / "tools", sandbox / "tools")
    (sandbox / "uv.lock").write_text("# placeholder\n")

    proc = subprocess.run(
        [sys.executable, str(sandbox / "tools" / "build_gallery.py")],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "No examples directory" in proc.stderr
