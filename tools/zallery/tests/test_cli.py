"""Tests for the `zallery` command line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _make_project(tmp_path: Path) -> tuple[Path, Path]:
    """Return a mini-project with a `[tool.zallery]` table and one example."""
    project = tmp_path / "project"
    examples = project / "docs" / "examples"
    (examples / "_assets").mkdir(parents=True)
    (examples / "_assets" / "default_thumb.png").write_bytes(b"\x89PNG\r\n")
    (examples / "io").mkdir()
    (examples / "io" / "_section.md").write_text("## Input/Output\n\nIO.\n")
    (examples / "io" / "hello.py").write_text(
        "# %% [markdown]\n# # Hello ┏\n\n# %%\nprint('hi')\n", encoding="utf-8"
    )
    (project / "pyproject.toml").write_text(
        '[tool.zallery]\nexamples = "docs/examples"\n'
    )
    (project / "uv.lock").write_text("# placeholder\n")
    return project, examples


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "zallery", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.slow
def test_cli_builds_the_gallery_of_the_enclosing_project(tmp_path: Path) -> None:
    """Run from a subdirectory, the CLI finds the table upwards and builds."""
    project, examples = _make_project(tmp_path)

    proc = _run(examples / "io")

    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    markdown = (examples / "_built" / "io" / "hello.md").read_text(encoding="utf-8")
    assert "# Hello ┏" in markdown
    assert '<div class="gallery-output" markdown>' in markdown
    assert (examples / "index.md").is_file()
    assert (project / ".gallery-cache").is_dir()


def test_cli_no_run_renders_every_example_without_executing(tmp_path: Path) -> None:
    """`--no-run` renders code and prose only, and caches nothing."""
    project, examples = _make_project(tmp_path)

    proc = _run(project, "--no-run")

    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    markdown = (examples / "_built" / "io" / "hello.md").read_text(encoding="utf-8")
    assert "print('hi')" in markdown
    assert "gallery-output" not in markdown
    assert "[not run]" in proc.stdout
    assert not list((project / ".gallery-cache").glob("**/hello"))


def test_cli_no_run_rejects_example_paths(tmp_path: Path) -> None:
    project, examples = _make_project(tmp_path)

    proc = _run(project, "--no-run", str(examples / "io" / "hello.py"))

    assert proc.returncode == 2
    assert "--no-run cannot be combined" in proc.stderr


def test_cli_returns_1_when_examples_dir_missing(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.zallery]\n")

    proc = _run(project)

    assert proc.returncode == 1
    assert "No examples directory" in proc.stderr


def test_cli_returns_2_without_configuration(tmp_path: Path) -> None:
    proc = _run(tmp_path)

    assert proc.returncode == 2
    assert "[tool.zallery]" in proc.stderr
