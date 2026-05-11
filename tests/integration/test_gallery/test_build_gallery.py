"""End-to-end test of the gallery builder."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.gallery._pipeline import build_gallery


def _seed_example(root: Path, section: str, name: str, body: str) -> Path:
    sec = root / section
    sec.mkdir(parents=True, exist_ok=True)
    (sec / "_section.md").write_text(f"# {section.title()}\n\nIntro.\n")
    src = sec / f"{name}.py"
    src.write_text(body)
    return src


@pytest.mark.slow
def test_build_gallery_produces_expected_artifacts(tmp_path: Path) -> None:
    examples_root = tmp_path / "docs" / "examples"
    built_dir = examples_root / "_built"
    cache_root = tmp_path / ".cache" / "gallery"
    examples_root.mkdir(parents=True)
    (examples_root / "_assets").mkdir()
    (examples_root / "_assets" / "default_thumb.svg").write_text("<svg/>")
    (examples_root / "_assets" / "default_thumb_dark.svg").write_text("<svg/>")

    _seed_example(
        examples_root,
        "io",
        "hello",
        "# %% [markdown]\n# # Hello\n#\n# Tiny example.\n\n# %%\nprint('hi')\n",
    )

    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="testdeps==1.0",
    )

    md = (built_dir / "io" / "hello.md").read_text()
    assert "# Hello" in md
    assert "```python\nprint('hi')\n```" in md
    assert (built_dir / "io" / "hello.py").is_file()
    assert (built_dir / "io" / "hello.ipynb").is_file()
    assert (examples_root / "index.md").read_text().count("Hello") >= 1


@pytest.mark.slow
def test_build_gallery_uses_cache_on_second_run(tmp_path: Path) -> None:
    examples_root = tmp_path / "docs" / "examples"
    built_dir = examples_root / "_built"
    cache_root = tmp_path / ".cache" / "gallery"
    examples_root.mkdir(parents=True)
    (examples_root / "_assets").mkdir()
    (examples_root / "_assets" / "default_thumb.svg").write_text("<svg/>")
    (examples_root / "_assets" / "default_thumb_dark.svg").write_text("<svg/>")
    _seed_example(
        examples_root,
        "io",
        "h",
        (
            "# %%\nimport time, pathlib\n"
            f"pathlib.Path({str((tmp_path / 'marker.txt').as_posix())!r}).write_text(str(time.time()))\n"
        ),
    )

    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="d",
    )
    first = (built_dir / "io" / "h.md").read_text()

    # Wipe the built dir but keep the cache. A second run must restore it
    # without re-executing.
    shutil.rmtree(built_dir)

    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="d",
    )
    second = (built_dir / "io" / "h.md").read_text()

    assert first == second  # Exact same artifact, including timestamp content.
