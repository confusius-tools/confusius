"""End-to-end test of the gallery builder."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.gallery._pipeline import build_gallery

GalleryPaths = tuple[Path, Path, Path]


def _seed_example(root: Path, section: str, name: str, body: str) -> Path:
    sec = root / section
    sec.mkdir(parents=True, exist_ok=True)
    (sec / "_section.md").write_text(f"# {section.title()}\n\nIntro.\n")
    src = sec / f"{name}.py"
    src.write_text(body)
    return src


@pytest.mark.slow
def test_build_gallery_produces_expected_artifacts(gallery_paths: GalleryPaths) -> None:
    examples_root, built_dir, cache_root = gallery_paths

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

    first = (built_dir / "io" / "hello.md").read_text()
    assert "# Hello" in first
    assert "```python\nprint('hi')\n```" in first
    assert (built_dir / "io" / "hello.py").is_file()
    assert (built_dir / "io" / "hello.ipynb").is_file()
    assert (examples_root / "index.md").read_text().count("Hello") >= 1

    # Wipe the built dir but keep the cache. A second run must restore it
    # without re-executing.
    shutil.rmtree(built_dir)

    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="testdeps==1.0",
    )
    second = (built_dir / "io" / "hello.md").read_text()

    assert first == second  # Exact same artifact, including timestamp content.
    assert (built_dir / "io" / "hello.py").is_file()
    assert (built_dir / "io" / "hello.ipynb").is_file()


@pytest.mark.slow
def test_build_gallery_embeds_binder_launch_url(gallery_paths: GalleryPaths) -> None:
    examples_root, built_dir, cache_root = gallery_paths
    repo_root = examples_root.parent.parent

    _seed_example(
        examples_root,
        "io",
        "hello",
        "# %% [markdown]\n# # Hello\n\n# %%\nprint('hi')\n",
    )

    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="testdeps==1.0",
        repo_root=repo_root,
        binder_repo="confusius-tools/confusius",
        binder_ref="v9.9.9",
    )

    md = (built_dir / "io" / "hello.md").read_text()
    expected = (
        "https://mybinder.org/v2/gh/confusius-tools/confusius/v9.9.9"
        "?urlpath=lab/tree/docs/examples/io/hello.py"
    )
    assert f"[Launch in Binder]({expected})" in md
