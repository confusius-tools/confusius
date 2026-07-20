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
def test_build_gallery_prints_progress_when_non_interactive(
    gallery_paths: GalleryPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    examples_root, built_dir, cache_root = gallery_paths

    _seed_example(
        examples_root,
        "io",
        "hello",
        "# %% [markdown]\n# # Hello\n\n# %%\nprint('hi')\n",
    )

    # Under pytest stdout is captured and reports ``isatty() == False``, so the
    # builder takes its non-interactive path and emits plain progress lines
    # instead of the rich live bar.
    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="testdeps==1.0",
    )
    fresh = capsys.readouterr().out
    assert "-> io/hello" in fresh
    assert "done in" in fresh

    # A second run hits the cache and announces it.
    shutil.rmtree(built_dir)
    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="testdeps==1.0",
    )
    assert "[cached]" in capsys.readouterr().out


@pytest.mark.slow
def test_build_gallery_runs_only_selected_and_stubs_the_rest(
    gallery_paths: GalleryPaths,
) -> None:
    examples_root, built_dir, cache_root = gallery_paths

    wanted = _seed_example(
        examples_root,
        "io",
        "hello",
        "# %% [markdown]\n# # Hello\n\n# %%\nprint('hi')\n",
    )
    _seed_example(
        examples_root,
        "misc",
        "skipme",
        "# %% [markdown]\n# # Skip\n\n# %%\nprint('bye')\n",
    )

    build_gallery(
        examples_root=examples_root,
        built_dir=built_dir,
        cache_root=cache_root,
        deps_fingerprint="testdeps==1.0",
        only=[wanted],
    )

    # The selected example is executed: its output is rendered.
    hello = (built_dir / "io" / "hello.md").read_text()
    assert '<div class="gallery-output" markdown>' in hello
    # The rest are rendered without running: the source is there, but no outputs.
    skip = (built_dir / "misc" / "skipme.md").read_text()
    assert "print('bye')" in skip
    assert "gallery-output" not in skip
    # It is not cached, so a later run would still execute it.
    assert not list(cache_root.glob("**/misc/skipme"))
    # The index still lists the full gallery.
    index_md = (examples_root / "index.md").read_text()
    assert "Hello" in index_md
    assert "Skip" in index_md


def test_build_gallery_raises_for_unknown_only_path(
    gallery_paths: GalleryPaths,
) -> None:
    examples_root, built_dir, cache_root = gallery_paths
    _seed_example(
        examples_root, "io", "hello", "# %% [markdown]\n# # Hi\n\n# %%\nx = 1\n"
    )

    with pytest.raises(ValueError, match="Not discoverable"):
        build_gallery(
            examples_root=examples_root,
            built_dir=built_dir,
            cache_root=cache_root,
            deps_fingerprint="testdeps==1.0",
            only=[examples_root / "io" / "ghost.py"],
        )


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
