"""High-level orchestrator: discover, cache, execute, render, index."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nbformat
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from . import cache, discover, execute, index, render
from ._types import ExampleSpec, RenderedExample


def _python_version() -> str:
    """Return the current Python version string."""
    return sys.version.split()[0]


def _extract_title_and_summary(
    nb: nbformat.NotebookNode, fallback_stem: str
) -> tuple[str, str]:
    """Pull the first H1 heading and the following paragraph from a parsed notebook."""
    title = fallback_stem
    summary_lines: list[str] = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            if title != fallback_stem:
                break
            continue
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            stripped = line.strip()
            if title == fallback_stem:
                if stripped.startswith("# "):
                    title = stripped[2:].strip()
            elif not stripped:
                if summary_lines:
                    return title, " ".join(summary_lines)
            elif not stripped.startswith("#"):
                summary_lines.append(stripped)
            else:
                return title, " ".join(summary_lines)
    return title, " ".join(summary_lines)


def _copy_dir_contents(src: Path, dst: Path) -> None:
    """Copy the contents of ``src`` into ``dst``, merging subdirectories."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, target)


def _write_cached_artifacts(
    cache_dir: Path, dest_dir: Path, base_name: str
) -> tuple[Path, Path] | None:
    """Copy cached files for one example into ``dest_dir``."""
    _copy_dir_contents(cache_dir, dest_dir)
    thumb_light = dest_dir / f"{base_name}_thumb_light.png"
    thumb_dark = dest_dir / f"{base_name}_thumb_dark.png"
    if thumb_light.is_file() and thumb_dark.is_file():
        return thumb_light, thumb_dark
    return None


def _make_progress() -> Progress:
    """Return a rich Progress configured for the gallery builder."""
    return Progress(
        TextColumn("  {task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} cells"),
        TimeElapsedColumn(),
    )


def _advance_on_cell(progress: Progress, task_id: TaskID):
    """Return an ``on_cell_executed`` hook that advances ``task_id`` by one.

    The injected theme-setup cell (always at ``cell_index == 0``) is skipped so the
    bar's totals reflect only the cells the user wrote.
    """

    def hook(**kwargs: object) -> None:
        if kwargs.get("cell_index") == 0:
            return
        progress.update(task_id, advance=1)

    return hook


def _build_one(
    spec: ExampleSpec,
    *,
    built_dir: Path,
    cache_root: Path,
    deps_fingerprint: str,
    progress: Progress,
    binder_url: str | None = None,
) -> RenderedExample:
    """Build one example and return its rendered metadata."""
    base_name = spec.base_name
    out_dir = built_dir / spec.section
    out_dir.mkdir(parents=True, exist_ok=True)

    key = cache.cache_key(
        spec,
        deps_fingerprint=deps_fingerprint,
        python_version=_python_version(),
    )
    cache_entry = cache.cache_dir(cache_root, key) / spec.section / base_name

    source_notebook = execute.read_example(spec.source)
    title, summary = _extract_title_and_summary(source_notebook, base_name)
    # nbclient only invokes `on_cell_executed` for code cells, so the bar's
    # total has to match that — counting markdown cells would leave it stuck.
    n_cells = sum(1 for c in source_notebook.cells if c.cell_type == "code")

    if cache_entry.is_dir():
        progress.add_task(f"{spec.section}/{base_name} [cached]", total=1, completed=1)
        thumb = _write_cached_artifacts(cache_entry, out_dir, base_name)
        return RenderedExample(
            spec=spec,
            title=title,
            summary=summary,
            md_path=out_dir / f"{base_name}.md",
            thumbnail_light=thumb[0] if thumb is not None else None,
            thumbnail_dark=thumb[1] if thumb is not None else None,
        )

    light_task = progress.add_task(f"{spec.section}/{base_name} (light)", total=n_cells)
    light_notebook, light_seconds = execute.execute_example(
        spec.source,
        theme="light",
        on_cell_executed=_advance_on_cell(progress, light_task),
    )
    dark_task = progress.add_task(f"{spec.section}/{base_name} (dark)", total=n_cells)
    dark_notebook, _ = execute.execute_example(
        spec.source,
        theme="dark",
        on_cell_executed=_advance_on_cell(progress, dark_task),
    )

    scratch = cache_root / "_scratch" / spec.section / base_name
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    render.render_notebook(
        source_notebook,
        light_notebook,
        dark_notebook,
        out_dir=scratch,
        base_name=base_name,
        runtime_seconds=light_seconds,
        binder_url=binder_url,
    )

    shutil.copyfile(spec.source, scratch / f"{base_name}.py")
    nbformat.write(source_notebook, scratch / f"{base_name}.ipynb")

    cache_entry.parent.mkdir(parents=True, exist_ok=True)
    if cache_entry.exists():
        shutil.rmtree(cache_entry)
    shutil.move(scratch, cache_entry)

    thumb = _write_cached_artifacts(cache_entry, out_dir, base_name)
    return RenderedExample(
        spec=spec,
        title=title,
        summary=summary,
        md_path=out_dir / f"{base_name}.md",
        thumbnail_light=thumb[0] if thumb is not None else None,
        thumbnail_dark=thumb[1] if thumb is not None else None,
    )


def _binder_url(
    source: Path, *, repo_root: Path, binder_repo: str, binder_ref: str
) -> str:
    """Return the mybinder.org URL that opens ``source`` as a notebook."""
    rel = source.relative_to(repo_root).as_posix()
    return (
        f"https://mybinder.org/v2/gh/{binder_repo}/{binder_ref}?urlpath=lab/tree/{rel}"
    )


def build_gallery(
    *,
    examples_root: Path,
    built_dir: Path,
    cache_root: Path,
    deps_fingerprint: str,
    repo_root: Path | None = None,
    binder_repo: str | None = None,
    binder_ref: str = "main",
) -> None:
    """Run the full pipeline and write the gallery to disk."""
    specs = discover.discover(examples_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    rendered: list[RenderedExample] = []
    with _make_progress() as progress:
        for spec in specs:
            binder_url = (
                _binder_url(
                    spec.source,
                    repo_root=repo_root,
                    binder_repo=binder_repo,
                    binder_ref=binder_ref,
                )
                if binder_repo is not None and repo_root is not None
                else None
            )
            rendered.append(
                _build_one(
                    spec,
                    built_dir=built_dir,
                    cache_root=cache_root,
                    deps_fingerprint=deps_fingerprint,
                    progress=progress,
                    binder_url=binder_url,
                )
            )

    index_markdown = index.build_index(rendered, root=examples_root)
    (examples_root / "index.md").write_text(index_markdown, encoding="utf-8")
