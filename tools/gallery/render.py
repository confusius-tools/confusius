"""Render an executed notebook to Markdown + image folder."""

from __future__ import annotations

import base64
import shutil
from pathlib import Path

import nbformat


def render_notebook(
    nb: nbformat.NotebookNode,
    *,
    out_dir: Path,
    base_name: str,
    runtime_seconds: float,
) -> tuple[Path, Path | None]:
    """Render an executed notebook to Markdown.

    Writes ``<out_dir>/<base_name>.md`` along with a sibling
    ``<base_name>_output/`` directory containing one PNG per ``image/png``
    output. Returns the path to the Markdown file and to the thumbnail (the
    last PNG produced, copied to ``<base_name>_thumb.png``), or ``None`` if
    the notebook produced no images.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        The already-executed notebook.
    out_dir : pathlib.Path
        Directory where the Markdown and image folder will be written. Must
        exist or be creatable.
    base_name : str
        File stem used for the ``.md``, the output folder, and the thumbnail.
    runtime_seconds : float
        Wall-clock time the notebook took to execute, used in the footer.

    Returns
    -------
    md_path : pathlib.Path
        Path to the rendered Markdown file.
    thumbnail : pathlib.Path or None
        Path to the thumbnail PNG, or ``None`` if no image outputs were found.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / f"{base_name}_output"
    img_dir.mkdir(exist_ok=True)

    parts: list[str] = []
    last_image: Path | None = None

    for cell_idx, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "markdown":
            parts.append(cell.source.rstrip() + "\n")
            continue
        if cell.cell_type != "code":
            continue

        parts.append("```python\n" + cell.source.rstrip() + "\n```\n")

        for out_idx, output in enumerate(cell.get("outputs", [])):
            otype = output.get("output_type")
            if otype == "stream":
                parts.append("```\n" + output.get("text", "").rstrip("\n") + "\n```\n")
            elif otype in {"execute_result", "display_data"}:
                data = output.get("data", {})
                if "image/png" in data:
                    fname = f"cell_{cell_idx:02d}_{out_idx}.png"
                    fpath = img_dir / fname
                    fpath.write_bytes(base64.b64decode(data["image/png"]))
                    parts.append(f"![]({base_name}_output/{fname})\n")
                    last_image = fpath
                elif "text/plain" in data:
                    parts.append("```\n" + data["text/plain"].rstrip("\n") + "\n```\n")
            elif otype == "error":
                # Should not happen on success — execution would have raised.
                parts.append(
                    "```\n" + "\n".join(output.get("traceback", [])) + "\n```\n"
                )

    parts.append("\n---\n")
    parts.append(f"**Total running time:** {runtime_seconds:.1f} s\n\n")
    parts.append(
        f"[Download .py]({base_name}.py){{ .md-button }} "
        f"[Download .ipynb]({base_name}.ipynb){{ .md-button }}\n"
    )

    md_path = out_dir / f"{base_name}.md"
    md_path.write_text("\n".join(parts))

    thumbnail: Path | None = None
    if last_image is not None:
        thumbnail = out_dir / f"{base_name}_thumb.png"
        shutil.copyfile(last_image, thumbnail)

    return md_path, thumbnail
