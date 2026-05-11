"""Render executed notebooks to Markdown + themed image folders."""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import nbformat


def _cell_tags(cell: nbformat.NotebookNode) -> set[str]:
    """Return the tags attached to one notebook cell."""
    return set(cell.metadata.get("tags", []))


def _png_data(output: dict[str, object]) -> str | None:
    """Return the base64 PNG payload from an output if present."""
    data = output.get("data")
    if not isinstance(data, dict) or "image/png" not in data:
        return None
    png = data["image/png"]
    if isinstance(png, list):
        return "".join(png)
    if isinstance(png, str):
        return png
    return None


def _write_png(path: Path, payload: str) -> None:
    """Decode and write one base64 PNG payload."""
    path.write_bytes(base64.b64decode(payload))


def _image_tag(*, src: str, alt: str) -> str:
    """Return a plain HTML image tag for Markdown output."""
    return f'<img class="skip-lightbox" src="{src}" alt="{alt}">'


def _html_block(html: str) -> str:
    """Return a raw HTML block for Markdown output."""
    return '<div class="gallery-rich-output">' + html.rstrip() + "</div>\n"


def _normalize_html_output(html: str) -> str:
    """Normalize rich HTML reprs for static docs rendering.

    In particular, xarray's notebook HTML repr includes a plain-text fallback and uses
    theme selectors that do not match Zensical's `data-md-color-scheme` attribute.
    """
    if "xr-wrap" in html:
        html = re.sub(
            r"<pre class=['\"]xr-text-repr-fallback['\"]>.*?</pre>",
            "",
            html,
            flags=re.S,
        )
        html = html.replace(
            "<div class='xr-wrap' style='display:none'>", "<div class='xr-wrap'>"
        )
        html = html.replace(
            '<div class="xr-wrap" style="display:none">', '<div class="xr-wrap">'
        )
        html += (
            "\n<style>"
            ".xr-array-preview,.xr-array-preview span,.xr-preview,.xr-var-preview,.xr-var-dtype,.xr-var-dims,.xr-var-name,.xr-obj-type,.xr-obj-name{color:var(--xr-font-color0)!important;}"
            ".gallery-rich-output{overflow-x:auto;}"
            ".gallery-rich-output .xr-var-list,.gallery-rich-output .xr-dim-list,.gallery-rich-output .xr-attrs{padding-left:0!important;margin:0!important;list-style:none!important;}"
            # MkDocs Material's `.md-typeset ul:not([hidden])` has higher specificity
            # than xarray's `.xr-sections` and forces `display:flow-root`, collapsing
            # the grid. Force it back so each coordinate row stays on one line.
            ".gallery-rich-output .xr-sections{display:grid!important;}"
            # Same Material rule also clobbers `.xr-var-list` (another <ul>).
            ".gallery-rich-output .xr-var-list,.gallery-rich-output .xr-var-item{display:contents!important;}"
            ".gallery-rich-output .xr-var-name,.gallery-rich-output .xr-var-dims,.gallery-rich-output .xr-var-dtype,.gallery-rich-output .xr-var-preview{margin:0!important;}"
            # Dask's chunk SVG labels can sit near element boundaries; forcing
            # overflow visible prevents edge labels (e.g., diagonal chunk labels)
            # from being clipped in docs layout containers.
            ".gallery-rich-output .xr-array-data table,.gallery-rich-output .xr-array-data td,.gallery-rich-output .xr-array-data svg{overflow:visible!important;}"
            ".gallery-rich-output .xr-array-data svg text,.gallery-rich-output .xr-array-data svg tspan,.gallery-rich-output .xr-array-preview svg text,.gallery-rich-output .xr-array-preview svg tspan,.gallery-rich-output .xr-preview svg text,.gallery-rich-output .xr-preview svg tspan{fill:var(--xr-font-color0)!important;}"
            ".gallery-rich-output .xr-array-data svg{color:var(--xr-font-color0)!important;}"
            # Dask injects inline `stroke: rgb(0,0,0)` on the root SVG; in dark
            # mode we override stroke/text color explicitly to keep labels legible.
            "[data-md-color-scheme='slate'] .gallery-rich-output .xr-array-data svg{stroke:rgba(255,255,255,0.5)!important;}"
            "[data-md-color-scheme='slate'] .gallery-rich-output .xr-array-data svg text,[data-md-color-scheme='slate'] .gallery-rich-output .xr-array-data svg tspan{fill:rgba(255,255,255,0.92)!important;stroke:none!important;}"
            "[data-md-color-scheme='default'] .xr-wrap{"
            "--xr-font-color0: rgba(0,0,0,1);"
            "--xr-font-color2: rgba(0,0,0,0.62);"
            "--xr-font-color3: rgba(0,0,0,0.42);"
            "--xr-border-color: var(--md-default-bg-color--lightest);"
            "--xr-disabled-color: rgba(0,0,0,0.35);"
            "--xr-background-color: #fff;"
            "--xr-background-color-row-even: #f8f9fb;"
            "--xr-background-color-row-odd: #edf0f5;"
            "}"
            "[data-md-color-scheme='slate'] .xr-wrap{"
            "--xr-font-color0: rgba(255,255,255,0.95);"
            "--xr-font-color2: rgba(255,255,255,0.68);"
            "--xr-font-color3: rgba(255,255,255,0.45);"
            "--xr-border-color: #2a3347;"
            "--xr-disabled-color: rgba(255,255,255,0.28);"
            "--xr-background-color: #111720;"
            "--xr-background-color-row-even: #161d29;"
            "--xr-background-color-row-odd: #1d2533;"
            "}"
            "</style>\n"
        )
    return html


def _clean_stream_text(text: str) -> str:
    """Remove known noisy notebook warnings from stream output."""
    cleaned_lines: list[str] = []
    skip_next = False
    for line in text.splitlines():
        if "TqdmWarning: IProgress not found" in line:
            skip_next = True
            continue
        if skip_next and "from .autonotebook import tqdm as notebook_tqdm" in line:
            skip_next = False
            continue
        skip_next = False
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip("\n")


def render_notebook(
    source_notebook: nbformat.NotebookNode,
    light_notebook: nbformat.NotebookNode,
    dark_notebook: nbformat.NotebookNode,
    *,
    out_dir: Path,
    base_name: str,
    runtime_seconds: float,
) -> tuple[Path, tuple[Path, Path] | None]:
    """Render an executed notebook pair to Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    light_image_dir = out_dir / f"{base_name}_output_light"
    dark_image_dir = out_dir / f"{base_name}_output_dark"
    light_image_dir.mkdir(exist_ok=True)
    dark_image_dir.mkdir(exist_ok=True)

    parts: list[str] = []
    thumbnail: tuple[Path, Path] | None = None
    light_cells = [
        cell
        for cell in light_notebook.cells
        if "_gallery_internal" not in _cell_tags(cell)
    ]
    dark_cells = [
        cell
        for cell in dark_notebook.cells
        if "_gallery_internal" not in _cell_tags(cell)
    ]

    for cell_index, (cell, light_cell, dark_cell) in enumerate(
        zip(source_notebook.cells, light_cells, dark_cells, strict=True),
        start=1,
    ):
        if cell.cell_type == "markdown":
            parts.append(cell.source.rstrip() + "\n")
            continue
        if cell.cell_type != "code":
            continue

        parts.append("```python\n" + cell.source.rstrip() + "\n```\n")

        for output_index, (light_output, dark_output) in enumerate(
            zip(
                light_cell.get("outputs", []), dark_cell.get("outputs", []), strict=True
            )
        ):
            output_type = light_output.get("output_type")
            if output_type == "stream":
                stream_text = _clean_stream_text(str(light_output.get("text", "")))
                if stream_text:
                    parts.append("```\n" + stream_text + "\n```\n")
                continue

            if output_type in {"execute_result", "display_data"}:
                light_png = _png_data(light_output)
                dark_png = _png_data(dark_output)
                if light_png is not None and dark_png is not None:
                    light_name = f"cell_{cell_index:02d}_{output_index}_light.png"
                    dark_name = f"cell_{cell_index:02d}_{output_index}_dark.png"
                    light_path = light_image_dir / light_name
                    dark_path = dark_image_dir / dark_name
                    _write_png(light_path, light_png)
                    _write_png(dark_path, dark_png)

                    alt = f"Example output from cell {cell_index}, image {output_index}"
                    parts.append(
                        _image_tag(
                            src=f"{base_name}_output_light/{light_name}#only-light",
                            alt=alt,
                        )
                        + _image_tag(
                            src=f"{base_name}_output_dark/{dark_name}#only-dark",
                            alt=alt,
                        )
                        + "\n"
                    )

                    if "thumbnail" in _cell_tags(cell) and thumbnail is None:
                        thumb_light = out_dir / f"{base_name}_thumb_light.png"
                        thumb_dark = out_dir / f"{base_name}_thumb_dark.png"
                        shutil.copyfile(light_path, thumb_light)
                        shutil.copyfile(dark_path, thumb_dark)
                        thumbnail = (thumb_light, thumb_dark)
                    continue

                light_data = light_output.get("data", {})
                if isinstance(light_data, dict) and "text/html" in light_data:
                    html = light_data["text/html"]
                    if isinstance(html, list):
                        html = "".join(html)
                    parts.append(_html_block(_normalize_html_output(str(html))))
                    continue
                if isinstance(light_data, dict) and "text/plain" in light_data:
                    parts.append(
                        "```\n" + str(light_data["text/plain"]).rstrip("\n") + "\n```\n"
                    )
                continue

            if output_type == "error":
                traceback = "\n".join(light_output.get("traceback", []))
                parts.append("```\n" + traceback + "\n```\n")

    parts.append("\n---\n")
    parts.append(f"**Total running time:** {runtime_seconds:.1f} s\n\n")
    parts.append(
        f"[Download .py]({base_name}.py){{ .md-button }} "
        f"[Download .ipynb]({base_name}.ipynb){{ .md-button }}\n"
    )

    markdown_path = out_dir / f"{base_name}.md"
    markdown_path.write_text("\n".join(parts), encoding="utf-8")
    return markdown_path, thumbnail
