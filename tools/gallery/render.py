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
    return f'<img src="{src}" alt="{alt}">'


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
            # Zensical's `.md-typeset ul li` rule adds margin-left:1.25em to <li>
            # elements, including those inside .xr-dim-list.
            ".gallery-rich-output .xr-dim-list li{margin-left:0!important;}"
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


def _is_blank_widget_output(output: dict[str, object]) -> bool:
    """Return whether `output` is a content-free rich/Jupyter display artifact.

    Some dependency prints via `rich`'s Jupyter console protocol, which renders a
    `display_data` output carrying both `text/html` and `text/plain` regardless of
    whether there's anything meaningful to show (confirmed in CI: an empty
    `<pre style="...">` tag with a blank `text/plain`). Because *when* this fires
    isn't tied to a specific cell in a deterministic way, it can show up in only
    one of the light/dark passes.

    `text/plain` is the one part of Jupyter's mimetype-bundle convention every
    real result populates (it's the accessibility/fallback representation, and
    matplotlib figures instead carry `image/png`), so a blank one reliably means
    "nothing a reader would want to see" regardless of what the accompanying
    `text/html` contains. Safe to drop: real cell output (a matplotlib figure, a
    DataArray repr, printed text) never has a blank `text/plain`.
    """
    if output.get("output_type") != "display_data":
        return False
    data = output.get("data")
    if not isinstance(data, dict) or not data:
        return False
    if any(key.startswith("image/") for key in data):
        return False
    if not set(data).issubset({"text/html", "text/plain"}):
        return False

    plain = data.get("text/plain", "")
    text = "".join(plain) if isinstance(plain, list) else str(plain)
    return not text.strip()


def _summarize_output(output: dict[str, object]) -> str:
    """Return a one-line human-readable summary of one notebook output.

    Used to build a useful error message when the light and dark runs disagree on
    the number of outputs for a cell: showing what each side actually produced makes
    the offending extra (a stray warning, a one-shot download log, etc.) obvious.

    Parameters
    ----------
    output : dict[str, object]
        One nbformat output node (a dict with an `output_type` key).

    Returns
    -------
    str
        A short, single-line description of the output's type and content.
    """
    output_type = output.get("output_type")
    if output_type == "stream":
        name = output.get("name", "stdout")
        text = str(output.get("text", "")).replace("\n", "\\n")
        return f"stream[{name}]: {text[:200]}"
    if output_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        keys = sorted(data) if isinstance(data, dict) else []
        summary = f"{output_type}: {{{', '.join(keys)}}}"
        if isinstance(data, dict):
            for key in keys:
                value = data[key]
                text = "".join(value) if isinstance(value, list) else str(value)
                text = text.replace("\n", "\\n")
                summary += f"\n    {key}: {text[:500]}"
        metadata = output.get("metadata")
        if isinstance(metadata, dict) and metadata:
            summary += f"\n    metadata: {metadata}"
        return summary
    if output_type == "error":
        return f"error: {output.get('ename')}: {output.get('evalue')}"
    return str(output_type)


def _summarize_outputs(outputs: list[dict[str, object]]) -> str:
    """Return a numbered, indented summary of a cell's outputs.

    Parameters
    ----------
    outputs : list[dict[str, object]]
        The outputs of one notebook cell.

    Returns
    -------
    str
        One `_summarize_output` line per output, numbered and indented, or `(none)`
        when the cell produced no outputs.
    """
    if not outputs:
        return "  (none)"
    return "\n".join(
        f"  {i}. {_summarize_output(output)}" for i, output in enumerate(outputs, 1)
    )


def render_notebook(
    source_notebook: nbformat.NotebookNode,
    light_notebook: nbformat.NotebookNode,
    dark_notebook: nbformat.NotebookNode,
    *,
    out_dir: Path,
    base_name: str,
    runtime_seconds: float,
    binder_url: str | None = None,
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

        # Light and dark outputs are paired by index. They must have the
        # same length: a mismatch means non-deterministic output snuck in
        # (typically a one-shot download or a warning that fired only once),
        # and silently dropping the extras would hide a real difference
        # between the two rendered notebooks. Pre-warm caches outside the
        # gallery so both runs start from the same state. Blank rich/Jupyter
        # display artifacts are the one known exception: they carry no
        # content, so dropping them can't hide a real difference (see
        # `_is_blank_widget_output`).
        light_outputs = [
            o for o in light_cell.get("outputs", []) if not _is_blank_widget_output(o)
        ]
        dark_outputs = [
            o for o in dark_cell.get("outputs", []) if not _is_blank_widget_output(o)
        ]
        if len(light_outputs) != len(dark_outputs):
            raise ValueError(
                f"{base_name}: cell {cell_index} produced "
                f"{len(light_outputs)} light outputs and {len(dark_outputs)} "
                "dark outputs. Light/dark executions must be deterministic; "
                "make sure any one-shot side effects (dataset downloads, "
                "first-import warnings, etc.) happen before the gallery "
                "build.\n"
                f"Light outputs ({len(light_outputs)}):\n"
                f"{_summarize_outputs(light_outputs)}\n"
                f"Dark outputs ({len(dark_outputs)}):\n"
                f"{_summarize_outputs(dark_outputs)}"
            )
        for output_index, (light_output, dark_output) in enumerate(
            zip(light_outputs, dark_outputs)
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
    buttons: list[str] = []
    if binder_url is not None:
        buttons.append(
            f"[Launch in Binder]({binder_url}){{ .md-button .md-button--primary }}"
        )
    buttons.append(f"[Download .py]({base_name}.py){{ .md-button }}")
    buttons.append(f"[Download .ipynb]({base_name}.ipynb){{ .md-button }}")
    parts.append(" ".join(buttons) + "\n")

    markdown_path = out_dir / f"{base_name}.md"
    markdown_path.write_text("\n".join(parts), encoding="utf-8")
    return markdown_path, thumbnail
