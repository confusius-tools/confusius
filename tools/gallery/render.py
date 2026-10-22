"""Render executed notebooks to Markdown + themed image folders."""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import nbformat


def _join_str_list(value: object) -> str | None:
    """Join a list of strings, returning `None` for any other input."""
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        parts.append(item)
    return "".join(parts)


def _cell_tags(cell: nbformat.NotebookNode) -> set[str]:
    """Return the tags attached to one notebook cell."""
    return set(cell.metadata.get("tags", []))


_COLLAPSE_TAG = re.compile(r"collapse(?:\[(?P<kind>[^\]]+)\])?(?::(?P<title>.*))?\Z")


def _collapse_spec(tags: set[str]) -> tuple[str, str] | None:
    """Return the (admonition type, title) for a collapsed code cell, or `None`.

    A cell tagged `collapse` has its code hidden behind a collapsed admonition —
    useful for tucking away setup/boilerplate. The tag accepts an optional admonition
    type in brackets and an optional title after a colon: `collapse`,
    `collapse: <title>`, `collapse[<type>]`, or `collapse[<type>]: <title>`. The type
    is any admonition name Zensical understands (`example`, `warning`, `tip`, ...).

    Parameters
    ----------
    tags : set[str]
        The cell's tags.

    Returns
    -------
    kind : str
        The admonition type, defaulting to `example`.
    title : str
        The callout title, defaulting to `Show code`.
    """
    # Sorted so that a cell mistakenly carrying several collapse tags picks the same
    # one on every build instead of following set iteration order.
    for tag in sorted(tags):
        match = _COLLAPSE_TAG.match(tag)
        if match:
            kind = (match["kind"] or "").strip() or "example"
            title = (match["title"] or "").strip() or "Show code"
            return kind, title
    return None


def _collapsible_code(kind: str, title: str, code_block: str) -> str:
    """Wrap a fenced code block in a collapsed `??? <kind>` admonition.

    Parameters
    ----------
    kind : str
        The admonition type (e.g. `example`, `warning`).
    title : str
        The admonition title shown on the collapsed callout.
    code_block : str
        The fenced code block Markdown to hide.

    Returns
    -------
    str
        The Markdown for the collapsed callout, with `code_block` indented as its content.
    """
    indented = "\n".join(
        ("    " + line) if line.strip() else "" for line in code_block.splitlines()
    )
    return f'??? {kind} "{title}"\n\n{indented}\n'


def _png_data(output: dict[str, object]) -> str | None:
    """Return the base64 PNG payload from an output if present."""
    data = output.get("data")
    if not isinstance(data, dict):
        return None
    png = data.get("image/png")
    png_joined = _join_str_list(png)
    if png_joined is not None:
        return png_joined
    if isinstance(png, str):
        return png
    return None


def _dark_png_data(output: dict[str, object]) -> str | None:
    """Return the dark-theme PNG payload stored by `tools.gallery.theme`, if any."""
    metadata = output.get("metadata")
    if not isinstance(metadata, dict):
        return None
    png_metadata = metadata.get("image/png")
    if not isinstance(png_metadata, dict):
        return None
    dark = png_metadata.get("dark")
    return dark if isinstance(dark, str) else None


def _write_png(path: Path, payload: str) -> None:
    """Decode and write one base64 PNG payload."""
    path.write_bytes(base64.b64decode(payload))


def _image_tag(*, src: str, alt: str) -> str:
    """Return a plain HTML image tag for Markdown output."""
    return f'<img src="{src}" alt="{alt}">'


def _html_block(html: str) -> str:
    """Return a raw HTML block for Markdown output."""
    return '<div class="gallery-rich-output">' + html.rstrip() + "</div>\n"


def _output_code_block(text: str) -> str:
    """Return a fenced code block for plain-text cell output.

    The block is wrapped in a `gallery-output` container so docs CSS can wrap long
    output lines (warnings, reprs, tracebacks) instead of showing a horizontal
    scrollbar, without affecting the syntax-highlighted input code cells. The
    `markdown` attribute lets Zensical still render the fence as a highlighted block.

    Parameters
    ----------
    text : str
        The plain-text output to display.

    Returns
    -------
    str
        The Markdown for the wrapped, fenced output block.
    """
    return (
        '<div class="gallery-output" markdown>\n\n```\n'
        + text.rstrip("\n")
        + "\n```\n\n</div>\n"
    )


def _normalize_html_output(html: str) -> str:
    """Normalize rich HTML reprs for static docs rendering.

    In particular, xarray's notebook HTML repr includes a plain-text fallback and uses
    theme selectors that do not match Zensical's `data-md-color-scheme` attribute.
    """
    if "font-family:Menlo" in html and any(corner in html for corner in "┏┌╔╭"):
        html = html.replace("<pre ", '<pre class="gallery-rich-table" ', 1)
    if "xr-wrap" in html:
        html = re.sub(
            r"<pre class=['\"]xr-text-repr-fallback['\"]>.*?</pre>",
            "",
            html,
            flags=re.DOTALL,
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
    `display_data` output carrying both `text/html` and `text/plain` even when
    there is nothing to show (confirmed in CI: an empty `<pre style="...">` tag
    with a blank `text/plain`). `text/plain` is the one part of Jupyter's
    mimetype-bundle convention every real result populates (matplotlib figures
    carry `image/png` instead), so a blank one reliably means "nothing a reader
    would want to see". Real cell output (a figure, a DataArray repr, printed
    text) never has a blank `text/plain`.
    """
    if output.get("output_type") != "display_data":
        return False
    data = output.get("data")
    if not isinstance(data, dict) or not data:
        return False
    if any(isinstance(key, str) and key.startswith("image/") for key in data):
        return False
    if not set(data).issubset({"text/html", "text/plain"}):
        return False

    plain = data.get("text/plain", "")
    text = _join_str_list(plain)
    if text is None:
        text = str(plain)
    return not text.strip()


def render_notebook(
    source_notebook: nbformat.NotebookNode,
    executed_notebook: nbformat.NotebookNode,
    *,
    out_dir: Path,
    base_name: str,
    runtime_seconds: float,
    binder_url: str | None = None,
) -> tuple[Path, tuple[Path, Path] | None]:
    """Render an executed notebook to Markdown plus light and dark image folders.

    Parameters
    ----------
    source_notebook : nbformat.NotebookNode
        The example as read from disk; provides code, Markdown and cell tags.
    executed_notebook : nbformat.NotebookNode
        The same cells after execution. Figure outputs carry their dark rendering
        under `metadata["image/png"]["dark"]` (see `tools.gallery.theme`); other
        PNGs are shown unchanged in both themes.
    out_dir : pathlib.Path
        Directory to write the page and image folders into.
    base_name : str
        Output stem for the page and its asset folders.
    runtime_seconds : float
        Execution time printed at the bottom of the page.
    binder_url : str, optional
        Binder launch URL for the page's button. If not provided, no button.

    Returns
    -------
    markdown_path : pathlib.Path
        The rendered page.
    thumbnail : tuple[pathlib.Path, pathlib.Path] or None
        Light and dark thumbnail paths, if a `thumbnail`-tagged cell produced an image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    light_image_dir = out_dir / f"{base_name}_output_light"
    dark_image_dir = out_dir / f"{base_name}_output_dark"
    light_image_dir.mkdir(exist_ok=True)
    dark_image_dir.mkdir(exist_ok=True)

    parts: list[str] = []
    thumbnail: tuple[Path, Path] | None = None

    for cell_index, (cell, executed_cell) in enumerate(
        zip(source_notebook.cells, executed_notebook.cells, strict=True), start=1
    ):
        if cell.cell_type == "markdown":
            parts.append(cell.source.rstrip() + "\n")
            continue
        if cell.cell_type != "code":
            continue

        code_block = "```python\n" + cell.source.rstrip() + "\n```\n"
        collapse = _collapse_spec(_cell_tags(cell))
        if collapse is not None:
            parts.append(_collapsible_code(*collapse, code_block))
        else:
            parts.append(code_block)

        outputs = [
            o
            for o in executed_cell.get("outputs", [])
            if not _is_blank_widget_output(o)
        ]
        for output_index, output in enumerate(outputs):
            output_type = output.get("output_type")
            if output_type == "stream":
                stream_text = _clean_stream_text(str(output.get("text", "")))
                if stream_text:
                    parts.append(_output_code_block(stream_text))
                continue

            if output_type in {"execute_result", "display_data"}:
                light_png = _png_data(output)
                if light_png is not None:
                    # Images that did not come through the matplotlib hook have no
                    # dark rendering: show the same picture in both themes.
                    dark_png = _dark_png_data(output) or light_png
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

                data = output.get("data", {})
                if isinstance(data, dict) and "text/html" in data:
                    html = data["text/html"]
                    if isinstance(html, list):
                        html = "".join(html)
                    parts.append(_html_block(_normalize_html_output(str(html))))
                    continue
                if isinstance(data, dict) and "text/plain" in data:
                    parts.append(_output_code_block(str(data["text/plain"])))
                continue

            if output_type == "error":
                traceback = "\n".join(output.get("traceback", []))
                parts.append(_output_code_block(traceback))

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
