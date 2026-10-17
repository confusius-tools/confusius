# Authoring Examples

This directory is `zallery`, the gallery builder ConfUSIus uses, kept as a uv
workspace member so it can become its own distribution later. Its tests live in
`tests/` here and run with `just test`. The ConfUSIus side of the setup is
the `[tool.zallery]` table in `pyproject.toml` (paths, Binder, hook) and
`tools/gallery_hooks.py` (the theme hook described below).

Examples live under `docs/examples/<section>/<name>.py` in jupytext percent-format.
Section folders contain a `_section.md` whose first heading becomes the section title in
the generated index.

## Quick Start

```bash
mkdir -p docs/examples/my-section
printf '# My section\n\nIntro.\n' > docs/examples/my-section/_section.md
cat > docs/examples/my-section/hello.py <<'EOF'
# %% [markdown]
# # My example
#
# Short description.

# %%
print("hi")
EOF

just gallery
```

After running `just gallery`, browse `docs/examples/_built/` for the rendered output.
To iterate on prose or layout without executing anything, pass `--no-run` to the
builder (`uv run zallery --no-run`): every page is restored
from cache if present, or rendered with code and Markdown only. Name example
scripts instead to run just those. Add `--no-cache` to execute the selected
examples even when a cached result exists; the result replaces the cache entry.
The first markdown heading in each example becomes the page title; the next markdown
paragraph is used as the index-card summary.

## Light and dark figures

Every example runs once, under a light matplotlib style. Each figure is then
re-rendered for dark mode without re-executing anything: the gallery swaps the
theme colors (figure background, axes background, black foreground) and the
default diverging colormap (`coolwarm` to `berlin`) on the drawn artists, prints a
second PNG, and swaps everything back. Both PNGs land on the page, and the site
shows the one matching the reader's theme.

What this means when writing an example:

- Read the background from matplotlib rather than hardcoding it, and pass it on
  to the plotting functions: `bg_color = mpl.colors.to_hex(mpl.rcParams["figure.facecolor"])`.
  Pure black and pure white are theme colors wherever they appear (text, lines, line
  markers, and backgrounds); always pass `bg_color` to the plotting functions rather
  than relying on their `"black"` default. Per-element colors (`scatter(c="black")`,
  collections) are left alone, and any other explicit color is kept as is.
- Do not branch on the theme in the example: there is only one run. Leave `cmap`
  unset on `plot_stat_map` and `plot_matrix` so the library's background-aware
  default applies, or pick a colormap that reads well on both backgrounds.
- Images not drawn by matplotlib (PIL images, screenshots) are shown unchanged in
  both themes.

When the built-in swap is not enough for the whole gallery, edit `gallery_retheme`
in `tools/gallery_hooks.py`. It runs in the kernel after the swap, with `"dark"`
before the dark rendering and `"light"` after the figure is swapped back, and must
undo for `"light"` what it did for `"dark"`. As shipped it flips sequential
colormaps whose low end should fade into the page: draw with `inferno_r` on the
light page and the dark page gets `inferno`. Add pairs to `SEQUENTIAL_CMAP_SWAPS`
there to extend it. Editing the file invalidates the whole gallery cache, like any
change to the `zallery` sources or the hook.

## Configuration

`zallery` reads the `[tool.zallery]` table of the nearest `pyproject.toml` above the
working directory; relative paths resolve against that file's directory. All keys
are optional:

| Key | Default | Meaning |
|---|---|---|
| `examples` | `docs/examples` | Directory of example sections. |
| `built` | `<examples>/_built` | Where rendered pages and images go. |
| `cache` | `.gallery-cache` | Execution/render cache. |
| `binder_repo` | unset | `owner/repo` for the Binder launch buttons. |
| `hook` | unset | Module defining `gallery_retheme(fig, theme)`. |
| `fingerprint` | `["uv.lock"]` | Files whose change invalidates the cache. |

The git ref stamped into Binder URLs comes from the `ZALLERY_BINDER_REF`
environment variable (default `main`).

## Tips

- Files starting with `_` are skipped by the discovery pass.
- Tag a cell with `thumbnail` to use its first image output on the gallery index page.
- Tag a cell with `collapse` (or `collapse: <title>`) to hide its code behind a collapsed
  callout — handy for setup/boilerplate the reader can expand on demand. The cell still
  runs, and its output renders normally below the callout. The callout defaults to the
  `example` admonition; pick another type with `collapse[<type>]` or
  `collapse[<type>]: <title>`, where `<type>` is any admonition name Zensical understands
  (`warning`, `tip`, `note`, ...) (see https://zensical.org/docs/authoring/admonitions/).
- Keep examples self-contained: prefer small public dataset subsets over local files.
