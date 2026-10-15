# Authoring Examples

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
  Pure black text, lines and markers count as theme colors and turn light in dark
  mode; any other explicit color is kept as is.
- Do not branch on the theme in the example: there is only one run. Leave `cmap`
  unset on `plot_stat_map` and `plot_matrix` so the library's background-aware
  default applies, or pick a colormap that reads well on both backgrounds.
- Images not drawn by matplotlib (PIL images, screenshots) are shown unchanged in
  both themes.

When the built-in swap is not enough for the whole gallery (say, a sequential
colormap that should flip direction on dark backgrounds), edit `gallery_retheme`
in `tools/gallery/hooks.py`. It runs in the kernel after the swap, with `"dark"`
before the dark rendering and `"light"` after the figure is swapped back, and must
undo for `"light"` what it did for `"dark"`. It is a no-op by default. Editing it
invalidates the whole gallery cache, like any change to `tools/gallery`.

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
