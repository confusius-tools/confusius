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
