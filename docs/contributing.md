---
hide:
    - navigation
---

# Community

ConfUSIus is developed in the open. This page gathers ways to contribute, report
issues, and connect with the people maintaining the project.

## Core Developers

<div class="grid cards" markdown>

-   [:fontawesome-brands-github: **Samuel Le Meur-Diebolt**](https://github.com/sdiebolt)

    ---

    Postdoctoral researcher,
    [Cortexlab](https://www.ucl.ac.uk/brain-sciences/cortexlab).

-   [:fontawesome-brands-github: **Felipe Cybis Pereira**](https://github.com/FelipeCybis)

    ---

    Independent researcher.

</div>

## Acknowledgments

We gratefully acknowledge Mark S. Blumberg, PhD, and Brett Dwyer, Research Associate, of
the University of Iowa Department of Psychological and Brain Sciences, for their
feedback on the ConfUSIus GUI plugin. We also thank them for providing an example SCAN
v2 file that helped guide the implementation of the experimental SCAN v2 loader.

## Contributing

We welcome contributions to ConfUSIus. This guide will help you get started.

### Getting Started

1. Open an issue to discuss your idea or bug fix.
2. Fork the repository on GitHub.
3. Clone your fork locally.
4. Set up the development environment.
5. Make your changes.
6. Submit a pull request.

### Development Installation

```bash
# Clone your fork
git clone https://github.com/confusius-tools/confusius.git
cd confusius

# Install with development dependencies
uv sync

# Run tests
just test

# Run pre-commit hooks
just pre-commit
```

### Code Style

- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Run pre-commits (e.g., using [prek](https://prek.j178.dev/)) and tests before
  committing.
- Add tests for new functionality.
- Update documentation as needed.

### Documentation

Documentation is built on GitHub Actions and deployed to a separate GitHub Pages
repository ([confusius-tools/confusius-docs](https://github.com/confusius-tools/confusius-docs)).
Every pull request gets an automatic preview, with a link posted as a comment on the PR.
The preview is cleaned up automatically when the PR is closed.

To build and serve the docs locally:

```bash
just sd
```

**Adding documentation images.** Image generators live in `docs/images/<topic>/generate.py`
(outputs are gitignored; only the script is committed). If you add a new one:

- Add it to the `just generate-doc-images` recipe in `justfile`.
- Register it in the **Generate documentation images** step of `.github/workflows/docs.yml`.
- If it uses an already-cached dataset, add the script path to that dataset's
  `hashFiles(...)` call in the workflow so the cache invalidates when the script changes.
  If it needs a new dataset, add a dedicated cache step.

**Adding examples.** Example scripts live in `docs/examples/` and are discovered
automatically by the gallery builder — no cache update needed (the cache key already
covers `docs/examples/**/*.py`). After adding a script, add its built output path to
the `nav` in `zensical.toml`.

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists.
2. Create a new issue with a clear description.
3. Include steps to reproduce (for bugs).
4. Include code examples if applicable.

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://github.com/confusius-tools/confusius/blob/main/CODE_OF_CONDUCT.md).
By participating, you are expected to uphold it.
