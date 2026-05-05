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

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists.
2. Create a new issue with a clear description.
3. Include steps to reproduce (for bugs).
4. Include code examples if applicable.

### Code of Conduct

Be respectful and constructive in all interactions.
