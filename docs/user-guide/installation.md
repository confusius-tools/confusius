---
icon: lucide/download
---

# Installation

ConfUSIus is compatible with Python 3.13 and above. We recommend using
[uv](https://docs.astral.sh/uv/) for fast and efficient package management, but you can
also install ConfUSIus using the Python package manager of your choice.

## Install with uv (recommended)

```bash
# Create a new project.
uv init my-project
cd my-project

# Add confusius.
uv add confusius
```

## Development Installation

Please find all development installation instructions in the [Contributing
Guide](contributing.md#development-installation).

## Verify Installation

To verify that ConfUSIus has been installed correctly, you may run the following code
snippet in a Python environment:

```python
import confusius as cf

print(cf.__version__)
```

If the installation was successful, this code will print the installed version of
ConfUSIus without any errors. If you encounter an error, please open an issue on the
[GitHub repository](https://github.com/sdiebolt/confusius/issues/).
