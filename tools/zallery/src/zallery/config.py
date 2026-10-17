"""Project configuration, read from the `[tool.zallery]` table of `pyproject.toml`.

The table is looked up from the working directory upwards, so `zallery` can run
from anywhere inside the project. Relative paths in it are resolved against the
directory holding that `pyproject.toml`, which is also the project root the kernel
gets on its `sys.path` so the theme hook module imports.

Keys (all optional):

- `examples`: directory of example sections, default `docs/examples`.
- `built`: where rendered pages go, default `<examples>/_built`.
- `cache`: execution/render cache directory, default `.gallery-cache`.
- `binder_repo`: `owner/repo` slug for the Binder launch buttons; no buttons if unset.
- `hook`: dotted name of the module defining `gallery_retheme(fig, theme)`.
- `fingerprint`: files whose content invalidates the cache when it changes,
  default `["uv.lock"]`. The `zallery` sources, the hook module and the table
  itself are always part of the fingerprint.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import zallery

_TABLE = "zallery"
"""Name of the table under `[tool]`."""

_KNOWN_KEYS = frozenset(
    {"examples", "built", "cache", "binder_repo", "hook", "fingerprint"}
)
"""Keys the table may contain; anything else is a typo and rejected."""


class ConfigError(ValueError):
    """Raised when no usable `[tool.zallery]` table is found or it is malformed."""


@dataclass(frozen=True)
class Config:
    """Resolved project configuration.

    Parameters
    ----------
    root : pathlib.Path
        Directory holding the `pyproject.toml` the table was read from.
    examples : pathlib.Path
        Directory of example sections.
    built : pathlib.Path
        Directory rendered pages and assets are written to.
    cache : pathlib.Path
        Execution/render cache directory.
    binder_repo : str, optional
        `owner/repo` slug for Binder launch buttons. If not provided, no buttons.
    hook : str, optional
        Dotted name of the project's theme hook module. If not provided, only the
        built-in swap runs.
    fingerprint_files : tuple[pathlib.Path, ...]
        Files whose content is part of the cache fingerprint.
    """

    root: Path
    examples: Path
    built: Path
    cache: Path
    binder_repo: str | None
    hook: str | None
    fingerprint_files: tuple[Path, ...]

    @property
    def hook_file(self) -> Path | None:
        """Source file of the hook module under `root`, if the module is configured.

        Returns
        -------
        pathlib.Path or None
            `<root>/a/b.py` or `<root>/a/b/__init__.py` for `hook = "a.b"`, whichever
            exists, else the module path with `.py`; `None` without a hook.
        """
        if self.hook is None:
            return None
        base = self.root.joinpath(*self.hook.split("."))
        package_init = base / "__init__.py"
        return package_init if package_init.is_file() else base.with_suffix(".py")


def find_pyproject(start: Path) -> Path | None:
    """Return the nearest `pyproject.toml` with a `[tool.zallery]` table.

    Parameters
    ----------
    start : pathlib.Path
        Directory to search from; its parents are searched in turn.

    Returns
    -------
    pathlib.Path or None
        The file, or `None` if no ancestor holds one with the table.
    """
    for directory in (start.resolve(), *start.resolve().parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            with candidate.open("rb") as handle:
                data = tomllib.load(handle)
            if _TABLE in data.get("tool", {}):
                return candidate
    return None


def load_config(start: Path) -> Config:
    """Load the configuration that applies to `start`.

    Parameters
    ----------
    start : pathlib.Path
        Directory to search from, typically the working directory.

    Returns
    -------
    Config
        The resolved configuration, with relative paths made absolute under the
        project root.

    Raises
    ------
    ConfigError
        If no `pyproject.toml` with a `[tool.zallery]` table is found, or the table
        has unknown keys.
    """
    pyproject = find_pyproject(start)
    if pyproject is None:
        raise ConfigError(
            f"No pyproject.toml with a [tool.zallery] table found in {start} or its "
            "parents."
        )
    with pyproject.open("rb") as handle:
        table = tomllib.load(handle)["tool"][_TABLE]
    unknown = sorted(set(table) - _KNOWN_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown keys in [tool.zallery] of {pyproject}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(_KNOWN_KEYS))}."
        )

    root = pyproject.parent
    examples = root / table.get("examples", "docs/examples")
    return Config(
        root=root,
        examples=examples,
        built=root / table["built"] if "built" in table else examples / "_built",
        cache=root / table.get("cache", ".gallery-cache"),
        binder_repo=table.get("binder_repo"),
        hook=table.get("hook"),
        fingerprint_files=tuple(
            root / p for p in table.get("fingerprint", ["uv.lock"])
        ),
    )


def deps_fingerprint(config: Config) -> str:
    """Return a string identifying everything that shapes an example's render.

    Any change to it makes every cached example miss: the configured fingerprint
    files (the lockfile by default), the `zallery` sources, the hook module, and
    the configuration values themselves. The Binder ref is deliberately not part of
    it, so renders are reused across branches.

    Parameters
    ----------
    config : Config
        The project configuration.

    Returns
    -------
    str
        Concatenated, locale-independent text of all the inputs.
    """
    parts: list[str] = [
        json.dumps(
            {
                "examples": str(config.examples.relative_to(config.root)),
                "built": str(config.built.relative_to(config.root)),
                "binder_repo": config.binder_repo,
                "hook": config.hook,
            },
            sort_keys=True,
        )
    ]
    # Read everything as UTF-8 so the text does not depend on the platform's locale.
    for path in config.fingerprint_files:
        if path.is_file():
            parts.append(f"\n# {path.relative_to(config.root)}\n")
            parts.append(path.read_text(encoding="utf-8"))
    for path in sorted(Path(zallery.__file__).resolve().parent.glob("*.py")):
        parts.append(f"\n# zallery/{path.name}\n")
        parts.append(path.read_text(encoding="utf-8"))
    hook_file = config.hook_file
    if hook_file is not None and hook_file.is_file():
        parts.append(f"\n# {hook_file.relative_to(config.root)}\n")
        parts.append(hook_file.read_text(encoding="utf-8"))
    return "".join(parts)
