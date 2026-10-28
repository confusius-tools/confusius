"""Discovery and installation of bundled Agent Skills."""

import shutil
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import yaml

AGENT_SKILL_DIRS = {
    "pi": ".agents/skills",
    "codex": ".agents/skills",
    "claude-code": ".claude/skills",
    "cursor": ".cursor/skills",
}
"""Documented skill directories relative to a project root or user home."""


def list_skills() -> dict[str, str]:
    """List bundled Agent Skills and their frontmatter descriptions.

    Returns
    -------
    dict[str, str]
        Skill names mapped to descriptions, sorted by name.

    Raises
    ------
    ValueError
        If a bundled skill has invalid frontmatter or no description.
    OSError
        If bundled resources cannot be read.
    """
    root = resources.files("confusius").joinpath("_skills")
    skills = {}
    for child in sorted(root.iterdir(), key=lambda child: child.name):
        document = child.joinpath("SKILL.md")
        if not child.is_dir() or not document.is_file():
            continue
        lines = document.read_text(encoding="utf-8").splitlines()
        try:
            if not lines or lines[0] != "---":
                raise ValueError("Missing YAML frontmatter.")
            end = lines.index("---", 1)
            metadata = yaml.safe_load("\n".join(lines[1:end]))
            description = (
                metadata.get("description") if isinstance(metadata, dict) else None
            )
            if not isinstance(description, str) or not description.strip():
                raise ValueError("A non-empty description is required.")
        except (ValueError, yaml.YAMLError) as error:
            raise ValueError(
                f"Invalid bundled skill {child.name!r}: {error}"
            ) from error
        skills[child.name] = " ".join(description.split())
    return skills


def _copy_skill(source: Traversable, destination: Path) -> None:
    """Copy a resource directory without requiring filesystem-backed resources.

    Parameters
    ----------
    source : importlib.resources.abc.Traversable
        Bundled skill directory.
    destination : pathlib.Path
        New directory to create.

    Raises
    ------
    OSError
        If a resource cannot be read or a destination cannot be written.
    """
    destination.mkdir()
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_skill(child, target)
        else:
            with child.open("rb") as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)


def install_skills(
    name: str | None = None,
    *,
    agent: str,
    scope: Literal["global", "project"],
    force: bool = False,
) -> list[Path]:
    """Install one or all bundled skills in an explicitly chosen scope.

    Parameters
    ----------
    name : str, optional
        Name returned by `list_skills`. If not provided, install every bundled
        skill. All destinations are checked and resources staged before replacement.
    agent : {"pi", "codex", "claude-code", "cursor"}
        Assistant whose documented skill directory to use.
    scope : {"global", "project"}
        Installation relative to the user's home or current working directory.
    force : bool, default: False
        Whether to replace an existing skill, including stale files. An existing
        skill symlink is replaced without modifying its target.

    Returns
    -------
    list[pathlib.Path]
        Installed skill directories, sorted by name.

    Raises
    ------
    ValueError
        If the skill, assistant, or scope is unknown.
    FileExistsError
        If the destination exists and replacement was not requested.
    OSError
        If resources cannot be copied or filesystem operations fail.
    """
    skills = list_skills()
    if name is not None and name not in skills:
        raise ValueError(f"Unknown bundled skill: {name!r}.")
    if agent not in AGENT_SKILL_DIRS:
        raise ValueError(f"Unknown assistant: {agent!r}.")
    if scope not in ("global", "project"):
        raise ValueError(f"Unknown installation scope: {scope!r}.")

    names = [name] if name is not None else list(skills)
    root = Path.home() if scope == "global" else Path.cwd()
    parent = root / AGENT_SKILL_DIRS[agent]
    destinations = [parent / skill for skill in names]
    for destination in destinations:
        if (destination.exists() or destination.is_symlink()) and not force:
            raise FileExistsError(
                f"Skill already exists at {destination}. Use --force to replace it."
            )
    if not destinations:
        return []

    source = resources.files("confusius").joinpath("_skills")
    parent.mkdir(parents=True, exist_ok=True)
    # Stage all resources before replacing anything, so read errors preserve old skills.
    with TemporaryDirectory(prefix=".confusius-skill-", dir=parent) as staging:
        for skill in names:
            _copy_skill(source.joinpath(skill), Path(staging) / skill)
        for destination in destinations:
            if destination.exists() or destination.is_symlink():
                if not force:
                    raise FileExistsError(
                        f"Skill already exists at {destination}. Use --force to replace it."
                    )
                if destination.is_symlink() or not destination.is_dir():
                    destination.unlink()
                else:
                    shutil.rmtree(destination)
            (Path(staging) / destination.name).rename(destination)
    return destinations
