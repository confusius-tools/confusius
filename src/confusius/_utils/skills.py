"""Discovery and installation of bundled Agent Skills."""

import shutil
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from tempfile import mkdtemp

import yaml


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
    global_install: bool = False,
    target: str | Path = ".agents/skills",
    force: bool = False,
) -> list[Path]:
    """Install one or all bundled skills locally or in the user's home.

    Parameters
    ----------
    name : str, optional
        Name returned by `list_skills`. If not provided, install every bundled
        skill. All destinations are checked and resources staged before replacement.
        Existing installations are backed up and restored on filesystem errors.
    global_install : bool, default: False
        Whether to install relative to the user's home instead of the current
        working directory.
    target : str or pathlib.Path, default: ".agents/skills"
        Skills directory. Relative paths resolve against the current working
        directory, or the user's home with `global_install`. Absolute paths are
        used directly; a leading `~` is expanded.
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
        If the skill is unknown.
    FileExistsError
        If the destination exists and replacement was not requested.
    OSError
        If resources cannot be copied or filesystem operations fail. If rollback
        also fails, the error reports where backups are preserved for recovery.
    """
    skills = list_skills()
    if name is not None and name not in skills:
        raise ValueError(f"Unknown bundled skill: {name!r}.")

    names = [name] if name is not None else list(skills)
    root = Path.home() if global_install else Path.cwd()
    parent = root / Path(target).expanduser()
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
    # Keep backups on the same filesystem, and retain them if restoration fails.
    staging = Path(mkdtemp(prefix=".confusius-skill-", dir=parent))
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        (staging / "new").mkdir()
        (staging / "backups").mkdir()
        for skill in names:
            _copy_skill(source.joinpath(skill), staging / "new" / skill)
        for destination in destinations:
            if destination.exists() or destination.is_symlink():
                if not force:
                    raise FileExistsError(
                        f"Skill already exists at {destination}. Use --force to replace it."
                    )
                backup = staging / "backups" / destination.name
                destination.rename(backup)
                backups[destination] = backup
            (staging / "new" / destination.name).rename(destination)
            installed.append(destination)
    except OSError as error:
        rollback_errors = []
        for destination in reversed(destinations):
            try:
                if destination in installed:
                    shutil.rmtree(destination)
                if destination in backups:
                    backups[destination].rename(destination)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        if rollback_errors:
            raise OSError(
                f"Installation failed ({error}); rollback incomplete "
                f"({'; '.join(rollback_errors)}). Recover backups from {staging}."
            ) from error
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(staging, ignore_errors=True)
    return destinations
