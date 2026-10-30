"""Tests for bundled skill discovery and installation."""

import sys
from importlib import resources
from pathlib import Path

import pytest

from confusius._cli import build_parser, main
from confusius._utils.skills import install_skills, list_skills

SKILL = "confusius-load-data"


def _snapshot(root: Path) -> dict[Path, bytes]:
    """Read every file below `root`, keyed by its relative path."""
    return {
        file.relative_to(root): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
    }


def test_discover_bundled_skills() -> None:
    """The distributed import skill is discoverable, unlike the release workflow."""
    skills = list_skills()
    assert skills.get(SKILL)
    assert "release" not in skills
    assert list(skills) == sorted(skills)


def test_discovery_requires_skill_file(tmp_path: Path, monkeypatch) -> None:
    """Only direct resource directories containing SKILL.md are skills."""
    root = tmp_path / "_skills"
    for name in ("z-example", "a-example", "incomplete"):
        (root / name).mkdir(parents=True)
    for name in ("z-example", "a-example"):
        (root / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Description for {name}.\n---\nInstructions",
            encoding="utf-8",
        )
    (root / "README.md").write_text("index", encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    assert list_skills() == {
        "a-example": "Description for a-example.",
        "z-example": "Description for z-example.",
    }


@pytest.mark.parametrize(
    "description",
    [
        '"Load custom data: arrays and metadata."',
        ">-\n  Load custom data:\n  arrays and metadata.",
        "|\n  Load custom data:\n  arrays and metadata.",
    ],
)
def test_description_yaml_forms(tmp_path: Path, monkeypatch, description: str) -> None:
    """Descriptions support normal YAML quoting and multiline scalars."""
    skill = tmp_path / "_skills" / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: example\ndescription: {description}\n---\n# Instructions",
        encoding="utf-8",
    )
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    assert list_skills() == {"example": "Load custom data: arrays and metadata."}


@pytest.mark.parametrize(
    "document",
    [
        "No frontmatter",
        "---\nno closing delimiter",
        "---\ndescription: [\n---",
        "---\n- not a mapping\n---",
        "---\nname: example\n---",
        "---\ndescription: 123\n---",
        '---\ndescription: ""\n---',
    ],
)
def test_invalid_frontmatter(tmp_path: Path, monkeypatch, document: str) -> None:
    """Broken bundled metadata produces a descriptive error, not an incomplete listing."""
    skill = tmp_path / "_skills" / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(document, encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="Invalid bundled skill 'example'"):
        list_skills()


@pytest.mark.parametrize(
    "target", [".agents/skills", "custom/skills", "~/tilde/skills", "absolute"]
)
@pytest.mark.parametrize("global_install", [False, True])
def test_install_location(
    tmp_path: Path, monkeypatch, target: str, global_install: bool
) -> None:
    """Relative targets follow the scope; absolute and `~` targets are used directly."""
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project)
    if target == "absolute":
        target = str(tmp_path / "absolute/skills")
        expected = Path(target)
    elif target.startswith("~"):
        expected = home / "tilde/skills"
    else:
        expected = (home if global_install else project) / target
    (destination,) = install_skills(SKILL, global_install=global_install, target=target)
    assert destination == expected / SKILL
    source = Path(str(resources.files("confusius").joinpath("_skills", SKILL)))
    assert _snapshot(destination) == _snapshot(source)
    # Nothing is written outside the skill, and no staging directory is left behind.
    for file in tmp_path.rglob("*"):
        assert not file.is_file() or file.is_relative_to(destination)
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "broken-symlink"])
def test_overwrite_protection(tmp_path: Path, monkeypatch, kind: str) -> None:
    """Replacement requires force, removes stale content, and never follows a skill symlink."""
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / ".agents" / "skills" / SKILL
    destination.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    if kind == "directory":
        destination.mkdir()
        (destination / "stale.txt").write_text("custom skill", encoding="utf-8")
    elif kind == "file":
        destination.write_text("custom skill", encoding="utf-8")
    else:
        try:
            destination.symlink_to(
                outside if kind == "symlink" else tmp_path / "missing",
                target_is_directory=True,
            )
        except OSError:
            pytest.skip("Creating symlinks is unavailable on this platform.")

    with pytest.raises(FileExistsError, match="--force"):
        install_skills(SKILL)
    if kind == "directory":
        assert (destination / "stale.txt").read_text(encoding="utf-8") == "custom skill"
    elif kind == "file":
        assert destination.read_text(encoding="utf-8") == "custom skill"
    else:
        assert destination.is_symlink()

    assert install_skills(SKILL, force=True) == [destination]
    assert not destination.is_symlink()
    assert (destination / "SKILL.md").is_file()
    assert not (destination / "stale.txt").exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("name", ["../release", "missing"])
def test_invalid_installation(tmp_path: Path, monkeypatch, name: str) -> None:
    """Invalid names fail before writing anything."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        install_skills(name)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def two_skills(tmp_path: Path, monkeypatch) -> Path:
    """Provide two independent bundled skills and a clean project directory."""
    bundle = tmp_path / "bundle"
    for name in ("a-example", "z-example"):
        skill = bundle / "_skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Description for {name}.\n---\nInstructions",
            encoding="utf-8",
        )
        (skill / "reference.txt").write_text(name, encoding="utf-8")
    monkeypatch.setattr(resources, "files", lambda package: bundle)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return project


def test_install_one_or_all(two_skills: Path) -> None:
    """A name installs only that skill; no name installs every bundled skill."""
    parent = two_skills / ".agents" / "skills"
    assert install_skills("z-example") == [parent / "z-example"]
    assert [path.name for path in parent.iterdir()] == ["z-example"]
    assert install_skills(force=True) == [parent / "a-example", parent / "z-example"]
    assert sorted(path.name for path in parent.iterdir()) == ["a-example", "z-example"]
    for name in ("a-example", "z-example"):
        assert (parent / name / "reference.txt").read_text(encoding="utf-8") == name


def test_bulk_conflict_preflight(two_skills: Path) -> None:
    """A conflict in the last skill prevents installing even the first skill."""
    parent = two_skills / ".agents" / "skills"
    existing = parent / "z-example"
    existing.mkdir(parents=True)
    (existing / "custom.txt").write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="z-example"):
        install_skills()
    assert list(parent.iterdir()) == [existing]
    assert (existing / "custom.txt").read_text(encoding="utf-8") == "original"


def test_copy_failure_preserves_existing_skills(two_skills: Path, tmp_path: Path) -> None:
    """An unreadable resource fails before any existing skill is replaced."""
    destinations = install_skills()
    broken = tmp_path / "bundle/_skills/z-example/broken"
    try:
        broken.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform.")
    for destination in destinations:
        (destination / "custom.txt").write_text("original", encoding="utf-8")
    with pytest.raises(OSError):
        install_skills(force=True)
    for destination in destinations:
        assert (destination / "custom.txt").read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in destinations[0].parent.iterdir()) == [
        "a-example",
        "z-example",
    ]


def test_cli_list(tmp_path: Path, monkeypatch, capsys) -> None:
    """The `skills list` namespace renders the bundled skill table."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["confusius", "skills", "list"])
    main()
    listing = capsys.readouterr().out
    assert "Available Agent Skills" in listing
    assert "Name" in listing and "Description" in listing
    assert SKILL in listing


def test_cli_install(tmp_path: Path, monkeypatch, capsys) -> None:
    """The `skills install` namespace forwards `--global` and `--target`."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["confusius", "skills", "install", SKILL, "--global", "--target", ".claude/skills"],
    )
    main()
    destination = home / ".claude" / "skills" / SKILL
    assert (destination / "SKILL.md").is_file()
    assert str(destination) in capsys.readouterr().out


def test_cli_install_requires_force(tmp_path: Path, monkeypatch, capsys) -> None:
    """Reinstalling an existing skill fails until `--force` is passed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["confusius", "skills", "install", SKILL])
    main()
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--force" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", [*sys.argv, "--force"])
    main()
    assert "Installed" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [[], ["install", "missing"]])
def test_cli_errors(arguments, monkeypatch, capsys) -> None:
    """CLI failures exit with a usage error."""
    monkeypatch.setattr("sys.argv", ["confusius", "skills", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_list_table_wraps_literal_descriptions(monkeypatch, capsys) -> None:
    """The listing wraps to terminal width without interpreting skill markup."""
    description = (
        "Load custom arrays with calibrated coordinates and acquisition metadata. "
        "Keep [bold]literal[/bold] text."
    )
    monkeypatch.setattr(
        "confusius._utils.skills.list_skills",
        lambda: {
            "first-skill": description,
            "second-skill": "Another import workflow.",
        },
    )
    monkeypatch.setenv("COLUMNS", "72")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr("sys.argv", ["confusius", "skills", "list"])
    main()
    output = capsys.readouterr().out
    assert "first-skill" in output and "second-skill" in output
    for word in description.split():
        assert word in output
    assert "Another import workflow." in output
    assert max(len(line) for line in output.splitlines()) <= 72
    assert "\x1b[" not in output


def test_main_help_advertises_skills() -> None:
    """The default launcher's help advertises the new namespace."""
    assert "skills" in build_parser().format_help()
