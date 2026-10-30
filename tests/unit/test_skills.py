"""Tests for bundled skill discovery and safe installation."""

import shutil
import sys
from importlib import resources
from pathlib import Path

import pytest

from confusius._cli import build_parser, main
from confusius._utils.skills import install_skills, list_skills

SKILL = "confusius-load-data"


def test_discover_bundled_skills() -> None:
    """The distributed import skill is discoverable, unlike the release workflow."""
    assert SKILL in list_skills()
    assert "release" not in list_skills()
    assert list(list_skills()) == sorted(list_skills())
    assert "Load custom" in list_skills()[SKILL]


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
    ("target", "directory"),
    [
        (".agents/skills", ".agents"),
        (".claude/skills", ".claude"),
        ("custom/skills", "custom"),
    ],
)
@pytest.mark.parametrize("global_install", [False, True])
def test_install_complete_skill(
    tmp_path: Path,
    monkeypatch,
    target: str,
    directory: str,
    global_install: bool,
) -> None:
    """Relative targets in both scopes receive the complete skill."""
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(project)
    (destination,) = install_skills(SKILL, global_install=global_install, target=target)
    expected_root = home if global_install else project
    assert destination == expected_root / directory / "skills" / SKILL
    source = Path(str(resources.files("confusius").joinpath("_skills", SKILL)))
    expected = {
        file.relative_to(source): file.read_bytes()
        for file in source.rglob("*")
        if file.is_file()
    }
    actual = {
        file.relative_to(destination): file.read_bytes()
        for file in destination.rglob("*")
        if file.is_file()
    }
    assert actual == expected
    assert Path("references/geometry.md") in actual
    assert Path("scripts/examples.py") in actual
    other_root = project if global_install else home
    assert not (other_root / directory).exists()
    other_directory = ".claude" if directory == ".agents" else ".agents"
    assert not (expected_root / other_directory).exists()


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


def test_copy_failure_preserves_existing_skill(tmp_path: Path, monkeypatch) -> None:
    """A missing resource during staging does not destroy a forced replacement."""
    root = tmp_path / "resources"
    skill = root / "_skills" / SKILL
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {SKILL}\ndescription: Example skill.\n---\nInstructions",
        encoding="utf-8",
    )
    try:
        (skill / "broken").symlink_to(root / "missing")
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform.")
    monkeypatch.setattr(resources, "files", lambda package: root)
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / ".agents" / "skills" / SKILL
    shutil.copytree(skill, destination, ignore=shutil.ignore_patterns("broken"))
    (destination / "SKILL.md").write_text("original", encoding="utf-8")
    with pytest.raises(OSError):
        install_skills(SKILL, force=True)
    assert (destination / "SKILL.md").read_text(encoding="utf-8") == "original"
    assert list(destination.parent.iterdir()) == [destination]


def test_cli_list_and_install(tmp_path: Path, monkeypatch, capsys) -> None:
    """Namespace dispatch lists and installs without invoking the napari launcher."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["confusius", "skills", "list"])
    main()
    listing = capsys.readouterr().out
    assert "Available Agent Skills" in listing
    assert "Name" in listing and "Description" in listing
    assert SKILL in listing and "Load custom" in listing
    monkeypatch.setattr(
        "sys.argv",
        [
            "confusius",
            "skills",
            "install",
            SKILL,
            "--target",
            ".claude/skills",
        ],
    )
    main()
    destination = tmp_path / ".claude" / "skills" / SKILL
    assert (destination / "SKILL.md").is_file()
    assert str(destination) in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--force" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", [*sys.argv, "--force"])
    main()
    assert "Installed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["install", SKILL, "--agent", "pi"],
        ["install", SKILL, "--scope", "project"],
        ["install", "--claude"],
        ["install", "--target"],
        ["install", "missing"],
    ],
)
def test_cli_errors(arguments, monkeypatch, capsys) -> None:
    """CLI failures report unknown skills and reject obsolete options."""
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
    assert "Available Agent Skills" in output
    assert "Name" in output and "Description" in output
    assert "first-skill" in output and "second-skill" in output
    for word in description.split():
        assert word in output
    assert "Another import workflow." in output
    assert max(len(line) for line in output.splitlines()) <= 72
    assert "\x1b[" not in output


def test_main_help_advertises_skills() -> None:
    """The default launcher's help advertises the new namespace."""
    assert "skills" in build_parser().format_help()


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


@pytest.mark.parametrize("global_install", [False, True])
@pytest.mark.parametrize("target", [None, ".claude/skills", "custom/skills"])
def test_default_installs_all_skills(
    two_skills: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
    global_install: bool,
    target: str | None,
) -> None:
    """The CLI defaults to the project and supports global and target overrides."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    arguments = ["confusius", "skills", "install"]
    if global_install:
        arguments.append("--global")
    if target is not None:
        arguments.extend(["--target", target])
    monkeypatch.setattr("sys.argv", arguments)
    main()
    root = home if global_install else two_skills
    parent = root / (target or ".agents/skills")
    other_root = two_skills if global_install else home
    assert not (other_root / ".agents").exists()
    assert not (other_root / ".claude").exists()
    assert not (other_root / "custom").exists()
    assert sorted(path.name for path in parent.iterdir()) == ["a-example", "z-example"]
    for name in ("a-example", "z-example"):
        assert (parent / name / "reference.txt").read_text(encoding="utf-8") == name
    assert len(capsys.readouterr().out.splitlines()) == 2


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
    destinations = install_skills(force=True)
    assert destinations == [parent / "a-example", existing]
    assert not (existing / "custom.txt").exists()


def test_named_install_is_selective(two_skills: Path) -> None:
    """An explicit name still installs only the requested skill."""
    parent = two_skills / ".agents" / "skills"
    assert install_skills("z-example") == [parent / "z-example"]
    assert not (parent / "a-example").exists()


def test_bulk_stages_before_replacement(two_skills: Path, tmp_path: Path) -> None:
    """An unreadable second skill preserves the entire previous installation."""
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


@pytest.mark.parametrize("failure_phase", ["backup", "install"])
def test_bulk_replacement_failure_restores_originals(
    two_skills: Path, monkeypatch, failure_phase: str
) -> None:
    """A failed second backup or replacement restores local changes in both skills."""
    destinations = install_skills()
    for destination in destinations:
        (destination / "custom.txt").write_text(destination.name, encoding="utf-8")
    originals = {
        destination: {
            file.relative_to(destination): file.read_bytes()
            for file in destination.rglob("*")
            if file.is_file()
        }
        for destination in destinations
    }
    original_rename = Path.rename

    def fail_second_replacement(source: Path, target: Path) -> Path:
        failing_parent = "new" if failure_phase == "install" else "skills"
        if source.name == "z-example" and source.parent.name == failing_parent:
            raise OSError("simulated replacement failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second_replacement)
    with pytest.raises(OSError, match="simulated replacement failure"):
        install_skills(force=True)
    for destination in destinations:
        assert {
            file.relative_to(destination): file.read_bytes()
            for file in destination.rglob("*")
            if file.is_file()
        } == originals[destination]
    assert sorted(path.name for path in destinations[0].parent.iterdir()) == [
        "a-example",
        "z-example",
    ]


@pytest.mark.parametrize("existing", [None, "a-example", "z-example"])
def test_failed_bulk_install_removes_new_skills(
    two_skills: Path, monkeypatch, existing: str | None
) -> None:
    """Rollback removes fresh installs and restores any previously existing skill."""
    parent = two_skills / ".agents/skills"
    if existing is not None:
        (destination,) = install_skills(existing)
        (destination / "custom.txt").write_text("original", encoding="utf-8")
    original_rename = Path.rename

    def fail_second_replacement(source: Path, target: Path) -> Path:
        if source.name == "z-example" and source.parent.name == "new":
            raise OSError("simulated replacement failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second_replacement)
    with pytest.raises(OSError, match="simulated replacement failure"):
        install_skills(force=True)
    expected = [] if existing is None else [existing]
    assert sorted(path.name for path in parent.iterdir()) == expected
    if existing is not None:
        assert (parent / existing / "custom.txt").read_text(
            encoding="utf-8"
        ) == "original"


@pytest.mark.parametrize("kind", ["file", "symlink", "broken-symlink"])
def test_failed_replacement_restores_non_directory(
    two_skills: Path, tmp_path: Path, monkeypatch, kind: str
) -> None:
    """Rollback restores files and symlinks exactly, without following symlink targets."""
    parent = two_skills / ".agents/skills"
    parent.mkdir(parents=True)
    destination = parent / "z-example"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    link_target = outside if kind == "symlink" else tmp_path / "missing"
    if kind == "file":
        destination.write_bytes(b"original file")
    else:
        try:
            destination.symlink_to(link_target, target_is_directory=True)
        except OSError:
            pytest.skip("Creating symlinks is unavailable on this platform.")
    original_rename = Path.rename

    def fail_replacement(source: Path, target: Path) -> Path:
        if source.name == "z-example" and source.parent.name == "new":
            raise OSError("simulated replacement failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_replacement)
    with pytest.raises(OSError, match="simulated replacement failure"):
        install_skills(force=True)
    assert list(parent.iterdir()) == [destination]
    if kind == "file":
        assert destination.read_bytes() == b"original file"
    else:
        assert destination.is_symlink()
        assert destination.readlink() == link_target
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("rollback_failure", ["restore", "remove"])
def test_rollback_failure_preserves_backups(
    two_skills: Path, monkeypatch, rollback_failure: str
) -> None:
    """Failed restoration preserves recovery files and still restores other skills."""
    destinations = install_skills()
    for destination in destinations:
        (destination / "custom.txt").write_text(destination.name, encoding="utf-8")
    original_rename = Path.rename
    original_rmtree = shutil.rmtree
    failed_name = "z-example" if rollback_failure == "restore" else "a-example"

    def fail_rename(source: Path, target: Path) -> Path:
        if source.name == "z-example" and source.parent.name == "new":
            raise OSError("simulated install failure")
        if (
            rollback_failure == "restore"
            and source.parent.name == "backups"
            and source.name == failed_name
        ):
            raise OSError("simulated restore failure")
        return original_rename(source, target)

    def fail_remove(path, *args, **kwargs) -> None:
        if rollback_failure == "remove" and Path(path) == destinations[0]:
            raise OSError("simulated removal failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", fail_rename)
    monkeypatch.setattr(shutil, "rmtree", fail_remove)
    with pytest.raises(OSError, match="rollback incomplete") as error:
        install_skills(force=True)
    parent = destinations[0].parent
    (recovery,) = parent.glob(".confusius-skill-*")
    assert str(recovery) in str(error.value)
    assert "simulated install failure" in str(error.value)
    assert (recovery / "backups" / failed_name / "custom.txt").read_text(
        encoding="utf-8"
    ) == failed_name
    restored_name = "a-example" if rollback_failure == "restore" else "z-example"
    assert (parent / restored_name / "custom.txt").read_text(
        encoding="utf-8"
    ) == restored_name


def test_late_destination_is_preserved(two_skills: Path, monkeypatch) -> None:
    """A destination appearing after preflight is protected and earlier installs roll back."""
    parent = two_skills / ".agents/skills"
    destination = parent / "z-example"
    original_copy = shutil.copyfileobj

    def create_late_destination(reader, writer) -> None:
        original_copy(reader, writer)
        if not destination.exists():
            destination.mkdir()
            (destination / "custom.txt").write_text(
                "created concurrently", encoding="utf-8"
            )

    monkeypatch.setattr(shutil, "copyfileobj", create_late_destination)
    with pytest.raises(FileExistsError, match="z-example"):
        install_skills()
    assert list(parent.iterdir()) == [destination]
    assert (destination / "custom.txt").read_text(
        encoding="utf-8"
    ) == "created concurrently"


@pytest.mark.parametrize("global_install", [False, True])
@pytest.mark.parametrize("use_tilde", [False, True])
def test_absolute_targets(
    tmp_path: Path, monkeypatch, global_install: bool, use_tilde: bool
) -> None:
    """Absolute and home-expanded targets are used directly, with either scope flag."""
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project)
    target = "~/custom/skills" if use_tilde else tmp_path / "absolute/skills"
    expected = home / "custom/skills" if use_tilde else target
    assert install_skills(SKILL, target=target, global_install=global_install) == [
        Path(expected) / SKILL
    ]
    assert (Path(expected) / SKILL / "SKILL.md").is_file()
    assert not (project / ".agents").exists()
    assert not (home / ".agents").exists()


def test_empty_bundle(tmp_path: Path, monkeypatch) -> None:
    """An empty bundle creates no installation directories."""
    (tmp_path / "_skills").mkdir()
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    monkeypatch.chdir(tmp_path)
    assert install_skills() == []
    assert not (tmp_path / ".agents").exists()
