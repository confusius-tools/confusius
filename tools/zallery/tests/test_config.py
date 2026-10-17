"""Tests for zallery.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from zallery.config import ConfigError, deps_fingerprint, load_config


def _write_pyproject(root: Path, table: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f"[tool.zallery]\n{table}\n")


def test_load_config_resolves_paths_under_the_project_root(tmp_path: Path) -> None:
    _write_pyproject(
        tmp_path,
        'examples = "site/examples"\ncache = "build/.cache"\n'
        'binder_repo = "me/repo"\nhook = "conf.hooks"\nfingerprint = ["lock.txt"]\n',
    )

    config = load_config(tmp_path)

    assert config.root == tmp_path.resolve()
    assert config.examples == tmp_path.resolve() / "site" / "examples"
    assert config.built == tmp_path.resolve() / "site" / "examples" / "_built"
    assert config.cache == tmp_path.resolve() / "build" / ".cache"
    assert config.binder_repo == "me/repo"
    assert config.hook == "conf.hooks"
    assert config.hook_file == tmp_path.resolve() / "conf" / "hooks.py"
    assert config.fingerprint_files == (tmp_path.resolve() / "lock.txt",)


def test_load_config_defaults(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, "")

    config = load_config(tmp_path)

    assert config.examples == tmp_path.resolve() / "docs" / "examples"
    assert config.built == config.examples / "_built"
    assert config.cache == tmp_path.resolve() / ".gallery-cache"
    assert config.binder_repo is None
    assert config.hook is None
    assert config.hook_file is None
    assert config.fingerprint_files == (tmp_path.resolve() / "uv.lock",)


def test_load_config_searches_parent_directories(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, 'examples = "ex"\n')
    nested = tmp_path / "docs" / "deep"
    nested.mkdir(parents=True)
    # A pyproject without the table on the way up is skipped, not an error.
    (tmp_path / "docs" / "pyproject.toml").write_text('[project]\nname = "x"\n')

    config = load_config(nested)

    assert config.root == tmp_path.resolve()
    assert config.examples == tmp_path.resolve() / "ex"


def test_load_config_without_table_raises(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

    with pytest.raises(ConfigError, match=r"\[tool\.zallery\]"):
        load_config(tmp_path)


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, 'exmaples = "docs/examples"\n')

    with pytest.raises(ConfigError, match="Unknown keys .*exmaples"):
        load_config(tmp_path)


def test_deps_fingerprint_tracks_lockfile_hook_and_config(tmp_path: Path) -> None:
    _write_pyproject(tmp_path, 'hook = "hooks"\n')
    (tmp_path / "uv.lock").write_text("v1\n")
    (tmp_path / "hooks.py").write_text("def gallery_retheme(fig, theme): pass\n")
    baseline = deps_fingerprint(load_config(tmp_path))

    (tmp_path / "uv.lock").write_text("v2\n")
    after_lock = deps_fingerprint(load_config(tmp_path))
    (tmp_path / "hooks.py").write_text("def gallery_retheme(fig, theme): ...\n")
    after_hook = deps_fingerprint(load_config(tmp_path))
    _write_pyproject(tmp_path, 'hook = "hooks"\nbinder_repo = "me/repo"\n')
    after_config = deps_fingerprint(load_config(tmp_path))

    assert len({baseline, after_lock, after_hook, after_config}) == 4
    assert deps_fingerprint(load_config(tmp_path)) == after_config
