"""Unit tests for confusius.io.utils module."""

import sys
from pathlib import Path

import pytest

from confusius.io.utils import check_path


def test_check_path_str_resolution(tmp_path):
    """String path resolves correctly."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    result = check_path(str(file_path), type="file")

    assert isinstance(result, Path)
    assert result.resolve() == file_path.resolve()


def test_check_path_path_object(tmp_path):
    """Path object resolves correctly."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    result = check_path(file_path, type="file")

    assert result.resolve() == file_path.resolve()


def test_check_path_expands_tilde(tmp_path, monkeypatch):
    """Tilde expands to home directory."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    monkeypatch.setenv("HOME", str(tmp_path))
    if sys.platform == "win32":
        # On Windows, Path.expanduser() uses USERPROFILE, not HOME.
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    result = check_path("~/test.txt", type="file")

    assert result.resolve() == file_path.resolve()


def test_check_path_existing_file(tmp_path):
    """type='file' with existing file succeeds."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    result = check_path(file_path, label="my_file", type="file")

    assert result.is_file()


def test_check_path_existing_directory(tmp_path):
    """type='dir' with existing directory succeeds."""
    result = check_path(tmp_path, label="data_root", type="dir")

    assert result.is_dir()


def test_check_path_no_type_check(tmp_path):
    """type=None (no type check) succeeds even if path doesn't exist."""
    non_existent = tmp_path / "does_not_exist"

    result = check_path(non_existent, type=None)

    assert result.resolve() == non_existent.resolve()


def test_check_path_nonexistent_file(tmp_path):
    """type='file' with non-existent file raises `ValueError`."""
    non_existent = tmp_path / "does_not_exist.txt"

    with pytest.raises(ValueError, match="my_file argument must be a valid file path"):
        check_path(non_existent, label="my_file", type="file")


def test_check_path_directory_as_file(tmp_path):
    """type='file' with directory raises `ValueError`."""

    with pytest.raises(ValueError, match="my_path argument must be a valid file path"):
        check_path(tmp_path, label="my_path", type="file")


def test_check_path_nonexistent_directory(tmp_path):
    """type='dir' with non-existent directory raises `ValueError`."""
    non_existent = tmp_path / "does_not_exist"

    with pytest.raises(
        ValueError, match="data_dir argument must be a valid directory path"
    ):
        check_path(non_existent, label="data_dir", type="dir")


def test_check_path_file_as_directory(tmp_path):
    """type='dir' with file raises `ValueError`."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    with pytest.raises(
        ValueError, match="test_file argument must be a valid directory path"
    ):
        check_path(file_path, label="test_file", type="dir")


def test_check_path_invalid_type():
    """Invalid path type raises `TypeError`."""

    with pytest.raises(TypeError, match="path argument must be a pathlib.Path"):
        check_path(None, label="path", type="file")

    with pytest.raises(TypeError, match="path argument must be a pathlib.Path"):
        check_path(123, label="path", type="file")


def test_check_path_custom_error_labels(tmp_path):
    """Custom label appears in error messages."""
    non_existent = tmp_path / "missing.dat"

    with pytest.raises(
        ValueError, match="input_data argument must be a valid file path"
    ):
        check_path(non_existent, label="input_data", type="file")
