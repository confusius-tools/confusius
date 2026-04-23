"""Tests for the datasets registry and list_datasets()."""

from __future__ import annotations

import pytest

from confusius.datasets import list_datasets
from confusius.datasets._registry import _REGISTRY


# ---------------------------------------------------------------------------
# _REGISTRY
# ---------------------------------------------------------------------------


def test_registry_is_nonempty_tuple_of_triples():
    assert isinstance(_REGISTRY, tuple)
    assert len(_REGISTRY) > 0
    for name, size, bids_root in _REGISTRY:
        assert isinstance(name, str)
        assert isinstance(size, int)
        assert size > 0
        assert isinstance(bids_root, str)
        assert bids_root


def test_registry_entries_are_importable():
    """Each fetcher name in the registry should be importable from the package."""
    import confusius.datasets as ds

    for name, _, _ in _REGISTRY:
        assert hasattr(ds, name), f"{name} not found in confusius.datasets"


# ---------------------------------------------------------------------------
# list_datasets()
# ---------------------------------------------------------------------------


def test_list_datasets_prints_table(tmp_path, capsys):
    list_datasets(data_dir=tmp_path)
    captured = capsys.readouterr().out
    assert "Available Datasets" in captured
    assert "On disk" in captured
    for name, _, _ in _REGISTRY:
        assert name in captured


def test_list_datasets_marks_cached_datasets(tmp_path, capsys):
    """Datasets with a non-empty cache dir are marked as cached, others are not."""
    cached_name, _, cached_root = _REGISTRY[0]
    (tmp_path / cached_root).mkdir()
    (tmp_path / cached_root / "dataset_description.json").write_text("{}")

    list_datasets(data_dir=tmp_path)
    captured = capsys.readouterr().out

    # Find the line containing each fetcher name and check its marker.
    for name, _, _ in _REGISTRY:
        line = next(ln for ln in captured.splitlines() if name in ln)
        if name == cached_name:
            assert "✓" in line
        else:
            assert "✗" in line


def test_list_datasets_shows_human_readable_sizes(tmp_path, capsys):
    list_datasets(data_dir=tmp_path)
    captured = capsys.readouterr().out
    assert any(unit in captured for unit in (" KB", " MB", " GB", " TB"))
