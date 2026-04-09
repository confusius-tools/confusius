"""Tests for the datasets registry, list_datasets(), and format_bytes."""

from __future__ import annotations

import pytest

from confusius.datasets import _REGISTRY, list_datasets
from confusius.datasets._utils import format_bytes

# ---------------------------------------------------------------------------
# format_bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size_bytes", "expected"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1024, "1 KB"),
        (1_500_000, "1.431 MB"),
        (6_982_575_320, "6.503 GB"),
    ],
)
def test_format_bytes(size_bytes, expected):
    assert format_bytes(size_bytes) == expected


# ---------------------------------------------------------------------------
# _REGISTRY
# ---------------------------------------------------------------------------


def test_registry_is_nonempty_tuple_of_pairs():
    assert isinstance(_REGISTRY, tuple)
    assert len(_REGISTRY) > 0
    for name, size in _REGISTRY:
        assert isinstance(name, str)
        assert isinstance(size, int)
        assert size > 0


def test_registry_entries_are_importable():
    """Each fetcher name in the registry should be importable from the package."""
    import confusius.datasets as ds

    for name, _ in _REGISTRY:
        assert hasattr(ds, name), f"{name} not found in confusius.datasets"


# ---------------------------------------------------------------------------
# list_datasets()
# ---------------------------------------------------------------------------


def test_list_datasets_prints_table(capsys):
    list_datasets()
    captured = capsys.readouterr().out
    assert "Available Datasets" in captured
    for name, _ in _REGISTRY:
        assert name in captured
