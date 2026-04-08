"""Unit tests for confusius.datasets."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from confusius.datasets import fetch_nunez_elizalde_2022, get_datasets_dir
from confusius.datasets._nunez_elizalde_2022 import _BIDS_ROOT, _INDEX_FILENAME

# Minimal fake index representing the different file categories in the dataset.
_FAKE_INDEX = {
    # Top-level BIDS metadata — always included.
    "dataset_description.json": "/file001",
    "participants.tsv": "/file002",
    # Subject-level file — always included when subject passes the filter.
    "sub-CR020/sub-CR020_sessions.tsv": "/file003",
    # Angio — always included regardless of task filter.
    "sub-CR020/ses-20191122/angio/sub-CR020_ses-20191122_pwd.nii.gz": "/file004",
    # fUSI — task-filtered.
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz": "/file005",
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz": "/file006",
    # Second session — session-filtered.
    "sub-CR020/ses-20191121/angio/sub-CR020_ses-20191121_pwd.nii.gz": "/file007",
    "sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_task-kalatsky_acq-slice01_pwd.nii.gz": "/file008",
}


def _make_retrieve(bids_dir: Path):
    """Return a pooch.retrieve side-effect that creates stub files on disk."""

    def _retrieve(url, known_hash, fname, path, progressbar):
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
        return str(dest)

    return _retrieve


@pytest.fixture
def mock_get_index(tmp_path):
    """Patch _get_index to return the fake index without network access."""
    with patch(
        "confusius.datasets._nunez_elizalde_2022._get_index",
        return_value=_FAKE_INDEX,
    ) as mock:
        yield mock


@pytest.fixture
def mock_retrieve(tmp_path):
    """Patch pooch.retrieve to create stub files instead of downloading."""
    bids_dir = tmp_path / _BIDS_ROOT
    with patch(
        "confusius.datasets._nunez_elizalde_2022.pooch.retrieve",
        side_effect=_make_retrieve(bids_dir),
    ) as mock:
        yield mock


# ---------------------------------------------------------------------------
# get_datasets_dir
# ---------------------------------------------------------------------------


def test_get_datasets_dir_uses_provided_path(tmp_path):
    result = get_datasets_dir(tmp_path / "custom")
    assert result == tmp_path / "custom"


def test_get_datasets_dir_creates_directory(tmp_path):
    target = tmp_path / "new_dir"
    assert not target.exists()
    get_datasets_dir(target)
    assert target.is_dir()


def test_get_datasets_dir_uses_env_var(tmp_path, monkeypatch):
    env_dir = tmp_path / "from_env"
    monkeypatch.setenv("CONFUSIUS_DATA", str(env_dir))
    result = get_datasets_dir()
    assert result == env_dir


# ---------------------------------------------------------------------------
# fetch_nunez_elizalde_2022 — return value and caching
# ---------------------------------------------------------------------------


def test_fetch_returns_bids_root(tmp_path, mock_get_index, mock_retrieve):
    result = fetch_nunez_elizalde_2022(data_dir=tmp_path)
    assert result == tmp_path / _BIDS_ROOT
    assert isinstance(result, Path)


def test_fetch_downloads_all_missing_files(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX)


def test_fetch_skips_existing_files(tmp_path, mock_get_index, mock_retrieve):
    # Pre-create two files in the cache.
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in ["dataset_description.json", "participants.tsv"]:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    fetch_nunez_elizalde_2022(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX) - 2


def test_fetch_returns_immediately_when_all_cached(tmp_path, mock_get_index):
    # Pre-create every file in the cache.
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in _FAKE_INDEX:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with patch(
        "confusius.datasets._nunez_elizalde_2022.pooch.retrieve"
    ) as mock_retrieve:
        fetch_nunez_elizalde_2022(data_dir=tmp_path)
        mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_nunez_elizalde_2022 — filters
# ---------------------------------------------------------------------------


def test_fetch_task_filter_excludes_non_matching_fusi_includes_angio(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, tasks=["kalatsky"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz" in downloaded
    assert (
        "sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz"
        not in downloaded
    )
    # Angio is always included regardless of task filter.
    assert "sub-CR020_ses-20191122_pwd.nii.gz" in downloaded


def test_fetch_session_filter(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, sessions=["20191122"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-CR020_ses-20191122_pwd.nii.gz" in downloaded
    assert "sub-CR020_ses-20191121_pwd.nii.gz" not in downloaded


def test_fetch_subject_filter(tmp_path, mock_retrieve):
    index_with_two_subjects = {
        **_FAKE_INDEX,
        "sub-OTHER/sub-OTHER_sessions.tsv": "/file999",
        "sub-OTHER/ses-20191122/angio/sub-OTHER_ses-20191122_pwd.nii.gz": "/file998",
    }
    with patch(
        "confusius.datasets._nunez_elizalde_2022._get_index",
        return_value=index_with_two_subjects,
    ):
        fetch_nunez_elizalde_2022(data_dir=tmp_path, subjects=["CR020"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-OTHER_sessions.tsv" not in downloaded
    assert "sub-OTHER_ses-20191122_pwd.nii.gz" not in downloaded


# ---------------------------------------------------------------------------
# fetch_nunez_elizalde_2022 — refresh behaviour
# ---------------------------------------------------------------------------


def test_fetch_refresh_passes_flag_to_get_index(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, refresh=True)
    mock_get_index.assert_called_once_with(tmp_path / _BIDS_ROOT, refresh=True)
