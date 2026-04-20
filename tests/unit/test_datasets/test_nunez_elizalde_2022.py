"""Unit tests for confusius.datasets._nunez_elizalde_2022."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from confusius.datasets import fetch_nunez_elizalde_2022, get_datasets_dir
from confusius.datasets._nunez_elizalde_2022 import (
    _BIDS_ROOT,
    _MISSING_INDEX_HINT,
    _OSF_PROJECT_ID,
)
from confusius.datasets._pooch import _MAX_DOWNLOAD_RETRIES

# Minimal fake index representing the different file categories in the dataset.
_FAKE_INDEX = {
    # Top-level BIDS metadata — always included.
    "dataset_description.json": {"osf_path": "/file001", "size": 100},
    "participants.tsv": {"osf_path": "/file002", "size": 200},
    # Subject-level file — always included when subject passes the filter.
    "sub-CR020/sub-CR020_sessions.tsv": {"osf_path": "/file003", "size": 300},
    # Angio — always included regardless of task filter.
    "sub-CR020/ses-20191122/angio/sub-CR020_ses-20191122_pwd.nii.gz": {
        "osf_path": "/file004",
        "size": 400,
    },
    # fUSI — task-filtered.
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz": {
        "osf_path": "/file005",
        "size": 500,
    },
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz": {
        "osf_path": "/file006",
        "size": 600,
    },
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz": {
        "osf_path": "/file009",
        "size": 900,
    },
    # Second session — session-filtered.
    "sub-CR020/ses-20191121/angio/sub-CR020_ses-20191121_pwd.nii.gz": {
        "osf_path": "/file007",
        "size": 700,
    },
    "sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_task-kalatsky_acq-slice01_pwd.nii.gz": {
        "osf_path": "/file008",
        "size": 800,
    },
    # Derivatives — should also be subject/session-filtered.
    "derivatives/allenccf_align/dataset_description.json": {
        "osf_path": "/file010",
        "size": 1000,
    },
    "derivatives/allenccf_align/structure_tree_safe_2017.csv": {
        "osf_path": "/file011",
        "size": 1100,
    },
    "derivatives/allenccf_align/sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz": {
        "osf_path": "/file012",
        "size": 1200,
    },
    "derivatives/allenccf_align/sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_space-fusi_desc-allenccf_dseg.nii.gz": {
        "osf_path": "/file013",
        "size": 1300,
    },
    "derivatives/allenccf_align/sub-OTHER/ses-20191122/fusi/sub-OTHER_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz": {
        "osf_path": "/file014",
        "size": 1400,
    },
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
    """Stub `get_index` so fetch tests don't hit the network."""
    with patch(
        "confusius.datasets._nunez_elizalde_2022.get_index",
        return_value=_FAKE_INDEX,
    ) as mock:
        yield mock


@pytest.fixture
def mock_retrieve(tmp_path):
    """Patch pooch.retrieve to create stub files instead of downloading."""
    bids_dir = tmp_path / _BIDS_ROOT
    with patch(
        "confusius.datasets._pooch.pooch.retrieve",
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


def test_get_datasets_dir_defaults_to_os_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFUSIUS_DATA", raising=False)
    with patch(
        "confusius.datasets._utils.pooch.os_cache",
        return_value=str(tmp_path / "cache"),
    ):
        result = get_datasets_dir()
    assert result == tmp_path / "cache"
    assert result.is_dir()


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

    with patch("confusius.datasets._pooch.pooch.retrieve") as mock_retrieve:
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


def test_fetch_filters_derivatives_by_subject_and_session(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        subjects=["CR020"],
        sessions=["20191122"],
    )

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    # Shared derivative files (no subject/session folder) should still be included.
    assert "dataset_description.json" in downloaded
    assert "structure_tree_safe_2017.csv" in downloaded
    # Matching derivative subject/session is included.
    assert "sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz" in downloaded
    # Non-matching derivative subject/session are excluded.
    assert (
        "sub-CR020_ses-20191121_space-fusi_desc-allenccf_dseg.nii.gz" not in downloaded
    )
    assert (
        "sub-OTHER_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz" not in downloaded
    )


def test_fetch_acq_filter_excludes_non_matching_fusi_includes_angio(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, acqs=["slice03"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert (
        "sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz" in downloaded
    )
    assert (
        "sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz" not in downloaded
    )
    # Angio is always included regardless of acquisition filter.
    assert "sub-CR020_ses-20191122_pwd.nii.gz" in downloaded


def test_fetch_subject_filter(tmp_path, mock_retrieve):
    index_with_two_subjects = {
        **_FAKE_INDEX,
        "sub-OTHER/sub-OTHER_sessions.tsv": {"osf_path": "/file999", "size": 999},
        "sub-OTHER/ses-20191122/angio/sub-OTHER_ses-20191122_pwd.nii.gz": {
            "osf_path": "/file998",
            "size": 998,
        },
    }
    with patch(
        "confusius.datasets._nunez_elizalde_2022.get_index",
        return_value=index_with_two_subjects,
    ):
        fetch_nunez_elizalde_2022(data_dir=tmp_path, subjects=["CR020"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-OTHER_sessions.tsv" not in downloaded
    assert "sub-OTHER_ses-20191122_pwd.nii.gz" not in downloaded


def test_fetch_accepts_string_filters(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        subjects="CR020",
        sessions="20191122",
        tasks="kalatsky",
        acqs="slice01",
    )

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz" in downloaded
    assert (
        "sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz"
        not in downloaded
    )


# ---------------------------------------------------------------------------
# fetch_nunez_elizalde_2022 — retry behaviour
# ---------------------------------------------------------------------------


def test_fetch_retries_on_transient_failure(tmp_path, mock_get_index):
    """Transient network errors are retried and eventually succeed."""
    bids_dir = tmp_path / _BIDS_ROOT

    target_rel = (
        "sub-CR020/ses-20191122/fusi/"
        "sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz"
    )
    for rel in _FAKE_INDEX:
        if rel == target_rel:
            continue
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    call_count = {"n": 0}

    def flaky_retrieve(url, known_hash, fname, path, progressbar):
        call_count["n"] += 1
        if call_count["n"] < _MAX_DOWNLOAD_RETRIES:
            raise requests.exceptions.ReadTimeout("simulated timeout")
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
        return str(dest)

    with (
        patch(
            "confusius.datasets._pooch.pooch.retrieve",
            side_effect=flaky_retrieve,
        ),
        patch("confusius.datasets._pooch.time.sleep"),
    ):
        fetch_nunez_elizalde_2022(data_dir=tmp_path)

    assert call_count["n"] == _MAX_DOWNLOAD_RETRIES


def test_fetch_raises_after_max_retries(tmp_path, mock_get_index):
    """Persistent network errors propagate after the retry budget is exhausted."""
    bids_dir = tmp_path / _BIDS_ROOT

    target_rel = (
        "sub-CR020/ses-20191122/fusi/"
        "sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz"
    )
    for rel in _FAKE_INDEX:
        if rel == target_rel:
            continue
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    def always_fails(url, known_hash, fname, path, progressbar):
        raise requests.exceptions.ReadTimeout("persistent timeout")

    with (
        patch(
            "confusius.datasets._pooch.pooch.retrieve",
            side_effect=always_fails,
        ) as mock_retrieve,
        patch("confusius.datasets._pooch.time.sleep"),
    ):
        with pytest.raises(requests.exceptions.ReadTimeout):
            fetch_nunez_elizalde_2022(data_dir=tmp_path)

    assert mock_retrieve.call_count == _MAX_DOWNLOAD_RETRIES


# ---------------------------------------------------------------------------
# fetch_nunez_elizalde_2022 — refresh behaviour
# ---------------------------------------------------------------------------


def test_fetch_refresh_passes_flag_to_get_index(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, refresh=True)
    mock_get_index.assert_called_once_with(
        tmp_path / _BIDS_ROOT,
        _OSF_PROJECT_ID,
        _BIDS_ROOT,
        refresh=True,
        missing_index_hint=_MISSING_INDEX_HINT,
    )
