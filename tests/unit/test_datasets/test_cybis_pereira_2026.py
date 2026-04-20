"""Unit tests for confusius.datasets._cybis_pereira_2026."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from confusius.datasets import fetch_cybis_pereira_2026
from confusius.datasets._cybis_pereira_2026 import _BIDS_ROOT, _OSF_PROJECT_ID
from confusius.datasets._pooch import _MAX_DOWNLOAD_RETRIES

# Minimal fake index representing the different file categories in the dataset.
_FAKE_INDEX = {
    # Top-level BIDS metadata — always included.
    "dataset_description.json": {"osf_path": "/file001", "size": 100},
    "participants.tsv": {"osf_path": "/file002", "size": 200},
    # Rawdata — requires "rawdata" in datasets filter. Two sessions, two
    # acquisitions per session for sub-rat83; sub-rat84 has a single
    # session with no acquisition entity.
    "sub-rat83/ses-20220523/fus/sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz": {
        "osf_path": "/file003",
        "size": 1000,
    },
    "sub-rat83/ses-20220523/fus/sub-rat83_ses-20220523_task-openfield_acq-slice42_pwd.nii.gz": {
        "osf_path": "/file004",
        "size": 1000,
    },
    "sub-rat83/ses-20220524/fus/sub-rat83_ses-20220524_task-openfield_acq-slice32_pwd.nii.gz": {
        "osf_path": "/file005",
        "size": 1000,
    },
    "sub-rat83/ses-20220523/sub-rat83_ses-20220523_scans.tsv": {
        "osf_path": "/file006",
        "size": 50,
    },
    "sub-rat84/ses-20210407/fus/sub-rat84_ses-20210407_task-openfield_pwd.nii.gz": {
        "osf_path": "/file007",
        "size": 1000,
    },
    # Derivatives — filtered by derivative name and subject.
    "derivatives/glm-speed/dataset_description.json": {
        "osf_path": "/file008",
        "size": 100,
    },
    # Subject-level statmap (no session, has acq).
    "derivatives/glm-speed/sub-rat83/sub-rat83_acq-slice32_stat-t_statmap.nii.gz": {
        "osf_path": "/file009",
        "size": 500,
    },
    "derivatives/glm-speed/sub-rat83/sub-rat83_acq-slice42_stat-t_statmap.nii.gz": {
        "osf_path": "/file010",
        "size": 500,
    },
    # Session-level derivative file (has both ses and acq).
    "derivatives/glm-speed/sub-rat83/ses-20220523/fus/sub-rat83_ses-20220523_task-openfield_acq-slice32_dm.tsv": {
        "osf_path": "/file011",
        "size": 200,
    },
    "derivatives/glm-speed/sub-rat84/sub-rat84_stat-t_statmap.nii.gz": {
        "osf_path": "/file012",
        "size": 500,
    },
    "derivatives/decode-speed/sub-rat83/sub-rat83_acq-slice32_desc-accuracy_decode.tsv": {
        "osf_path": "/file013",
        "size": 300,
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
        "confusius.datasets._cybis_pereira_2026.get_index",
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
# fetch_cybis_pereira_2026 — return value and caching
# ---------------------------------------------------------------------------


def test_fetch_returns_bids_root(tmp_path, mock_get_index, mock_retrieve):
    result = fetch_cybis_pereira_2026(data_dir=tmp_path)
    assert result == tmp_path / _BIDS_ROOT
    assert isinstance(result, Path)


def test_fetch_downloads_all_missing_files(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX)


def test_fetch_skips_existing_files(tmp_path, mock_get_index, mock_retrieve):
    # Pre-create two files in the cache.
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in ["dataset_description.json", "participants.tsv"]:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    fetch_cybis_pereira_2026(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX) - 2


def test_fetch_returns_immediately_when_all_cached(tmp_path, mock_get_index):
    # Pre-create every file in the cache.
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in _FAKE_INDEX:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with patch("confusius.datasets._pooch.pooch.retrieve") as mock_retrieve:
        fetch_cybis_pereira_2026(data_dir=tmp_path)
        mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_cybis_pereira_2026 — filters
# ---------------------------------------------------------------------------


def _downloaded_paths(mock_retrieve) -> set[str]:
    """Return the set of file basenames passed to pooch.retrieve."""
    return {c.kwargs["fname"] for c in mock_retrieve.call_args_list}


def test_fetch_dataset_filter_rawdata_only(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path, datasets=["rawdata"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Rawdata files included.
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded
    # Derivatives excluded.
    assert "sub-rat83_acq-slice32_stat-t_statmap.nii.gz" not in downloaded


def test_fetch_dataset_filter_derivative(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path, datasets=["glm-speed"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching derivative included.
    assert "sub-rat83_acq-slice32_stat-t_statmap.nii.gz" in downloaded
    assert "dataset_description.json" in downloaded
    # Non-matching derivative excluded.
    assert "sub-rat83_acq-slice32_desc-accuracy_decode.tsv" not in downloaded
    # Rawdata excluded.
    assert (
        "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" not in downloaded
    )


def test_fetch_subject_filter(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path, subjects=["rat83"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching subject included (rawdata and derivatives).
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    assert "sub-rat83_acq-slice32_stat-t_statmap.nii.gz" in downloaded
    # Non-matching subject excluded.
    assert "sub-rat84_ses-20210407_task-openfield_pwd.nii.gz" not in downloaded
    assert "sub-rat84_stat-t_statmap.nii.gz" not in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded


def test_fetch_session_filter(tmp_path, mock_get_index, mock_retrieve):
    """`sessions` keeps files in matching session dirs and session-less files."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, sessions=["20220523"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching session files included.
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    assert "sub-rat83_ses-20220523_scans.tsv" in downloaded
    # Non-matching sessions excluded.
    assert (
        "sub-rat83_ses-20220524_task-openfield_acq-slice32_pwd.nii.gz" not in downloaded
    )
    assert "sub-rat84_ses-20210407_task-openfield_pwd.nii.gz" not in downloaded
    # Subject-level files (no session entity) pass through.
    assert "sub-rat83_acq-slice32_stat-t_statmap.nii.gz" in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded


def test_fetch_acq_filter(tmp_path, mock_get_index, mock_retrieve):
    """`acqs` keeps matching acquisitions and files with no acq entity."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, acqs=["slice32"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching acquisition included.
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    assert "sub-rat83_acq-slice32_stat-t_statmap.nii.gz" in downloaded
    # Non-matching acquisitions excluded.
    assert (
        "sub-rat83_ses-20220523_task-openfield_acq-slice42_pwd.nii.gz" not in downloaded
    )
    assert "sub-rat83_acq-slice42_stat-t_statmap.nii.gz" not in downloaded
    # Files with no acq entity pass through.
    assert "sub-rat83_ses-20220523_scans.tsv" in downloaded
    assert "sub-rat84_ses-20210407_task-openfield_pwd.nii.gz" in downloaded


def test_fetch_combined_session_and_acq_filters(
    tmp_path, mock_get_index, mock_retrieve
):
    """Session and acquisition filters compose."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, sessions=["20220523"], acqs=["slice42"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Only files matching both filters (or omitting the relevant entity).
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice42_pwd.nii.gz" in downloaded
    assert (
        "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" not in downloaded
    )
    assert (
        "sub-rat83_ses-20220524_task-openfield_acq-slice32_pwd.nii.gz" not in downloaded
    )


def test_fetch_invalid_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_cybis_pereira_2026(data_dir=tmp_path, datasets=["nonexistent"])


def test_fetch_accepts_string_datasets(tmp_path, mock_get_index, mock_retrieve):
    """A single string is accepted and normalized to a list."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, datasets="rawdata")

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    assert "sub-rat83_acq-slice32_stat-t_statmap.nii.gz" not in downloaded


def test_fetch_accepts_string_subjects(tmp_path, mock_get_index, mock_retrieve):
    """A single string is accepted and normalized to a list."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, subjects="rat83")

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    assert "sub-rat84_ses-20210407_task-openfield_pwd.nii.gz" not in downloaded


def test_fetch_accepts_string_sessions_and_acqs(
    tmp_path, mock_get_index, mock_retrieve
):
    """`sessions` and `acqs` strings are normalized to lists."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, sessions="20220523", acqs="slice32")

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz" in downloaded
    assert (
        "sub-rat83_ses-20220523_task-openfield_acq-slice42_pwd.nii.gz" not in downloaded
    )


# ---------------------------------------------------------------------------
# fetch_cybis_pereira_2026 — retry behaviour
# ---------------------------------------------------------------------------


def test_fetch_retries_on_transient_failure(tmp_path, mock_get_index):
    """Transient network errors are retried and eventually succeed."""
    bids_dir = tmp_path / _BIDS_ROOT

    # Pre-create every file except one so only that file goes through retrieve.
    target_rel = (
        "sub-rat83/ses-20220523/fus/"
        "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"
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
        fetch_cybis_pereira_2026(data_dir=tmp_path)

    assert call_count["n"] == _MAX_DOWNLOAD_RETRIES


def test_fetch_raises_after_max_retries(tmp_path, mock_get_index):
    """Persistent network errors propagate after the retry budget is exhausted."""
    bids_dir = tmp_path / _BIDS_ROOT

    target_rel = (
        "sub-rat83/ses-20220523/fus/"
        "sub-rat83_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"
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
            fetch_cybis_pereira_2026(data_dir=tmp_path)

    assert mock_retrieve.call_count == _MAX_DOWNLOAD_RETRIES


# ---------------------------------------------------------------------------
# fetch_cybis_pereira_2026 — refresh behaviour
# ---------------------------------------------------------------------------


def test_fetch_refresh_passes_flag_to_get_index(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_cybis_pereira_2026(data_dir=tmp_path, refresh=True)
    mock_get_index.assert_called_once_with(
        tmp_path / _BIDS_ROOT,
        _OSF_PROJECT_ID,
        _BIDS_ROOT,
        refresh=True,
    )
