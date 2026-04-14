"""Unit tests for confusius.datasets._cybis_pereira_2026."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from confusius.datasets import fetch_cybis_pereira_2026
from confusius.datasets._cybis_pereira_2026 import (
    _BIDS_ROOT,
    _INDEX_FILENAME,
    _MAX_DOWNLOAD_RETRIES,
)

# Minimal fake index representing the different file categories in the dataset.
_FAKE_INDEX = {
    # Top-level BIDS metadata — always included.
    "dataset_description.json": {"osf_path": "/file001", "size": 100},
    "participants.tsv": {"osf_path": "/file002", "size": 200},
    # Rawdata — requires "rawdata" in datasets filter.
    "sub-rat83/fusi/sub-rat83_task-navigation_pwd.nii.gz": {
        "osf_path": "/file003",
        "size": 1000,
    },
    "sub-rat84/fusi/sub-rat84_task-navigation_pwd.nii.gz": {
        "osf_path": "/file004",
        "size": 1000,
    },
    # Derivatives — filtered by derivative name and subject.
    "derivatives/glm-speed/dataset_description.json": {
        "osf_path": "/file005",
        "size": 100,
    },
    "derivatives/glm-speed/sub-rat83/sub-rat83_stat-t_statmap.nii.gz": {
        "osf_path": "/file006",
        "size": 500,
    },
    "derivatives/glm-speed/sub-rat84/sub-rat84_stat-t_statmap.nii.gz": {
        "osf_path": "/file007",
        "size": 500,
    },
    "derivatives/decode-speed/sub-rat83/sub-rat83_desc-accuracy_decode.tsv": {
        "osf_path": "/file008",
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
    """Patch _get_index to return the fake index without network access."""
    with patch(
        "confusius.datasets._cybis_pereira_2026._get_index",
        return_value=_FAKE_INDEX,
    ) as mock:
        yield mock


@pytest.fixture
def mock_retrieve(tmp_path):
    """Patch pooch.retrieve to create stub files instead of downloading."""
    bids_dir = tmp_path / _BIDS_ROOT
    with patch(
        "confusius.datasets._cybis_pereira_2026.pooch.retrieve",
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

    with patch(
        "confusius.datasets._cybis_pereira_2026.pooch.retrieve"
    ) as mock_retrieve:
        fetch_cybis_pereira_2026(data_dir=tmp_path)
        mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_cybis_pereira_2026 — filters
# ---------------------------------------------------------------------------


def test_fetch_dataset_filter_rawdata_only(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path, datasets=["rawdata"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    # Rawdata files included.
    assert "sub-rat83_task-navigation_pwd.nii.gz" in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded
    # Derivatives excluded.
    assert "sub-rat83_stat-t_statmap.nii.gz" not in downloaded


def test_fetch_dataset_filter_derivative(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path, datasets=["glm-speed"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    # Matching derivative included.
    assert "sub-rat83_stat-t_statmap.nii.gz" in downloaded
    assert "dataset_description.json" in downloaded
    # Non-matching derivative excluded.
    assert "sub-rat83_desc-accuracy_decode.tsv" not in downloaded
    # Rawdata excluded.
    assert "sub-rat83_task-navigation_pwd.nii.gz" not in downloaded


def test_fetch_subject_filter(tmp_path, mock_get_index, mock_retrieve):
    fetch_cybis_pereira_2026(data_dir=tmp_path, subjects=["rat83"])

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    # Matching subject included (rawdata and derivatives).
    assert "sub-rat83_task-navigation_pwd.nii.gz" in downloaded
    assert "sub-rat83_stat-t_statmap.nii.gz" in downloaded
    # Non-matching subject excluded.
    assert "sub-rat84_task-navigation_pwd.nii.gz" not in downloaded
    assert "sub-rat84_stat-t_statmap.nii.gz" not in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded


def test_fetch_invalid_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_cybis_pereira_2026(data_dir=tmp_path, datasets=["nonexistent"])


def test_fetch_accepts_string_datasets(tmp_path, mock_get_index, mock_retrieve):
    """A single string is accepted and normalized to a list."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, datasets="rawdata")

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-rat83_task-navigation_pwd.nii.gz" in downloaded
    assert "sub-rat83_stat-t_statmap.nii.gz" not in downloaded


def test_fetch_accepts_string_subjects(tmp_path, mock_get_index, mock_retrieve):
    """A single string is accepted and normalized to a list."""
    fetch_cybis_pereira_2026(data_dir=tmp_path, subjects="rat83")

    downloaded = {c.kwargs["fname"] for c in mock_retrieve.call_args_list}
    assert "sub-rat83_task-navigation_pwd.nii.gz" in downloaded
    assert "sub-rat84_task-navigation_pwd.nii.gz" not in downloaded


# ---------------------------------------------------------------------------
# fetch_cybis_pereira_2026 — retry behaviour
# ---------------------------------------------------------------------------


def test_fetch_retries_on_transient_failure(tmp_path, mock_get_index):
    """Transient network errors are retried and eventually succeed."""
    bids_dir = tmp_path / _BIDS_ROOT

    # Pre-create every file except one so only that file goes through retrieve.
    target_rel = "sub-rat83/fusi/sub-rat83_task-navigation_pwd.nii.gz"
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
            "confusius.datasets._cybis_pereira_2026.pooch.retrieve",
            side_effect=flaky_retrieve,
        ),
        patch("confusius.datasets._cybis_pereira_2026.time.sleep"),
    ):
        fetch_cybis_pereira_2026(data_dir=tmp_path)

    assert call_count["n"] == _MAX_DOWNLOAD_RETRIES


def test_fetch_raises_after_max_retries(tmp_path, mock_get_index):
    """Persistent network errors propagate after the retry budget is exhausted."""
    bids_dir = tmp_path / _BIDS_ROOT

    target_rel = "sub-rat83/fusi/sub-rat83_task-navigation_pwd.nii.gz"
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
            "confusius.datasets._cybis_pereira_2026.pooch.retrieve",
            side_effect=always_fails,
        ) as mock_retrieve,
        patch("confusius.datasets._cybis_pereira_2026.time.sleep"),
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
    mock_get_index.assert_called_once_with(tmp_path / _BIDS_ROOT, refresh=True)


# ---------------------------------------------------------------------------
# _get_index and _resolve_index_url — network paths
# ---------------------------------------------------------------------------


def _make_osf_responses(index_data: dict) -> list[MagicMock]:
    """Build mock requests.get responses for the full OSF resolution chain.

    Three sequential calls are mocked:
    1. OSF storage root listing (finds the BIDS folder)
    2. BIDS folder listing (finds dataset_index.json)
    3. dataset_index.json download
    """
    folder_href = "https://api.osf.io/v2/nodes/2v6f7/files/osfstorage/folder/"
    index_download_url = "https://files.osf.io/v1/resources/2v6f7/index"

    root_resp = MagicMock()
    root_resp.raise_for_status.return_value = None
    root_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": "cybis-pereira-2026-bids"},
                "relationships": {
                    "files": {"links": {"related": {"href": folder_href}}}
                },
            }
        ]
    }

    folder_resp = MagicMock()
    folder_resp.raise_for_status.return_value = None
    folder_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": "dataset_index.json"},
                "links": {"download": index_download_url},
            }
        ]
    }

    index_resp = MagicMock()
    index_resp.raise_for_status.return_value = None
    index_resp.json.return_value = index_data

    return [root_resp, folder_resp, index_resp]


def test_fetch_downloads_and_caches_index(tmp_path, mock_retrieve):
    """When no cache exists, index is fetched from OSF and written to disk."""
    responses = _make_osf_responses(_FAKE_INDEX)
    with patch(
        "confusius.datasets._cybis_pereira_2026.requests.get", side_effect=responses
    ):
        fetch_cybis_pereira_2026(data_dir=tmp_path)

    index_path = tmp_path / _BIDS_ROOT / _INDEX_FILENAME
    assert index_path.exists()
    assert json.loads(index_path.read_text()) == _FAKE_INDEX


def test_fetch_uses_cached_index_without_network(tmp_path, mock_retrieve):
    """With a warm cache and refresh=False, no HTTP calls are made."""
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    (bids_dir / _INDEX_FILENAME).write_text(json.dumps(_FAKE_INDEX))

    with patch(
        "confusius.datasets._cybis_pereira_2026.requests.get"
    ) as mock_requests:
        fetch_cybis_pereira_2026(data_dir=tmp_path)

    mock_requests.assert_not_called()


def test_fetch_refreshes_index_when_requested(tmp_path, mock_retrieve):
    """refresh=True re-fetches the index even when a cached copy exists."""
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    (bids_dir / _INDEX_FILENAME).write_text(json.dumps({"stale": "data"}))

    updated_index = {**_FAKE_INDEX}
    responses = _make_osf_responses(updated_index)
    with patch(
        "confusius.datasets._cybis_pereira_2026.requests.get", side_effect=responses
    ):
        fetch_cybis_pereira_2026(data_dir=tmp_path, refresh=True)

    cached = json.loads((bids_dir / _INDEX_FILENAME).read_text())
    assert cached == updated_index


def test_fetch_raises_if_bids_folder_not_on_osf(tmp_path):
    """RuntimeError propagates when the BIDS root folder is missing on OSF."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"attributes": {"name": "other-folder"}}]}

    with patch(
        "confusius.datasets._cybis_pereira_2026.requests.get", return_value=resp
    ):
        with pytest.raises(RuntimeError, match="cybis-pereira-2026-bids"):
            fetch_cybis_pereira_2026(data_dir=tmp_path)


def test_fetch_raises_if_index_file_not_on_osf(tmp_path):
    """RuntimeError propagates when dataset_index.json is absent from OSF."""
    folder_href = "https://api.osf.io/v2/nodes/2v6f7/files/osfstorage/folder/"

    root_resp = MagicMock()
    root_resp.raise_for_status.return_value = None
    root_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": "cybis-pereira-2026-bids"},
                "relationships": {
                    "files": {"links": {"related": {"href": folder_href}}}
                },
            }
        ]
    }

    folder_resp = MagicMock()
    folder_resp.raise_for_status.return_value = None
    folder_resp.json.return_value = {"data": []}

    with patch(
        "confusius.datasets._cybis_pereira_2026.requests.get",
        side_effect=[root_resp, folder_resp],
    ):
        with pytest.raises(RuntimeError, match="dataset_index.json"):
            fetch_cybis_pereira_2026(data_dir=tmp_path)
