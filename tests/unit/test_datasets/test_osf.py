"""Unit tests for confusius.datasets._osf (shared OSF fetch helpers)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from confusius.datasets._osf import (
    _INDEX_FILENAME,
    download_missing_osf_files,
    get_index,
    resolve_index_url,
)

_FAKE_PROJECT = "testproj"
_FAKE_BIDS_ROOT = "fake-bids"
_FAKE_INDEX = {"a/b/c.nii.gz": "/file001", "top.json": "/file002"}


def _make_osf_responses(
    index_data: dict,
    *,
    project_id: str = _FAKE_PROJECT,
    bids_root: str = _FAKE_BIDS_ROOT,
) -> list[MagicMock]:
    """Build mock `requests.get` responses for the full OSF resolution chain.

    Three sequential calls are mocked:
    1. OSF storage root listing (finds the BIDS folder).
    2. BIDS folder listing (finds dataset_index.json).
    3. dataset_index.json download.
    """
    folder_href = f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/folder/"
    index_download_url = f"https://files.osf.io/v1/resources/{project_id}/index"

    root_resp = MagicMock()
    root_resp.raise_for_status.return_value = None
    root_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": bids_root},
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
                "attributes": {"name": _INDEX_FILENAME},
                "links": {"download": index_download_url},
            }
        ]
    }

    index_resp = MagicMock()
    index_resp.raise_for_status.return_value = None
    index_resp.json.return_value = index_data

    return [root_resp, folder_resp, index_resp]


# ---------------------------------------------------------------------------
# get_index
# ---------------------------------------------------------------------------


def test_get_index_downloads_and_caches(tmp_path):
    """When no cache exists, the index is fetched from OSF and written to disk."""
    responses = _make_osf_responses(_FAKE_INDEX)
    with patch("confusius.datasets._osf.requests.get", side_effect=responses):
        result = get_index(tmp_path, _FAKE_PROJECT, _FAKE_BIDS_ROOT)

    assert result == _FAKE_INDEX
    index_path = tmp_path / _INDEX_FILENAME
    assert index_path.exists()
    assert json.loads(index_path.read_text()) == _FAKE_INDEX


def test_get_index_uses_cache_without_network(tmp_path):
    """With a warm cache and refresh=False, no HTTP calls are made."""
    (tmp_path / _INDEX_FILENAME).write_text(json.dumps(_FAKE_INDEX))

    with patch("confusius.datasets._osf.requests.get") as mock_requests:
        result = get_index(tmp_path, _FAKE_PROJECT, _FAKE_BIDS_ROOT)

    assert result == _FAKE_INDEX
    mock_requests.assert_not_called()


def test_get_index_refreshes_when_requested(tmp_path):
    """refresh=True re-fetches the index even when a cached copy exists."""
    (tmp_path / _INDEX_FILENAME).write_text(json.dumps({"stale": "data"}))

    responses = _make_osf_responses(_FAKE_INDEX)
    with patch("confusius.datasets._osf.requests.get", side_effect=responses):
        result = get_index(tmp_path, _FAKE_PROJECT, _FAKE_BIDS_ROOT, refresh=True)

    assert result == _FAKE_INDEX
    cached = json.loads((tmp_path / _INDEX_FILENAME).read_text())
    assert cached == _FAKE_INDEX


# ---------------------------------------------------------------------------
# resolve_index_url
# ---------------------------------------------------------------------------


def test_resolve_index_url_raises_if_bids_folder_not_on_osf():
    """RuntimeError propagates when the BIDS root folder is missing on OSF."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"attributes": {"name": "other-folder"}}]}

    with patch("confusius.datasets._osf.requests.get", return_value=resp):
        with pytest.raises(RuntimeError, match=_FAKE_BIDS_ROOT):
            resolve_index_url(_FAKE_PROJECT, _FAKE_BIDS_ROOT)


def test_resolve_index_url_raises_if_index_file_not_on_osf():
    """RuntimeError propagates when dataset_index.json is absent from OSF."""
    folder_href = (
        f"https://api.osf.io/v2/nodes/{_FAKE_PROJECT}/files/osfstorage/folder/"
    )

    root_resp = MagicMock()
    root_resp.raise_for_status.return_value = None
    root_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": _FAKE_BIDS_ROOT},
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
        "confusius.datasets._osf.requests.get",
        side_effect=[root_resp, folder_resp],
    ):
        with pytest.raises(RuntimeError, match=_INDEX_FILENAME):
            resolve_index_url(_FAKE_PROJECT, _FAKE_BIDS_ROOT)


# ---------------------------------------------------------------------------
# download_missing_osf_files — md5-aware refresh
# ---------------------------------------------------------------------------


def _recording_retrieve():
    """Return a (side_effect, calls) pair recording downloads and stubbing files.

    The side effect mimics `pooch.retrieve`: it creates the destination file so
    subsequent existence checks pass, and records each call's keyword arguments.
    """
    calls: list[dict] = []

    def _retrieve(url, known_hash, fname, path, progressbar):
        calls.append({"url": url, "known_hash": known_hash, "fname": fname})
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
        return str(dest)

    return _retrieve, calls


def _write(path: Path, content: bytes) -> str:
    """Write `content` to `path` and return its MD5 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.md5(content).hexdigest()


def test_download_skips_cached_file_without_refresh(tmp_path):
    """A cached file is not re-downloaded when refresh is False, even if stale."""
    dest = tmp_path / "a.nii.gz"
    _write(dest, b"local")
    files = {"a.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "deadbeef"}}

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_missing_osf_files(tmp_path, files, refresh=False)

    assert calls == []


def test_refresh_redownloads_on_md5_mismatch(tmp_path):
    """refresh re-downloads a cached file whose local MD5 differs from the index."""
    dest = tmp_path / "a.nii.gz"
    _write(dest, b"old-content")
    remote_md5 = hashlib.md5(b"new-content").hexdigest()
    files = {"a.nii.gz": {"osf_path": "/f1", "size": 11, "md5": remote_md5}}

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_missing_osf_files(tmp_path, files, refresh=True)

    assert [c["fname"] for c in calls] == ["a.nii.gz"]
    assert calls[0]["known_hash"] == f"md5:{remote_md5}"


def test_refresh_skips_when_md5_matches(tmp_path):
    """refresh leaves a cached file untouched when its MD5 matches the index."""
    dest = tmp_path / "a.nii.gz"
    local_md5 = _write(dest, b"same-content")
    files = {"a.nii.gz": {"osf_path": "/f1", "size": 12, "md5": local_md5}}

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_missing_osf_files(tmp_path, files, refresh=True)

    assert calls == []


def test_refresh_without_md5_warns_and_keeps_cached(tmp_path):
    """refresh cannot check an index entry lacking md5: keep the file and warn."""
    dest = tmp_path / "a.nii.gz"
    _write(dest, b"local")
    files = {"a.nii.gz": {"osf_path": "/f1", "size": 5}}

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        with pytest.warns(UserWarning, match="no md5"):
            download_missing_osf_files(tmp_path, files, refresh=True)

    assert calls == []


def test_missing_file_downloads_with_known_hash(tmp_path):
    """A missing file is downloaded and its index md5 is forwarded to pooch."""
    files = {
        "with_md5.nii.gz": {"osf_path": "/f1", "size": 3, "md5": "abc123"},
        "no_md5.nii.gz": {"osf_path": "/f2", "size": 3},
    }

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_missing_osf_files(tmp_path, files, refresh=False)

    by_name = {c["fname"]: c["known_hash"] for c in calls}
    assert by_name == {"with_md5.nii.gz": "md5:abc123", "no_md5.nii.gz": None}
