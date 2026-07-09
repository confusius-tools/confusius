"""Unit tests for confusius.datasets._osf (shared OSF fetch helpers)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from confusius.datasets._osf import (
    _INDEX_FILENAME,
    OsfFileInfo,
    download_osf_files,
    get_index,
    read_cached_index,
    resolve_index_url,
    update_cached_index,
)

_FAKE_PROJECT = "testproj"
_FAKE_BIDS_ROOT = "fake-bids"
_FAKE_INDEX: dict[str, OsfFileInfo] = {
    "a/b/c.nii.gz": {"osf_path": "/file001", "size": 10, "md5": "aaa"},
    "top.json": {"osf_path": "/file002", "size": 20, "md5": "bbb"},
}


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


def test_get_index_refreshes_without_persisting(tmp_path):
    """refresh=True re-fetches the remote index but leaves the cache for the caller.

    Persistence of a refreshed index is deferred to `update_cached_index`, which
    merges rather than replaces, so `get_index` must not overwrite the cache here.
    """
    (tmp_path / _INDEX_FILENAME).write_text(json.dumps({"stale": "data"}))

    responses = _make_osf_responses(_FAKE_INDEX)
    with patch("confusius.datasets._osf.requests.get", side_effect=responses):
        result = get_index(tmp_path, _FAKE_PROJECT, _FAKE_BIDS_ROOT, refresh=True)

    assert result == _FAKE_INDEX
    cached = json.loads((tmp_path / _INDEX_FILENAME).read_text())
    assert cached == {"stale": "data"}


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
# read_cached_index
# ---------------------------------------------------------------------------


def test_read_cached_index_returns_empty_when_absent(tmp_path):
    """No cached index on disk yields an empty mapping, not an error."""
    assert read_cached_index(tmp_path) == {}


def test_read_cached_index_decodes_existing(tmp_path):
    """An existing cached index is decoded from disk without network access."""
    (tmp_path / _INDEX_FILENAME).write_text(json.dumps(_FAKE_INDEX))
    assert read_cached_index(tmp_path) == _FAKE_INDEX


def test_read_cached_index_rejects_outdated_structure(tmp_path):
    """An index missing the current keys errors, naming the directory to delete."""
    (tmp_path / _INDEX_FILENAME).write_text(
        json.dumps({"a.nii.gz": {"osf_path": "/f1", "size": 5}})  # no md5 key
    )

    with pytest.raises(RuntimeError) as excinfo:
        read_cached_index(tmp_path)

    message = str(excinfo.value)
    assert str(tmp_path) in message
    assert "md5" in message


# ---------------------------------------------------------------------------
# download_osf_files — md5-aware, index-vs-index refresh
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


def test_download_skips_cached_file_without_refresh(tmp_path):
    """A cached file is not re-downloaded when refresh is False, even if stale."""
    (tmp_path / "a.nii.gz").touch()
    files: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "new"}
    }
    previous: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "old"}
    }

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_osf_files(tmp_path, files, previous, refresh=False)

    assert calls == []


def test_refresh_redownloads_on_md5_change(tmp_path):
    """refresh re-downloads a cached file whose remote md5 differs from the cache."""
    (tmp_path / "a.nii.gz").touch()
    files: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 11, "md5": "new"}
    }
    previous: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 11, "md5": "old"}
    }

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_osf_files(tmp_path, files, previous, refresh=True)

    assert [c["fname"] for c in calls] == ["a.nii.gz"]
    assert calls[0]["known_hash"] == "md5:new"


def test_refresh_skips_when_md5_unchanged(tmp_path):
    """refresh leaves a cached file untouched when the cached and remote md5 match."""
    (tmp_path / "a.nii.gz").touch()
    files: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 12, "md5": "same"}
    }
    previous: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 12, "md5": "same"}
    }

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_osf_files(tmp_path, files, previous, refresh=True)

    assert calls == []


def test_refresh_redownloads_when_cached_md5_null(tmp_path):
    """A cached file whose entry has a null md5 is re-downloaded, not trusted."""
    (tmp_path / "a.nii.gz").touch()
    files: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "new"}
    }
    previous: dict[str, OsfFileInfo] = {
        "a.nii.gz": {"osf_path": "/f1", "size": 5, "md5": None}
    }

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_osf_files(tmp_path, files, previous, refresh=True)

    assert [c["fname"] for c in calls] == ["a.nii.gz"]
    assert calls[0]["known_hash"] == "md5:new"


def test_missing_file_downloads_with_known_hash(tmp_path):
    """A missing file is downloaded and its index md5 is forwarded to pooch."""
    files: dict[str, OsfFileInfo] = {
        "with_md5.nii.gz": {"osf_path": "/f1", "size": 3, "md5": "abc123"},
        "no_md5.nii.gz": {"osf_path": "/f2", "size": 3, "md5": None},
    }

    retrieve, calls = _recording_retrieve()
    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_osf_files(tmp_path, files, refresh=False)

    by_name = {c["fname"]: c["known_hash"] for c in calls}
    assert by_name == {"with_md5.nii.gz": "md5:abc123", "no_md5.nii.gz": None}


def test_download_progress_callback_reports_cumulative_bytes(tmp_path):
    """GUI progress callbacks receive cumulative byte counts across files."""
    files = {
        "a.nii.gz": {"osf_path": "/f1", "size": 6, "md5": "aaa"},
        "b.nii.gz": {"osf_path": "/f2", "size": 4, "md5": "bbb"},
    }
    updates: list[tuple[int, int, str]] = []

    def progress_callback(current: int, total: int, description: str) -> None:
        updates.append((current, total, description))

    def retrieve(url, known_hash, fname, path, progressbar):
        total = files[fname]["size"]
        progressbar.total = total
        progressbar.update(total // 2)
        progressbar.update(total - (total // 2))
        progressbar.reset()
        progressbar.close()
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
        return str(dest)

    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        download_osf_files(
            tmp_path,
            files,
            refresh=False,
            progress_callback=progress_callback,
        )

    assert updates[0] == (0, 10, "Preparing download...")
    assert updates[-1] == (10, 10, "Download complete.")
    assert (6, 10, "Downloading a.nii.gz") in updates
    assert (10, 10, "Downloading b.nii.gz") in updates


# ---------------------------------------------------------------------------
# update_cached_index — merge, don't replace
# ---------------------------------------------------------------------------


def test_update_cached_index_merges_without_replacing(tmp_path):
    """Requested files adopt the remote entry; other files keep their cached one."""
    remote_index: dict[str, OsfFileInfo] = {
        "requested.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "new"},
        "unrequested.nii.gz": {"osf_path": "/f2", "size": 6, "md5": "remote-changed"},
        "brand_new.nii.gz": {"osf_path": "/f3", "size": 7, "md5": "fresh"},
    }
    previous_index: dict[str, OsfFileInfo] = {
        "requested.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "old"},
        "unrequested.nii.gz": {"osf_path": "/f2", "size": 6, "md5": "local-baseline"},
    }
    files: dict[str, OsfFileInfo] = {
        "requested.nii.gz": remote_index["requested.nii.gz"]
    }

    update_cached_index(tmp_path, remote_index, previous_index, files)

    written = json.loads((tmp_path / _INDEX_FILENAME).read_text())
    assert written == {
        # Requested file was reconciled with disk, so it takes the remote md5.
        "requested.nii.gz": {"osf_path": "/f1", "size": 5, "md5": "new"},
        # Untouched file keeps its cached baseline instead of the remote md5,
        # so a later refresh can still detect it changed upstream.
        "unrequested.nii.gz": {"osf_path": "/f2", "size": 6, "md5": "local-baseline"},
        # A file never seen locally is catalogued from the remote index.
        "brand_new.nii.gz": {"osf_path": "/f3", "size": 7, "md5": "fresh"},
    }
