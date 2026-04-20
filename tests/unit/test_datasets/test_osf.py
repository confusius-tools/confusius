"""Unit tests for confusius.datasets._osf (shared OSF fetch helpers)."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from confusius.datasets._osf import (
    _INDEX_FILENAME,
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
    with patch(
        "confusius.datasets._osf.requests.get", side_effect=responses
    ):
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
    with patch(
        "confusius.datasets._osf.requests.get", side_effect=responses
    ):
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
    folder_href = f"https://api.osf.io/v2/nodes/{_FAKE_PROJECT}/files/osfstorage/folder/"

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


def test_resolve_index_url_appends_missing_index_hint():
    """The optional hint is appended verbatim to the error message."""
    folder_href = f"https://api.osf.io/v2/nodes/{_FAKE_PROJECT}/files/osfstorage/folder/"

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

    hint = "Run 'some-tool --index-only' to regenerate it."

    with patch(
        "confusius.datasets._osf.requests.get",
        side_effect=[root_resp, folder_resp],
    ):
        with pytest.raises(RuntimeError, match=re.escape(hint)):
            resolve_index_url(
                _FAKE_PROJECT, _FAKE_BIDS_ROOT, missing_index_hint=hint
            )
