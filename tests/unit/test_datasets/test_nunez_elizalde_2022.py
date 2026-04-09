"""Unit tests for confusius.datasets."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz": "/file009",
    # Second session — session-filtered.
    "sub-CR020/ses-20191121/angio/sub-CR020_ses-20191121_pwd.nii.gz": "/file007",
    "sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_task-kalatsky_acq-slice01_pwd.nii.gz": "/file008",
    # Derivatives — should also be subject/session-filtered.
    "derivatives/allenccf_align/dataset_description.json": "/file010",
    "derivatives/allenccf_align/structure_tree_safe_2017.csv": "/file011",
    "derivatives/allenccf_align/sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz": "/file012",
    "derivatives/allenccf_align/sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_space-fusi_desc-allenccf_dseg.nii.gz": "/file013",
    "derivatives/allenccf_align/sub-OTHER/ses-20191122/fusi/sub-OTHER_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz": "/file014",
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


def test_fetch_refresh_passes_flag_to_get_index(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, refresh=True)
    mock_get_index.assert_called_once_with(tmp_path / _BIDS_ROOT, refresh=True)


# ---------------------------------------------------------------------------
# get_datasets_dir — default (pooch.os_cache) branch
# ---------------------------------------------------------------------------


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
# _get_index and _resolve_index_url — network paths
# ---------------------------------------------------------------------------


def _make_osf_responses(index_data: dict) -> list[MagicMock]:
    """Build mock requests.get responses for the full OSF resolution chain.

    Three sequential calls are mocked:
    1. OSF storage root listing (finds the BIDS folder)
    2. BIDS folder listing (finds dataset_index.json)
    3. dataset_index.json download
    """
    folder_href = "https://api.osf.io/v2/nodes/43skw/files/osfstorage/folder/"
    index_download_url = "https://files.osf.io/v1/resources/43skw/index"

    root_resp = MagicMock()
    root_resp.raise_for_status.return_value = None
    root_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": "nunez-elizalde-2022-bids"},
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
        "confusius.datasets._nunez_elizalde_2022.requests.get", side_effect=responses
    ):
        fetch_nunez_elizalde_2022(data_dir=tmp_path)

    index_path = tmp_path / _BIDS_ROOT / _INDEX_FILENAME
    assert index_path.exists()
    assert json.loads(index_path.read_text()) == _FAKE_INDEX


def test_fetch_uses_cached_index_without_network(tmp_path, mock_retrieve):
    """With a warm cache and refresh=False, no HTTP calls are made."""
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    (bids_dir / _INDEX_FILENAME).write_text(json.dumps(_FAKE_INDEX))

    with patch("confusius.datasets._nunez_elizalde_2022.requests.get") as mock_requests:
        fetch_nunez_elizalde_2022(data_dir=tmp_path)

    mock_requests.assert_not_called()


def test_fetch_refreshes_index_when_requested(tmp_path, mock_retrieve):
    """refresh=True re-fetches the index even when a cached copy exists."""
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    (bids_dir / _INDEX_FILENAME).write_text(json.dumps({"stale": "data"}))

    updated_index = {**_FAKE_INDEX}
    responses = _make_osf_responses(updated_index)
    with patch(
        "confusius.datasets._nunez_elizalde_2022.requests.get", side_effect=responses
    ):
        fetch_nunez_elizalde_2022(data_dir=tmp_path, refresh=True)

    cached = json.loads((bids_dir / _INDEX_FILENAME).read_text())
    assert cached == updated_index


def test_fetch_raises_if_bids_folder_not_on_osf(tmp_path):
    """RuntimeError propagates when the BIDS root folder is missing on OSF."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"attributes": {"name": "other-folder"}}]}

    with patch(
        "confusius.datasets._nunez_elizalde_2022.requests.get", return_value=resp
    ):
        with pytest.raises(RuntimeError, match="nunez-elizalde-2022-bids"):
            fetch_nunez_elizalde_2022(data_dir=tmp_path)


def test_fetch_raises_if_index_file_not_on_osf(tmp_path):
    """RuntimeError propagates when dataset_index.json is absent from OSF."""
    folder_href = "https://api.osf.io/v2/nodes/43skw/files/osfstorage/folder/"

    root_resp = MagicMock()
    root_resp.raise_for_status.return_value = None
    root_resp.json.return_value = {
        "data": [
            {
                "attributes": {"name": "nunez-elizalde-2022-bids"},
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
        "confusius.datasets._nunez_elizalde_2022.requests.get",
        side_effect=[root_resp, folder_resp],
    ):
        with pytest.raises(RuntimeError, match="dataset_index.json"):
            fetch_nunez_elizalde_2022(data_dir=tmp_path)
