"""Unit tests for confusius.datasets._landemard_2026."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from confusius.datasets import fetch_landemard_2026
from confusius.datasets._landemard_2026 import _BIDS_ROOT, _OSF_PROJECT_ID
from confusius.datasets._pooch import _MAX_DOWNLOAD_RETRIES

# Minimal fake index representing the different file categories in the dataset.
_FAKE_INDEX = {
    # Top-level BIDS metadata — always included.
    "dataset_description.json": {"osf_path": "/file001", "size": 100},
    "participants.tsv": {"osf_path": "/file002", "size": 200},
    "README.txt": {"osf_path": "/file002b", "size": 50},
    # Rawdata — fusi and angio, two acquisitions (ref04 and ref11_run-1) and
    # a subject-level scans.tsv with no datatype.
    "sub-ALD001/angio/sub-ALD001_pwd.json": {"osf_path": "/file003", "size": 50},
    "sub-ALD001/angio/sub-ALD001_pwd.nii.gz": {"osf_path": "/file004", "size": 800},
    "sub-ALD001/fusi/sub-ALD001_task-awake_acq-ref04_pwd.nii.gz": {
        "osf_path": "/file005",
        "size": 1000,
    },
    "sub-ALD001/fusi/sub-ALD001_task-awake_acq-ref04_pwd.json": {
        "osf_path": "/file006",
        "size": 50,
    },
    "sub-ALD001/fusi/sub-ALD001_task-awake_acq-ref11_run-1_pwd.nii.gz": {
        "osf_path": "/file007",
        "size": 1000,
    },
    "sub-ALD001/fusi/sub-ALD001_task-awake_acq-ref11_run-1_recording-wheel_physio.tsv.gz": {
        "osf_path": "/file008",
        "size": 200,
    },
    # Fusi file with no `acq-` entity (mirrors the cybis fetcher's fixture).
    "sub-ALD001/fusi/sub-ALD001_task-awake_pwd.nii.gz": {
        "osf_path": "/file008b",
        "size": 1000,
    },
    "sub-ALD001/sub-ALD001_scans.tsv": {"osf_path": "/file009", "size": 50},
    # Second subject with one acquisition.
    "sub-ALD002/fusi/sub-ALD002_task-awake_acq-ref04_pwd.nii.gz": {
        "osf_path": "/file010",
        "size": 1000,
    },
    "sub-ALD002/fusi/sub-ALD002_task-awake_acq-ref04_pwd.json": {
        "osf_path": "/file011",
        "size": 50,
    },
    "sub-ALD002/sub-ALD002_scans.tsv": {"osf_path": "/file012", "size": 50},
    # Derivatives — atlas_mapping has per-subject files, processed_data is
    # dataset-level with subject IDs embedded in the filename.
    "derivatives/atlas_mapping/dataset_description.json": {
        "osf_path": "/file013",
        "size": 100,
    },
    "derivatives/atlas_mapping/sub-ALD001/Atlas_alignment_ALD001.npz": {
        "osf_path": "/file014",
        "size": 500,
    },
    "derivatives/atlas_mapping/sub-ALD002/Atlas_alignment_ALD002.npz": {
        "osf_path": "/file015",
        "size": 500,
    },
    "derivatives/processed_data/Compact_data_regions_ALD001.npz": {
        "osf_path": "/file016",
        "size": 500,
    },
    "derivatives/processed_data/Compact_data_regions_ALD002.npz": {
        "osf_path": "/file017",
        "size": 500,
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
        "confusius.datasets._landemard_2026.get_index",
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
# fetch_landemard_2026 — return value and caching
# ---------------------------------------------------------------------------


def test_fetch_returns_bids_root(tmp_path, mock_get_index, mock_retrieve):
    result = fetch_landemard_2026(data_dir=tmp_path)
    assert result == tmp_path / _BIDS_ROOT
    assert isinstance(result, Path)


def test_fetch_downloads_all_missing_files(tmp_path, mock_get_index, mock_retrieve):
    fetch_landemard_2026(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX)


def test_fetch_skips_existing_files(tmp_path, mock_get_index, mock_retrieve):
    # Pre-create two files in the cache.
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in ["dataset_description.json", "participants.tsv"]:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    fetch_landemard_2026(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX) - 2


def test_fetch_returns_immediately_when_all_cached(tmp_path, mock_get_index):
    # Pre-create every file in the cache.
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in _FAKE_INDEX:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with patch("confusius.datasets._pooch.pooch.retrieve") as mock_retrieve:
        fetch_landemard_2026(data_dir=tmp_path)
        mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_landemard_2026 — filters
# ---------------------------------------------------------------------------


def _downloaded_paths(mock_retrieve) -> set[str]:
    """Return the set of file basenames passed to pooch.retrieve."""
    return {c.kwargs["fname"] for c in mock_retrieve.call_args_list}


def test_fetch_dataset_filter_rawdata_only(tmp_path, mock_get_index, mock_retrieve):
    fetch_landemard_2026(data_dir=tmp_path, datasets=["rawdata"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Rawdata files included.
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded
    # Derivatives excluded.
    assert "Atlas_alignment_ALD001.npz" not in downloaded
    assert "Compact_data_regions_ALD001.npz" not in downloaded


def test_fetch_dataset_filter_atlas_mapping(tmp_path, mock_get_index, mock_retrieve):
    fetch_landemard_2026(data_dir=tmp_path, datasets=["atlas_mapping"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching derivative included.
    assert "Atlas_alignment_ALD001.npz" in downloaded
    assert "Atlas_alignment_ALD002.npz" in downloaded
    assert "dataset_description.json" in downloaded
    # Non-matching derivative excluded.
    assert "Compact_data_regions_ALD001.npz" not in downloaded
    # Rawdata excluded.
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" not in downloaded


def test_fetch_dataset_filter_processed_data(tmp_path, mock_get_index, mock_retrieve):
    fetch_landemard_2026(data_dir=tmp_path, datasets=["processed_data"])

    downloaded = _downloaded_paths(mock_retrieve)
    assert "Compact_data_regions_ALD001.npz" in downloaded
    assert "Compact_data_regions_ALD002.npz" in downloaded
    assert "Atlas_alignment_ALD001.npz" not in downloaded
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" not in downloaded


def test_fetch_dataset_filter_combined(tmp_path, mock_get_index, mock_retrieve):
    """`datasets` may combine rawdata and a derivative name in a single call."""
    fetch_landemard_2026(data_dir=tmp_path, datasets=["rawdata", "atlas_mapping"])

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    assert "Atlas_alignment_ALD001.npz" in downloaded
    # Non-listed derivative still excluded.
    assert "Compact_data_regions_ALD001.npz" not in downloaded


def test_fetch_subject_filter(tmp_path, mock_get_index, mock_retrieve):
    fetch_landemard_2026(data_dir=tmp_path, subjects=["ALD001"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching subject included (rawdata and derivatives).
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    assert "Atlas_alignment_ALD001.npz" in downloaded
    # Non-matching subject excluded.
    assert "sub-ALD002_task-awake_acq-ref04_pwd.nii.gz" not in downloaded
    assert "Atlas_alignment_ALD002.npz" not in downloaded
    # Dataset-level processed_data files pass through (no sub-* directory).
    assert "Compact_data_regions_ALD001.npz" in downloaded
    assert "Compact_data_regions_ALD002.npz" in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded


def test_fetch_acq_filter(tmp_path, mock_get_index, mock_retrieve):
    """`acqs` keeps matching acquisitions and files with no acq entity."""
    fetch_landemard_2026(data_dir=tmp_path, acqs=["ref04"])

    downloaded = _downloaded_paths(mock_retrieve)
    # Matching acquisition included.
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    # Non-matching acquisition excluded (ref11 prefix not requested).
    assert "sub-ALD001_task-awake_acq-ref11_run-1_pwd.nii.gz" not in downloaded
    # Files with no acq entity pass through.
    assert "sub-ALD001_scans.tsv" in downloaded
    assert "sub-ALD001_pwd.nii.gz" in downloaded
    assert "sub-ALD001_task-awake_pwd.nii.gz" in downloaded
    assert "Atlas_alignment_ALD001.npz" in downloaded


def test_fetch_acq_filter_includes_sidecars(tmp_path, mock_get_index, mock_retrieve):
    """`acqs` keeps all sidecar files (e.g. physio) for the matching acquisition."""
    fetch_landemard_2026(data_dir=tmp_path, acqs=["ref11"])

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_task-awake_acq-ref11_run-1_pwd.nii.gz" in downloaded
    assert (
        "sub-ALD001_task-awake_acq-ref11_run-1_recording-wheel_physio.tsv.gz"
        in downloaded
    )
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" not in downloaded


def test_fetch_datatype_filter_fusi_only(tmp_path, mock_get_index, mock_retrieve):
    """`datatypes=["fusi"]` excludes angio files but keeps files without a datatype."""
    fetch_landemard_2026(data_dir=tmp_path, datatypes=["fusi"])

    downloaded = _downloaded_paths(mock_retrieve)
    # fusi files included.
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    # angio files excluded.
    assert "sub-ALD001_pwd.nii.gz" not in downloaded
    assert "sub-ALD001_pwd.json" not in downloaded
    # Files with no datatype layer pass through.
    assert "sub-ALD001_scans.tsv" in downloaded
    assert "Atlas_alignment_ALD001.npz" in downloaded
    # Top-level metadata always included.
    assert "dataset_description.json" in downloaded


def test_fetch_datatype_filter_angio_only(tmp_path, mock_get_index, mock_retrieve):
    """`datatypes=["angio"]` keeps angio files and excludes fusi files."""
    fetch_landemard_2026(data_dir=tmp_path, datatypes=["angio"])

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_pwd.nii.gz" in downloaded
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" not in downloaded
    # Files with no datatype layer still pass through.
    assert "sub-ALD001_scans.tsv" in downloaded


def test_fetch_combined_subject_and_acq_filters(
    tmp_path, mock_get_index, mock_retrieve
):
    """Subject and acquisition filters compose."""
    fetch_landemard_2026(
        data_dir=tmp_path, subjects=["ALD001"], acqs=["ref04"], datatypes=["fusi"]
    )

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    assert "sub-ALD001_task-awake_acq-ref11_run-1_pwd.nii.gz" not in downloaded
    assert "sub-ALD002_task-awake_acq-ref04_pwd.nii.gz" not in downloaded
    assert "Atlas_alignment_ALD001.npz" in downloaded


def test_fetch_invalid_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_landemard_2026(data_dir=tmp_path, datasets=["nonexistent"])


def test_fetch_invalid_datatype_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown datatype"):
        fetch_landemard_2026(data_dir=tmp_path, datatypes=["nonexistent"])


def test_fetch_accepts_string_datasets(tmp_path, mock_get_index, mock_retrieve):
    """A single string is accepted and normalized to a list."""
    fetch_landemard_2026(data_dir=tmp_path, datasets="rawdata")

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    assert "Atlas_alignment_ALD001.npz" not in downloaded


def test_fetch_accepts_string_subjects(tmp_path, mock_get_index, mock_retrieve):
    """A single string is accepted and normalized to a list."""
    fetch_landemard_2026(data_dir=tmp_path, subjects="ALD001")

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    assert "sub-ALD002_task-awake_acq-ref04_pwd.nii.gz" not in downloaded


def test_fetch_accepts_string_acqs_and_datatypes(
    tmp_path, mock_get_index, mock_retrieve
):
    """`acqs` and `datatypes` strings are normalized to lists."""
    fetch_landemard_2026(data_dir=tmp_path, acqs="ref04", datatypes="fusi")

    downloaded = _downloaded_paths(mock_retrieve)
    assert "sub-ALD001_task-awake_acq-ref04_pwd.nii.gz" in downloaded
    assert "sub-ALD001_task-awake_acq-ref11_run-1_pwd.nii.gz" not in downloaded
    assert "sub-ALD001_pwd.nii.gz" not in downloaded


# ---------------------------------------------------------------------------
# fetch_landemard_2026 — retry behaviour
# ---------------------------------------------------------------------------


def test_fetch_retries_on_transient_failure(tmp_path, mock_get_index):
    """Transient network errors are retried and eventually succeed."""
    bids_dir = tmp_path / _BIDS_ROOT

    # Pre-create every file except one so only that file goes through retrieve.
    target_rel = "sub-ALD001/fusi/sub-ALD001_task-awake_acq-ref04_pwd.nii.gz"
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
        fetch_landemard_2026(data_dir=tmp_path)

    assert call_count["n"] == _MAX_DOWNLOAD_RETRIES


def test_fetch_raises_after_max_retries(tmp_path, mock_get_index):
    """Persistent network errors propagate after the retry budget is exhausted."""
    bids_dir = tmp_path / _BIDS_ROOT

    target_rel = "sub-ALD001/fusi/sub-ALD001_task-awake_acq-ref04_pwd.nii.gz"
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
            fetch_landemard_2026(data_dir=tmp_path)

    assert mock_retrieve.call_count == _MAX_DOWNLOAD_RETRIES


# ---------------------------------------------------------------------------
# fetch_landemard_2026 — refresh behaviour
# ---------------------------------------------------------------------------


def test_fetch_refresh_passes_flag_to_get_index(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_landemard_2026(data_dir=tmp_path, refresh=True)
    mock_get_index.assert_called_once_with(
        tmp_path / _BIDS_ROOT,
        _OSF_PROJECT_ID,
        _BIDS_ROOT,
        refresh=True,
    )
