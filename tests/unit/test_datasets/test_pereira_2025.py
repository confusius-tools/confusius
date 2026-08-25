"""Unit tests for confusius.datasets._pereira_2025."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from confusius.datasets import fetch_pereira_2025
from confusius.datasets._pereira_2025 import _BIDS_ROOT, _CITATION
from confusius.datasets._utils import plain_citation

_FAKE_INDEX = {
    "dataset_description.json": {"osf_path": "/file001", "size": 100, "md5": None},
    "participants.tsv": {"osf_path": "/file002", "size": 200, "md5": None},
    "task-stim_events.tsv": {"osf_path": "/file003", "size": 50, "md5": None},
    "sub-r11582/sub-r11582_sessions.tsv": {
        "osf_path": "/file004",
        "size": 50,
        "md5": None,
    },
    "sub-r11582/ses-awakebaseline/fusi/sub-r11582_ses-awakebaseline_task-rest_acq-2dfus_run-01_pwd.nii.gz": {
        "osf_path": "/file005",
        "size": 1000,
        "md5": None,
    },
    "sub-r11582/ses-awake1h/fusi/sub-r11582_ses-awake1h_task-rest_acq-2dfus_run-01_pwd.nii.gz": {
        "osf_path": "/file006",
        "size": 1000,
        "md5": None,
    },
    "sub-r21595/ses-awakebaseline/fusi/sub-r21595_ses-awakebaseline_task-rest_acq-2dfus_run-01_pwd.nii.gz": {
        "osf_path": "/file007",
        "size": 1000,
        "md5": None,
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
def mock_get_index():
    """Stub `get_index` so fetch tests do not hit the network."""
    with patch("confusius.datasets._pereira_2025.get_index", return_value=_FAKE_INDEX):
        yield


@pytest.fixture
def mock_retrieve(tmp_path):
    """Patch pooch.retrieve to create stub files instead of downloading."""
    with patch(
        "confusius.datasets._pooch.pooch.retrieve",
        side_effect=_make_retrieve(tmp_path / _BIDS_ROOT),
    ) as mock:
        yield mock


def _downloaded_paths(mock_retrieve, bids_dir: Path) -> set[str]:
    """Return BIDS-relative paths requested from pooch.retrieve."""
    return {
        str((Path(c.kwargs["path"]) / c.kwargs["fname"]).relative_to(bids_dir))
        for c in mock_retrieve.call_args_list
    }


def test_fetch_returns_bids_root(tmp_path, mock_get_index, mock_retrieve):
    result = fetch_pereira_2025(data_dir=tmp_path, print_citation=False)
    assert result == tmp_path / _BIDS_ROOT


def test_fetch_citation_message(tmp_path, mock_get_index, mock_retrieve, capsys):
    fetch_pereira_2025(data_dir=tmp_path)
    out = capsys.readouterr().out
    assert plain_citation(_CITATION) in " ".join(out.split())


def test_fetch_refresh_updates_cached_index(tmp_path, mock_get_index, mock_retrieve):
    with patch("confusius.datasets._pereira_2025.update_cached_index") as mock_update:
        fetch_pereira_2025(data_dir=tmp_path, refresh=True, print_citation=False)

    mock_update.assert_called_once()


def test_fetch_filters_entities(tmp_path, mock_get_index, mock_retrieve):
    fetch_pereira_2025(
        data_dir=tmp_path,
        subjects="r11582",
        sessions="awakebaseline",
        tasks="rest",
        print_citation=False,
    )

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT)
    assert (
        "sub-r11582/ses-awakebaseline/fusi/sub-r11582_ses-awakebaseline_task-rest_acq-2dfus_run-01_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-r11582/ses-awake1h/fusi/sub-r11582_ses-awake1h_task-rest_acq-2dfus_run-01_pwd.nii.gz" not in downloaded
    )
    assert (
        "sub-r21595/ses-awakebaseline/fusi/sub-r21595_ses-awakebaseline_task-rest_acq-2dfus_run-01_pwd.nii.gz"
        not in downloaded
    )
    assert "dataset_description.json" in downloaded
    assert "task-stim_events.tsv" in downloaded


def test_fetch_task_filter(tmp_path, mock_get_index, mock_retrieve):
    fetch_pereira_2025(data_dir=tmp_path, tasks="stim", print_citation=False)
    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT)
    assert (
        "sub-r11582/ses-awakebaseline/fusi/sub-r11582_ses-awakebaseline_task-rest_acq-2dfus_run-01_pwd.nii.gz"
        not in downloaded
    )
    assert "dataset_description.json" in downloaded
