"""Unit tests for confusius.datasets._pepe_mariani_2026_bids."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from confusius.datasets import fetch_pepe_mariani_2026
from confusius.datasets._pepe_mariani_2026_bids import _BIDS_ROOT, _CITATION
from confusius.datasets._utils import plain_citation

_FAKE_INDEX = {
    "dataset_description.json": {"osf_path": "/file001", "size": 100, "md5": None},
    "participants.tsv": {"osf_path": "/file002", "size": 200, "md5": None},
    "sub-m01/sub-m01_sessions.tsv": {"osf_path": "/file003", "size": 50, "md5": None},
    "sub-m01/ses-rest/fusi/sub-m01_ses-rest_task-rest_acq-coronal_chunk-01_pwd.nii.gz": {
        "osf_path": "/file004",
        "size": 1000,
        "md5": None,
    },
    "sub-m01/ses-rest/angio/sub-m01_ses-rest_acq-coronal_pwd.nii.gz": {
        "osf_path": "/file005",
        "size": 1000,
        "md5": None,
    },
    "sub-m02/ses-rest/fusi/sub-m02_ses-rest_task-rest_acq-coronal_chunk-01_pwd.nii.gz": {
        "osf_path": "/file006",
        "size": 1000,
        "md5": None,
    },
    "sub-m01/ses-other/fusi/sub-m01_ses-other_task-rest_acq-coronal_chunk-01_pwd.nii.gz": {
        "osf_path": "/file006a",
        "size": 1000,
        "md5": None,
    },
    "sub-m01/ses-rest/fusi/sub-m01_ses-rest_task-rest_acq-sagittal_chunk-01_pwd.nii.gz": {
        "osf_path": "/file006b",
        "size": 1000,
        "md5": None,
    },
    "derivatives/registered/sub-m01/ses-rest/fusi/sub-m01_ses-rest_task-rest_acq-coronal_pwd.nii.gz": {
        "osf_path": "/file007",
        "size": 1000,
        "md5": None,
    },
    "derivatives/registered/sub-m02/ses-rest/fusi/sub-m02_ses-rest_task-rest_acq-coronal_pwd.nii.gz": {
        "osf_path": "/file007a",
        "size": 1000,
        "md5": None,
    },
    "derivatives/registered/sub-m01/ses-other/fusi/sub-m01_ses-other_task-rest_acq-coronal_pwd.nii.gz": {
        "osf_path": "/file007b",
        "size": 1000,
        "md5": None,
    },
    "derivatives/registered/sub-m01/ses-rest/angio/sub-m01_ses-rest_acq-coronal_pwd.nii.gz": {
        "osf_path": "/file007c",
        "size": 1000,
        "md5": None,
    },
    "derivatives/preprocessed/sub-m01/ses-rest/fusi/sub-m01_ses-rest_task-rest_acq-coronal_pwd.nii.gz": {
        "osf_path": "/file008",
        "size": 1000,
        "md5": None,
    },
    "derivatives/Params/sub-m01/sub-m01_params.tsv": {
        "osf_path": "/file009",
        "size": 100,
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
    with patch(
        "confusius.datasets._pepe_mariani_2026_bids.get_index",
        return_value=_FAKE_INDEX,
    ):
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
    result = fetch_pepe_mariani_2026(data_dir=tmp_path, print_citation=False)
    assert result == tmp_path / _BIDS_ROOT


def test_fetch_citation_message(tmp_path, mock_get_index, mock_retrieve, capsys):
    fetch_pepe_mariani_2026(data_dir=tmp_path)
    out = capsys.readouterr().out
    assert plain_citation(_CITATION) in " ".join(out.split())


def test_fetch_refresh_updates_cached_index(tmp_path, mock_get_index, mock_retrieve):
    with patch(
        "confusius.datasets._pepe_mariani_2026_bids.update_cached_index"
    ) as mock_update:
        fetch_pepe_mariani_2026(data_dir=tmp_path, refresh=True, print_citation=False)

    mock_update.assert_called_once()


def test_fetch_filters_derivatives(tmp_path, mock_get_index, mock_retrieve):
    fetch_pepe_mariani_2026(
        data_dir=tmp_path,
        datasets="registered",
        subjects="m01",
        sessions="rest",
        acqs="coronal",
        datatypes="fusi",
        print_citation=False,
    )

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT)
    assert (
        "derivatives/registered/sub-m01/ses-rest/fusi/"
        "sub-m01_ses-rest_task-rest_acq-coronal_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-m01/ses-rest/fusi/"
        "sub-m01_ses-rest_task-rest_acq-coronal_chunk-01_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "derivatives/registered/sub-m01/ses-rest/angio/"
        "sub-m01_ses-rest_acq-coronal_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "derivatives/registered/sub-m02/ses-rest/fusi/"
        "sub-m02_ses-rest_task-rest_acq-coronal_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "derivatives/registered/sub-m01/ses-other/fusi/"
        "sub-m01_ses-other_task-rest_acq-coronal_pwd.nii.gz"
        not in downloaded
    )
    assert "dataset_description.json" in downloaded


def test_fetch_filters_rawdata(tmp_path, mock_get_index, mock_retrieve):
    fetch_pepe_mariani_2026(
        data_dir=tmp_path,
        datasets=["rawdata"],
        subjects=["m01"],
        sessions=["rest"],
        acqs=["coronal"],
        datatypes=["fusi"],
        print_citation=False,
    )

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT)
    assert (
        "sub-m01/ses-rest/fusi/"
        "sub-m01_ses-rest_task-rest_acq-coronal_chunk-01_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-m01/ses-rest/angio/sub-m01_ses-rest_acq-coronal_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "sub-m01/ses-other/fusi/sub-m01_ses-other_task-rest_acq-coronal_chunk-01_pwd.nii.gz" not in downloaded
    )
    assert (
        "sub-m01/ses-rest/fusi/sub-m01_ses-rest_task-rest_acq-sagittal_chunk-01_pwd.nii.gz" not in downloaded
    )
    assert (
        "sub-m02/ses-rest/fusi/sub-m02_ses-rest_task-rest_acq-coronal_chunk-01_pwd.nii.gz" not in downloaded
    )


def test_fetch_rejects_unknown_dataset(tmp_path, mock_get_index, mock_retrieve):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_pepe_mariani_2026(data_dir=tmp_path, datasets="glm")


def test_fetch_rejects_unknown_datatype(tmp_path, mock_get_index, mock_retrieve):
    with pytest.raises(ValueError, match="Unknown datatype"):
        fetch_pepe_mariani_2026(data_dir=tmp_path, datatypes="motion")
