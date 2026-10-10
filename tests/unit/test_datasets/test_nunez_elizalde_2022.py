"""Unit tests for confusius.datasets._nunez_elizalde_2022."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from confusius.datasets import fetch_nunez_elizalde_2022, get_datasets_dir
from confusius.datasets._nunez_elizalde_2022 import (
    _BIDS_ROOT,
    _CITATION,
)
from confusius.datasets._utils import plain_citation

# Minimal fake index representing the different file categories in the dataset.
_FAKE_INDEX = {
    # Top-level BIDS metadata — always included.
    "dataset_description.json": {"url": "/file001", "size": 100},
    "participants.tsv": {"url": "/file002", "size": 200},
    # Subject-level file — always included when subject passes the filter.
    "sub-CR020/sub-CR020_sessions.tsv": {"url": "/file003", "size": 300},
    # Angio — always included regardless of task filter.
    "sub-CR020/ses-20191122/susi/sub-CR020_ses-20191122_pwd.nii.gz": {
        "url": "/file004",
        "size": 400,
    },
    # fUSI — task-filtered.
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz": {
        "url": "/file005",
        "size": 500,
    },
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz": {
        "url": "/file006",
        "size": 600,
    },
    "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz": {
        "url": "/file009",
        "size": 900,
    },
    # Second session — session-filtered.
    "sub-CR020/ses-20191121/susi/sub-CR020_ses-20191121_pwd.nii.gz": {
        "url": "/file007",
        "size": 700,
    },
    "sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_task-kalatsky_acq-slice01_pwd.nii.gz": {
        "url": "/file008",
        "size": 800,
    },
    # Derivatives — should also be subject/session-filtered.
    "derivatives/allenccf_align/dataset_description.json": {
        "url": "/file010",
        "size": 1000,
    },
    "derivatives/allenccf_align/structure_tree_safe_2017.csv": {
        "url": "/file011",
        "size": 1100,
    },
    "derivatives/allenccf_align/sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz": {
        "url": "/file012",
        "size": 1200,
    },
    "derivatives/allenccf_align/sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_space-fusi_desc-allenccf_dseg.nii.gz": {
        "url": "/file013",
        "size": 1300,
    },
    "derivatives/allenccf_align/sub-OTHER/ses-20191122/fusi/sub-OTHER_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz": {
        "url": "/file014",
        "size": 1400,
    },
    "sourcedata/allenccf_align/sub-CR022/ses-20201011/fusi/2020-10-11_CR022_estimated_probe00_3Dtrack_manual.hdf": {
        "url": "/file015",
        "size": 1500,
    },
    "sourcedata/allenccf_align/sub-OTHER/ses-20201011/fusi/2020-10-11_OTHER_estimated_probe00_3Dtrack_manual.hdf": {
        "url": "/file016",
        "size": 1600,
    },
}


for _relative, _info in _FAKE_INDEX.items():
    _info["url"] = (
        "https://confusius-datasets.s3.us-west-2.amazonaws.com"
        f"/datasets/{_BIDS_ROOT}/1.0.0/{_relative}"
    )
    _info["sha256"] = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _make_retrieve(bids_dir: Path):
    """Return a pooch.retrieve side-effect that creates stub files on disk."""

    def _retrieve(url, known_hash, fname, path, progressbar, downloader):
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
    bids_dir = tmp_path / _BIDS_ROOT / "1.0.0"
    with patch(
        "confusius.datasets._pooch.pooch.retrieve",
        side_effect=_make_retrieve(bids_dir),
    ) as mock:
        yield mock


def _downloaded_paths(mock_retrieve, bids_dir: Path) -> set[str]:
    """Return BIDS-relative paths requested from pooch.retrieve."""
    return {
        (Path(c.kwargs["path"]) / c.kwargs["fname"]).relative_to(bids_dir).as_posix()
        for c in mock_retrieve.call_args_list
    }


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
    assert result == tmp_path / _BIDS_ROOT / "1.0.0"
    assert isinstance(result, Path)


def test_fetch_citation_message(tmp_path, mock_get_index, mock_retrieve, capsys):
    fetch_nunez_elizalde_2022(data_dir=tmp_path)
    out = capsys.readouterr().out
    assert (
        "If you use this dataset in your work, please cite the following source:" in out
    )
    # Rich word-wraps the printed citation, so compare on whitespace-normalized text.
    assert plain_citation(_CITATION) in " ".join(out.split())

    fetch_nunez_elizalde_2022(data_dir=tmp_path, print_citation=False)
    assert capsys.readouterr().out == ""


def test_fetch_downloads_all_missing_files(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX)


def test_fetch_skips_existing_files(tmp_path, mock_get_index, mock_retrieve):
    # Pre-create two files in the cache.
    bids_dir = tmp_path / _BIDS_ROOT / "1.0.0"
    for rel in ["dataset_description.json", "participants.tsv"]:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    fetch_nunez_elizalde_2022(data_dir=tmp_path)
    assert mock_retrieve.call_count == len(_FAKE_INDEX) - 2


def test_fetch_returns_immediately_when_all_cached(tmp_path, mock_get_index):
    # Pre-create every file in the cache.
    bids_dir = tmp_path / _BIDS_ROOT / "1.0.0"
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


def test_fetch_task_filter_excludes_non_matching_fusi_includes_susi(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, tasks=["kalatsky"])

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz"
        not in downloaded
    )
    # Angio is always included regardless of task filter.
    assert "sub-CR020/ses-20191122/susi/sub-CR020_ses-20191122_pwd.nii.gz" in downloaded


def test_fetch_session_filter(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, sessions=["20191122"])

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert "sub-CR020/ses-20191122/susi/sub-CR020_ses-20191122_pwd.nii.gz" in downloaded
    assert (
        "sub-CR020/ses-20191121/susi/sub-CR020_ses-20191121_pwd.nii.gz"
        not in downloaded
    )


def test_fetch_filters_derivatives_by_subject_and_session(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        subjects=["CR020"],
        sessions=["20191122"],
    )

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    # Shared derivative files (no subject/session folder) should still be included.
    assert "dataset_description.json" in downloaded
    assert "derivatives/allenccf_align/structure_tree_safe_2017.csv" in downloaded
    # Matching derivative subject/session is included.
    assert (
        "derivatives/allenccf_align/sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz"
        in downloaded
    )
    # Non-matching derivative subject/session are excluded.
    assert (
        "derivatives/allenccf_align/sub-CR020/ses-20191121/fusi/sub-CR020_ses-20191121_space-fusi_desc-allenccf_dseg.nii.gz"
        not in downloaded
    )
    assert (
        "derivatives/allenccf_align/sub-OTHER/ses-20191122/fusi/sub-OTHER_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz"
        not in downloaded
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"tasks": "spontaneous"},
        {"acqs": "slice03"},
        {"tasks": "spontaneous", "acqs": "slice03"},
    ],
)
def test_fetch_entity_filters_keep_unlabelled_derivatives(
    tmp_path, mock_get_index, mock_retrieve, filters
):
    fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        subjects="CR020",
        sessions="20191122",
        **filters,
    )
    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "derivatives/allenccf_align/sub-CR020/ses-20191122/fusi/"
        "sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz"
        in downloaded
    )
    assert (
        "derivatives/allenccf_align/sub-CR020/ses-20191121/fusi/"
        "sub-CR020_ses-20191121_space-fusi_desc-allenccf_dseg.nii.gz"
        not in downloaded
    )
    assert (
        "derivatives/allenccf_align/sub-OTHER/ses-20191122/fusi/"
        "sub-OTHER_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz"
        not in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/fusi/"
        "sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/fusi/"
        "sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz"
        not in downloaded
    )


def test_fetch_acq_filter_excludes_non_matching_fusi_includes_susi(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, acqs=["slice03"])

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz"
        not in downloaded
    )
    # Angio is always included regardless of acquisition filter.
    assert "sub-CR020/ses-20191122/susi/sub-CR020_ses-20191122_pwd.nii.gz" in downloaded


def test_fetch_rawdata_dataset_excludes_derivatives_and_sourcedata(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, datasets=["rawdata"])

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz"
        in downloaded
    )
    assert (
        "derivatives/allenccf_align/sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_space-fusi_desc-allenccf_dseg.nii.gz"
        not in downloaded
    )
    assert "derivatives/allenccf_align/structure_tree_safe_2017.csv" not in downloaded
    assert (
        "sourcedata/allenccf_align/sub-CR022/ses-20201011/fusi/2020-10-11_CR022_estimated_probe00_3Dtrack_manual.hdf"
        not in downloaded
    )


def test_fetch_allenccf_align_dataset_includes_matching_sourcedata(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        datasets=["allenccf_align"],
        subjects=["CR022"],
        sessions=["20201011"],
        datatypes=["fusi"],
    )

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "sourcedata/allenccf_align/sub-CR022/ses-20201011/fusi/2020-10-11_CR022_estimated_probe00_3Dtrack_manual.hdf"
        in downloaded
    )
    assert (
        "sourcedata/allenccf_align/sub-OTHER/ses-20201011/fusi/2020-10-11_OTHER_estimated_probe00_3Dtrack_manual.hdf"
        not in downloaded
    )


def test_fetch_fusi_datatype_excludes_susi(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, datatypes=["fusi"])

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice03_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/susi/sub-CR020_ses-20191122_pwd.nii.gz"
        not in downloaded
    )


def test_fetch_subject_filter(tmp_path, mock_retrieve):
    index_with_two_subjects = {
        **_FAKE_INDEX,
        "sub-OTHER/sub-OTHER_sessions.tsv": {"url": "/file999", "size": 999},
        "sub-OTHER/ses-20191122/susi/sub-OTHER_ses-20191122_pwd.nii.gz": {
            "url": "/file998",
            "size": 998,
        },
    }
    with patch(
        "confusius.datasets._nunez_elizalde_2022.get_index",
        return_value=index_with_two_subjects,
    ):
        fetch_nunez_elizalde_2022(data_dir=tmp_path, subjects=["CR020"])

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert "sub-OTHER/sub-OTHER_sessions.tsv" not in downloaded
    assert (
        "sub-OTHER/ses-20191122/susi/sub-OTHER_ses-20191122_pwd.nii.gz"
        not in downloaded
    )


def test_fetch_accepts_string_filters(tmp_path, mock_get_index, mock_retrieve):
    fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        datasets="rawdata",
        subjects="CR020",
        sessions="20191122",
        tasks="kalatsky",
        acqs="slice01",
        datatypes="fusi",
    )

    downloaded = _downloaded_paths(mock_retrieve, tmp_path / _BIDS_ROOT / "1.0.0")
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-kalatsky_acq-slice01_pwd.nii.gz"
        in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/fusi/sub-CR020_ses-20191122_task-spontaneous_acq-slice01_pwd.nii.gz"
        not in downloaded
    )
    assert (
        "sub-CR020/ses-20191122/susi/sub-CR020_ses-20191122_pwd.nii.gz"
        not in downloaded
    )


def test_fetch_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_nunez_elizalde_2022(data_dir=tmp_path, datasets=["not-a-dataset"])


def test_fetch_rejects_unknown_datatype(tmp_path):
    with pytest.raises(ValueError, match="Unknown datatype"):
        fetch_nunez_elizalde_2022(data_dir=tmp_path, datatypes=["not-a-datatype"])


# ---------------------------------------------------------------------------
# fetch_nunez_elizalde_2022 — refresh behaviour
# ---------------------------------------------------------------------------


def test_fetch_refresh_passes_flag_to_get_index(
    tmp_path, mock_get_index, mock_retrieve
):
    fetch_nunez_elizalde_2022(data_dir=tmp_path, refresh=True)
    mock_get_index.assert_called_once_with(
        tmp_path / _BIDS_ROOT,
        "datasets",
        _BIDS_ROOT,
        refresh=True,
    )
