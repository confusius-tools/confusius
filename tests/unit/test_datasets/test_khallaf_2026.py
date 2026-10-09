"""Unit tests for confusius.datasets._khallaf_2026."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from confusius.datasets import fetch_khallaf_2026
from confusius.datasets._khallaf_2026 import _BIDS_ROOT, _CITATION
from confusius.datasets._s3 import S3FileInfo
from confusius.datasets._utils import plain_citation

# Minimal fake index covering every bucket and reconstruction variant. Keys are
# release-relative file paths.
_FILE_SIZES = {
    # Top-level metadata — always included.
    "dataset_description.json": {"size": 100},
    "participants.tsv": {"size": 200},
    "README.md": {"size": 300},
    # Rawdata — sub-5622 / ses-IPM, raw and resampled, runs 1 and 2.
    "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii": {
        "size": 1000
    },
    "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.json": {
        "size": 50
    },
    "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-2_pwd.nii": {
        "size": 1000
    },
    "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_rec-resampled_run-1_space-5622run1_pwd.nii": {
        "size": 1100
    },
    "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_rec-resampled_run-1_space-5622run1_pwd.json": {
        "size": 60
    },
    "sub-5622/ses-IPM/sub-5622_ses-IPM_scans.tsv": {"size": 40},
    # Rawdata — sub-6036 / ses-Air, raw run-1.
    "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii": {
        "size": 1000
    },
    # Derivatives — glm (subject-level and group-level without a subject).
    "derivatives/glm/sub-5622/ses-IPM/sub-5622_ses-IPM_task-olfactory_desc-mean_pwd.nii": {
        "size": 700
    },
    "derivatives/glm/desc-replicate_pwd.nii": {"size": 500},
    # Derivatives — bootstrapping (folder file and loose zip).
    "derivatives/bootstrapping/sub-5622_bootstrap.nii": {"size": 600},
    "derivatives/bootstrapping.zip": {"size": 5000},
    # Sourcedata — Iconeus raw acquisitions (gated by the sourcedata flag).
    "sourcedata/IPM/IPM_5622_4Dscan_5.source.scan": {"size": 9000},
    "sourcedata/IPM/IPM_6036_4Dscan_7.source.bps": {"size": 9000},
}

_FAKE_INDEX: dict[str, S3FileInfo] = {
    relative: {
        "url": (
            "https://confusius-datasets.s3.us-west-2.amazonaws.com"
            f"/datasets/{_BIDS_ROOT}/1.0.0/{relative}"
        ),
        "size": info["size"],
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    for relative, info in _FILE_SIZES.items()
}


_SOURCEDATA_KEYS = {k for k in _FAKE_INDEX if k.startswith("sourcedata/")}


def _selected(opened: list[str]) -> set[str]:
    """Return prefix-stripped paths of members that were extracted."""
    return set(opened)


@pytest.fixture
def mock_get_index():
    """Stub `get_index` so fetch tests don't read the remote archive."""
    with patch(
        "confusius.datasets._khallaf_2026.get_index",
        return_value=dict(_FAKE_INDEX),
    ) as mock:
        yield mock


@pytest.fixture
def opened_members(tmp_path):
    """Record files downloaded by Pooch and create matching empty cache files."""
    opened = []

    def retrieve(url, known_hash, fname, path, progressbar):
        dest = Path(path) / fname
        opened.append(dest.relative_to(tmp_path / _BIDS_ROOT / "1.0.0").as_posix())
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()
        return str(dest)

    with patch("confusius.datasets._pooch.pooch.retrieve", side_effect=retrieve):
        yield opened


# ---------------------------------------------------------------------------
# fetch_khallaf_2026 — return value and caching
# ---------------------------------------------------------------------------


def test_fetch_returns_bids_root(tmp_path, mock_get_index, opened_members):
    result = fetch_khallaf_2026(data_dir=tmp_path)
    assert result == tmp_path / _BIDS_ROOT / "1.0.0"
    assert isinstance(result, Path)


def test_fetch_citation_message(tmp_path, mock_get_index, opened_members, capsys):
    fetch_khallaf_2026(data_dir=tmp_path)
    out = capsys.readouterr().out
    assert (
        "If you use this dataset in your work, please cite the following source:" in out
    )
    # Rich word-wraps the printed citation, so compare on whitespace-normalized text.
    assert plain_citation(_CITATION) in " ".join(out.split())

    fetch_khallaf_2026(data_dir=tmp_path, print_citation=False)
    assert capsys.readouterr().out == ""


def test_fetch_downloads_all_except_sourcedata_by_default(
    tmp_path, mock_get_index, opened_members
):
    fetch_khallaf_2026(data_dir=tmp_path)
    assert _selected(opened_members) == set(_FAKE_INDEX) - _SOURCEDATA_KEYS


def test_fetch_skips_existing_files(tmp_path, mock_get_index, opened_members):
    bids_dir = tmp_path / _BIDS_ROOT / "1.0.0"
    for rel in ["dataset_description.json", "participants.tsv"]:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    fetch_khallaf_2026(data_dir=tmp_path)
    selected = _selected(opened_members)
    assert "dataset_description.json" not in selected
    assert "participants.tsv" not in selected
    assert "README.md" in selected


def test_fetch_returns_immediately_when_all_cached(tmp_path, mock_get_index):
    bids_dir = tmp_path / _BIDS_ROOT / "1.0.0"
    for rel in set(_FAKE_INDEX) - _SOURCEDATA_KEYS:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with patch("confusius.datasets._pooch.pooch.retrieve") as mock_zip:
        fetch_khallaf_2026(data_dir=tmp_path)
        mock_zip.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_khallaf_2026 — filters
# ---------------------------------------------------------------------------


def test_fetch_dataset_filter_rawdata_only(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, datasets="rawdata")
    selected = _selected(opened_members)
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        in selected
    )
    # Metadata always included; derivatives and sourcedata excluded.
    assert "dataset_description.json" in selected
    assert "derivatives/glm/desc-replicate_pwd.nii" not in selected
    assert not (selected & _SOURCEDATA_KEYS)


def test_fetch_dataset_filter_bootstrapping(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, datasets="bootstrapping")
    selected = _selected(opened_members)
    # Both the folder file and the loose bootstrapping.zip map to bootstrapping.
    assert "derivatives/bootstrapping/sub-5622_bootstrap.nii" in selected
    assert "derivatives/bootstrapping.zip" in selected
    # The glm derivative and rawdata are excluded.
    assert "derivatives/glm/desc-replicate_pwd.nii" not in selected
    assert (
        "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii"
        not in selected
    )


def test_fetch_subject_filter(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, subjects="5622")
    selected = _selected(opened_members)
    # sub-5622 rawdata and derivatives included.
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        in selected
    )
    assert (
        "derivatives/glm/sub-5622/ses-IPM/sub-5622_ses-IPM_task-olfactory_desc-mean_pwd.nii"
        in selected
    )
    # sub-6036 excluded.
    assert (
        "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii"
        not in selected
    )
    # Group-level derivative (no subject) and metadata pass through.
    assert "derivatives/glm/desc-replicate_pwd.nii" in selected
    assert "dataset_description.json" in selected


def test_fetch_session_filter(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, sessions="Air")
    selected = _selected(opened_members)
    assert (
        "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii"
        in selected
    )
    # ses-IPM files excluded.
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        not in selected
    )
    assert "sub-5622/ses-IPM/sub-5622_ses-IPM_scans.tsv" not in selected
    # Files with no session entity pass through.
    assert "derivatives/glm/desc-replicate_pwd.nii" in selected


def test_fetch_run_filter(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, runs="2")
    selected = _selected(opened_members)
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-2_pwd.nii"
        in selected
    )
    # run-1 files excluded.
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        not in selected
    )
    # Files with no run entity pass through.
    assert "sub-5622/ses-IPM/sub-5622_ses-IPM_scans.tsv" in selected
    assert "dataset_description.json" in selected


def test_fetch_reconstruction_raw(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, reconstruction="raw")
    selected = _selected(opened_members)
    # Raw fusi volumes kept, resampled excluded.
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        in selected
    )
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_rec-resampled_run-1_space-5622run1_pwd.nii"
        not in selected
    )
    # Non-fusi rawdata files unaffected.
    assert "sub-5622/ses-IPM/sub-5622_ses-IPM_scans.tsv" in selected


def test_fetch_reconstruction_resampled(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, reconstruction="resampled")
    selected = _selected(opened_members)
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_rec-resampled_run-1_space-5622run1_pwd.nii"
        in selected
    )
    # Raw fusi volumes excluded.
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        not in selected
    )
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-2_pwd.nii"
        not in selected
    )
    # Derivatives are unaffected by the reconstruction filter.
    assert "derivatives/glm/desc-replicate_pwd.nii" in selected


def test_fetch_sourcedata_flag_downloads_unfiltered(
    tmp_path, mock_get_index, opened_members
):
    # Subject filter must not restrict sourcedata when the flag is set.
    fetch_khallaf_2026(data_dir=tmp_path, subjects="5622", sourcedata=True)
    selected = _selected(opened_members)
    assert _SOURCEDATA_KEYS <= selected


def test_fetch_combined_subject_and_reconstruction(
    tmp_path, mock_get_index, opened_members
):
    fetch_khallaf_2026(data_dir=tmp_path, subjects="5622", reconstruction="resampled")
    selected = _selected(opened_members)
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_rec-resampled_run-1_space-5622run1_pwd.nii"
        in selected
    )
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        not in selected
    )
    assert (
        "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii"
        not in selected
    )


def test_fetch_accepts_list_filters(tmp_path, mock_get_index, opened_members):
    fetch_khallaf_2026(data_dir=tmp_path, subjects=["5622", "6036"], runs=["1"])
    selected = _selected(opened_members)
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        in selected
    )
    assert (
        "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii"
        in selected
    )
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-2_pwd.nii"
        not in selected
    )


def test_fetch_invalid_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_khallaf_2026(data_dir=tmp_path, datasets="nonexistent")


def test_fetch_invalid_reconstruction_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown reconstruction"):
        fetch_khallaf_2026(data_dir=tmp_path, reconstruction="nonexistent")


def test_fetch_coerces_int_filters(tmp_path, mock_get_index, opened_members):
    """Integer subject/run IDs are coerced to str, not silently dropped."""
    fetch_khallaf_2026(data_dir=tmp_path, subjects=5622, runs=[1])
    selected = _selected(opened_members)
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-1_pwd.nii"
        in selected
    )
    # run-2 excluded by the run filter; sub-6036 excluded by the subject filter.
    assert (
        "sub-5622/ses-IPM/fusi/sub-5622_ses-IPM_task-olfactory_run-2_pwd.nii"
        not in selected
    )
    assert (
        "sub-6036/ses-Air/fusi/sub-6036_ses-Air_task-olfactory_run-1_pwd.nii"
        not in selected
    )
