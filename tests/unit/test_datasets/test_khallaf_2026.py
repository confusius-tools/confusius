"""Unit tests for confusius.datasets._khallaf_2026."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import ANY, patch

import pytest
import requests
import urllib3.exceptions

from confusius.datasets import fetch_khallaf_2026
from confusius.datasets._dataverse import (
    access_url,
    download_zip_members,
    get_zip_index,
    update_cached_zip_index,
)
from confusius.datasets._khallaf_2026 import _BIDS_ROOT, _DATAFILE_ID, _ZIP_ROOT
from confusius.datasets._pooch import _MAX_DOWNLOAD_RETRIES

# Minimal fake index covering every bucket and reconstruction variant. Keys are
# prefix-stripped member paths (the form get_zip_index returns).
_FAKE_INDEX = {
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

_SOURCEDATA_KEYS = {k for k in _FAKE_INDEX if k.startswith("sourcedata/")}


def _selected(opened: list[str]) -> set[str]:
    """Return prefix-stripped paths of members that were extracted."""
    return {member[len(_ZIP_ROOT) :] for member in opened}


@pytest.fixture
def mock_get_zip_index():
    """Stub `get_zip_index` so fetch tests don't read the remote archive."""
    with patch(
        "confusius.datasets._khallaf_2026.get_zip_index",
        return_value=dict(_FAKE_INDEX),
    ) as mock:
        yield mock


@pytest.fixture
def opened_members():
    """Patch `RemoteZip` with a fake that records and serves member reads."""
    opened: list[str] = []
    contents = {_ZIP_ROOT + rel: b"data" for rel in _FAKE_INDEX}

    class _FakeRemoteZip:
        def __init__(self, url, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            opened.append(member)
            return io.BytesIO(contents[member])

    with patch("confusius.datasets._dataverse.RemoteZip", _FakeRemoteZip):
        yield opened


# ---------------------------------------------------------------------------
# fetch_khallaf_2026 — return value and caching
# ---------------------------------------------------------------------------


def test_fetch_returns_bids_root(tmp_path, mock_get_zip_index, opened_members):
    result = fetch_khallaf_2026(data_dir=tmp_path)
    assert result == tmp_path / _BIDS_ROOT
    assert isinstance(result, Path)


def test_fetch_downloads_all_except_sourcedata_by_default(
    tmp_path, mock_get_zip_index, opened_members
):
    fetch_khallaf_2026(data_dir=tmp_path)
    assert _selected(opened_members) == set(_FAKE_INDEX) - _SOURCEDATA_KEYS


def test_fetch_skips_existing_files(tmp_path, mock_get_zip_index, opened_members):
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in ["dataset_description.json", "participants.tsv"]:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    fetch_khallaf_2026(data_dir=tmp_path)
    selected = _selected(opened_members)
    assert "dataset_description.json" not in selected
    assert "participants.tsv" not in selected
    assert "README.md" in selected


def test_fetch_returns_immediately_when_all_cached(tmp_path, mock_get_zip_index):
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in set(_FAKE_INDEX) - _SOURCEDATA_KEYS:
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()

    with patch("confusius.datasets._dataverse.RemoteZip") as mock_zip:
        fetch_khallaf_2026(data_dir=tmp_path)
        mock_zip.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_khallaf_2026 — filters
# ---------------------------------------------------------------------------


def test_fetch_dataset_filter_rawdata_only(
    tmp_path, mock_get_zip_index, opened_members
):
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


def test_fetch_dataset_filter_bootstrapping(
    tmp_path, mock_get_zip_index, opened_members
):
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


def test_fetch_subject_filter(tmp_path, mock_get_zip_index, opened_members):
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


def test_fetch_session_filter(tmp_path, mock_get_zip_index, opened_members):
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


def test_fetch_run_filter(tmp_path, mock_get_zip_index, opened_members):
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


def test_fetch_reconstruction_raw(tmp_path, mock_get_zip_index, opened_members):
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


def test_fetch_reconstruction_resampled(tmp_path, mock_get_zip_index, opened_members):
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
    tmp_path, mock_get_zip_index, opened_members
):
    # Subject filter must not restrict sourcedata when the flag is set.
    fetch_khallaf_2026(data_dir=tmp_path, subjects="5622", sourcedata=True)
    selected = _selected(opened_members)
    assert _SOURCEDATA_KEYS <= selected


def test_fetch_combined_subject_and_reconstruction(
    tmp_path, mock_get_zip_index, opened_members
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


def test_fetch_accepts_list_filters(tmp_path, mock_get_zip_index, opened_members):
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


def test_fetch_coerces_int_filters(tmp_path, mock_get_zip_index, opened_members):
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


# ---------------------------------------------------------------------------
# get_zip_index — real index builder and cache
# ---------------------------------------------------------------------------


class _FakeZipInfo:
    """Minimal stand-in for zipfile.ZipInfo used by get_zip_index."""

    def __init__(self, filename, file_size, crc=0):
        self.filename = filename
        self.file_size = file_size
        self.CRC = crc

    def is_dir(self):
        return self.filename.endswith("/")


def _fake_remotezip_from_infos(infos):
    """Return a fake RemoteZip class whose infolist() yields `infos`."""

    class _Fake:
        def __init__(self, url, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def infolist(self):
            return infos

    return _Fake


def test_get_zip_index_builds_strips_prefix_and_caches(tmp_path):
    infos = [
        _FakeZipInfo(_ZIP_ROOT, 0),  # directory entry, skipped
        _FakeZipInfo(_ZIP_ROOT + "sub-5622/ses-IPM/", 0),  # directory, skipped
        _FakeZipInfo(_ZIP_ROOT + "sub-5622/ses-IPM/x_pwd.nii", 100, crc=42),
        _FakeZipInfo("outside_root/other.txt", 50),  # not under prefix, skipped
    ]
    with patch(
        "confusius.datasets._dataverse.RemoteZip", _fake_remotezip_from_infos(infos)
    ):
        index = get_zip_index(tmp_path, "url", _ZIP_ROOT)

    assert index == {"sub-5622/ses-IPM/x_pwd.nii": {"size": 100, "crc": 42}}
    assert (tmp_path / "dataverse_zip_index.json").exists()

    # Second call (no refresh) reads the cache without touching the network.
    with patch("confusius.datasets._dataverse.RemoteZip") as mock_zip:
        cached = get_zip_index(tmp_path, "url", _ZIP_ROOT)
        mock_zip.assert_not_called()
    assert cached == index


def test_get_zip_index_raises_when_no_member_matches_prefix(tmp_path):
    infos = [_FakeZipInfo("some_other_root/file.txt", 10)]
    with patch(
        "confusius.datasets._dataverse.RemoteZip", _fake_remotezip_from_infos(infos)
    ):
        with pytest.raises(RuntimeError, match="No archive members"):
            get_zip_index(tmp_path, "url", _ZIP_ROOT)
    # An empty index must not be cached, or every later call becomes a no-op.
    assert not (tmp_path / "dataverse_zip_index.json").exists()


def test_download_rejects_zip_slip(tmp_path):
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    members = {"../escape.txt": {"size": 10}}

    class _Fake:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            return io.BytesIO(b"x")

    with patch("confusius.datasets._dataverse.RemoteZip", _Fake):
        with pytest.raises(ValueError, match="outside the cache"):
            download_zip_members(bids_dir, "url", members, _ZIP_ROOT)
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------------------
# refresh — CRC-based re-download and index merge
# ---------------------------------------------------------------------------


def _recording_remotezip(opened):
    """Return a fake RemoteZip class that records opened members."""

    class _Fake:
        def __init__(self, url, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            opened.append(member)
            return io.BytesIO(b"new-bytes")

    return _Fake


def test_download_zip_members_refresh_redownloads_changed_crc(tmp_path):
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    (bids_dir / "unchanged.nii").write_bytes(b"old")
    (bids_dir / "changed.nii").write_bytes(b"old")
    # "missing.nii" is absent from disk.

    members = {
        "unchanged.nii": {"size": 3, "crc": 111},
        "changed.nii": {"size": 3, "crc": 222},  # remote crc bumped
        "missing.nii": {"size": 3, "crc": 333},
    }
    previous_index = {
        "unchanged.nii": {"size": 3, "crc": 111},
        "changed.nii": {"size": 3, "crc": 111},  # cached crc is stale
        "missing.nii": {"size": 3, "crc": 333},
    }

    opened = []
    with patch("confusius.datasets._dataverse.RemoteZip", _recording_remotezip(opened)):
        download_zip_members(
            bids_dir, "url", members, _ZIP_ROOT, previous_index, refresh=True
        )

    assert _selected(opened) == {"changed.nii", "missing.nii"}


def test_download_zip_members_no_refresh_ignores_crc(tmp_path):
    bids_dir = tmp_path / _BIDS_ROOT
    bids_dir.mkdir(parents=True)
    (bids_dir / "present.nii").write_bytes(b"old")

    members = {
        "present.nii": {"size": 3, "crc": 999},  # crc differs, but no refresh
        "missing.nii": {"size": 3, "crc": 111},
    }
    previous_index = {"present.nii": {"size": 3, "crc": 111}}

    opened = []
    with patch("confusius.datasets._dataverse.RemoteZip", _recording_remotezip(opened)):
        download_zip_members(
            bids_dir, "url", members, _ZIP_ROOT, previous_index, refresh=False
        )

    assert _selected(opened) == {"missing.nii"}


def test_update_cached_zip_index_preserves_unrequested_baseline(tmp_path):
    remote = {"a.nii": {"size": 1, "crc": 2}, "b.nii": {"size": 1, "crc": 9}}
    previous = {"a.nii": {"size": 1, "crc": 1}, "b.nii": {"size": 1, "crc": 1}}
    members = {"a.nii": {"size": 1, "crc": 2}}  # only "a" was reconciled

    update_cached_zip_index(tmp_path, remote, previous, members)

    written = json.loads((tmp_path / "dataverse_zip_index.json").read_text())
    # Reconciled member advances to the remote crc; the unrequested member keeps
    # its cached baseline so a later refresh still detects it as changed.
    assert written["a.nii"]["crc"] == 2
    assert written["b.nii"]["crc"] == 1


def test_get_zip_index_refresh_reads_fresh_without_overwriting_cache(tmp_path):
    (tmp_path / "dataverse_zip_index.json").write_text(
        json.dumps({"a.nii": {"size": 1, "crc": 111}})
    )
    infos = [_FakeZipInfo(_ZIP_ROOT + "a.nii", 100, crc=999)]

    with patch(
        "confusius.datasets._dataverse.RemoteZip", _fake_remotezip_from_infos(infos)
    ):
        index = get_zip_index(tmp_path, "url", _ZIP_ROOT, refresh=True)

    # The fresh central directory is returned to the caller...
    assert index == {"a.nii": {"size": 100, "crc": 999}}
    # ...but the on-disk cache is left for the caller to merge, so the stale crc
    # is still available as the refresh baseline.
    cached = json.loads((tmp_path / "dataverse_zip_index.json").read_text())
    assert cached["a.nii"]["crc"] == 111


# ---------------------------------------------------------------------------
# fetch_khallaf_2026 — refresh behaviour
# ---------------------------------------------------------------------------


def test_fetch_refresh_passes_flag_to_get_zip_index(
    tmp_path, mock_get_zip_index, opened_members
):
    fetch_khallaf_2026(data_dir=tmp_path, refresh=True)
    mock_get_zip_index.assert_called_once_with(
        tmp_path / _BIDS_ROOT,
        access_url(_DATAFILE_ID),
        _ZIP_ROOT,
        refresh=True,
        session=ANY,
    )


# ---------------------------------------------------------------------------
# download retry behaviour
# ---------------------------------------------------------------------------


def _all_cached_except(tmp_path, target_rel):
    """Create every non-sourcedata member except `target_rel` on disk."""
    bids_dir = tmp_path / _BIDS_ROOT
    for rel in set(_FAKE_INDEX) - _SOURCEDATA_KEYS:
        if rel == target_rel:
            continue
        dest = bids_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.touch()


def test_fetch_retries_on_transient_failure(tmp_path, mock_get_zip_index):
    target_rel = "README.md"
    _all_cached_except(tmp_path, target_rel)
    calls = {"n": 0}

    class _FlakyRemoteZip:
        def __init__(self, url, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            calls["n"] += 1
            if calls["n"] < _MAX_DOWNLOAD_RETRIES:
                raise requests.exceptions.ReadTimeout("simulated timeout")
            return io.BytesIO(b"data")

    with (
        patch("confusius.datasets._dataverse.RemoteZip", _FlakyRemoteZip),
        patch("confusius.datasets._dataverse.time.sleep"),
    ):
        fetch_khallaf_2026(data_dir=tmp_path)

    assert calls["n"] == _MAX_DOWNLOAD_RETRIES
    assert (tmp_path / _BIDS_ROOT / target_rel).exists()


def test_fetch_raises_after_max_retries(tmp_path, mock_get_zip_index):
    target_rel = "README.md"
    _all_cached_except(tmp_path, target_rel)
    calls = {"n": 0}

    class _DeadRemoteZip:
        def __init__(self, url, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            calls["n"] += 1
            raise requests.exceptions.ReadTimeout("persistent timeout")

    with (
        patch("confusius.datasets._dataverse.RemoteZip", _DeadRemoteZip),
        patch("confusius.datasets._dataverse.time.sleep"),
        pytest.raises(requests.exceptions.ReadTimeout),
    ):
        fetch_khallaf_2026(data_dir=tmp_path)

    assert calls["n"] == _MAX_DOWNLOAD_RETRIES


def test_fetch_retries_on_midstream_read_error(tmp_path, mock_get_zip_index):
    """A urllib3 error raised mid-stream by read() is retried, not aborted."""
    target_rel = "README.md"
    _all_cached_except(tmp_path, target_rel)
    calls = {"n": 0}

    class _FlakyStream:
        def __init__(self, fail):
            self._fail = fail
            self._served = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            if self._fail:
                raise urllib3.exceptions.ProtocolError("connection broken")
            if self._served:
                return b""
            self._served = True
            return b"data"

    class _FlakyReadRemoteZip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            calls["n"] += 1
            return _FlakyStream(fail=calls["n"] < _MAX_DOWNLOAD_RETRIES)

    with (
        patch("confusius.datasets._dataverse.RemoteZip", _FlakyReadRemoteZip),
        patch("confusius.datasets._dataverse.time.sleep"),
    ):
        fetch_khallaf_2026(data_dir=tmp_path)

    assert calls["n"] == _MAX_DOWNLOAD_RETRIES
    assert (tmp_path / _BIDS_ROOT / target_rel).exists()


def test_part_file_cleaned_on_nonretriable_error(tmp_path, mock_get_zip_index):
    """A non-retriable error mid-write leaves no orphaned .part file."""
    target_rel = "README.md"
    _all_cached_except(tmp_path, target_rel)

    class _OSErrorStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            raise OSError("disk full")

    class _BadRemoteZip:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def open(self, member):
            return _OSErrorStream()

    with (
        patch("confusius.datasets._dataverse.RemoteZip", _BadRemoteZip),
        patch("confusius.datasets._dataverse.time.sleep"),
        pytest.raises(OSError, match="disk full"),
    ):
        fetch_khallaf_2026(data_dir=tmp_path)

    assert list((tmp_path / _BIDS_ROOT).rglob("*.part")) == []
