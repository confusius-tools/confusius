"""Exercise release discovery, downloads, and caching through public fetchers."""

from __future__ import annotations

import json
import threading

import pytest
import requests
from botocore.exceptions import ClientError

from confusius import datasets

_DATASETS = (
    ("fetch_nunez_elizalde_2022", "nunez-elizalde-2022-bids"),
    ("fetch_pereira_2025", "pereira-2025-bids"),
    ("fetch_cybis_pereira_2026", "cybis-pereira-2026-bids"),
    ("fetch_pepe_mariani_2026", "pepe-mariani-2026-bids"),
    ("fetch_landemard_2026", "landemard-2026-bids"),
    ("fetch_khallaf_2026", "khallaf-2026-bids"),
)
_NAME = "pereira-2025-bids"


@pytest.mark.parametrize("fetcher,name", _DATASETS)
def test_download_and_offline_cache(
    tmp_path, release_server, publish_release, fetcher, name
):
    content = b'{"Name": "test release"}'
    publish_release("datasets", name, "1.0.0", {"dataset_description.json": content})
    fetch = getattr(datasets, fetcher)
    root = fetch(data_dir=tmp_path / "cache", print_citation=False)
    assert root == tmp_path / "cache" / name / "1.0.0"
    assert (root / "dataset_description.json").read_bytes() == content
    _, calls = release_server
    assert calls == [
        "/last_versions.conf",
        f"/datasets/{name}/1.0.0/manifest.json",
        f"/datasets/{name}/1.0.0/dataset_description.json",
    ]
    calls.clear()
    assert fetch(data_dir=tmp_path / "cache", print_citation=False) == root
    assert calls == []


def test_refresh_isolates_versions_and_preserves_previous_files(
    tmp_path, release_server, publish_release
):
    cache = tmp_path / "cache"
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"first"})
    old = datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    publish_release("datasets", _NAME, "2.0.0", {"README.md": b"second"})
    assert datasets.fetch_pereira_2025(data_dir=cache, print_citation=False) == old
    new = datasets.fetch_pereira_2025(
        data_dir=cache, refresh=True, print_citation=False
    )
    assert new == cache / _NAME / "2.0.0"
    assert (new / "README.md").read_bytes() == b"second"
    assert (old / "README.md").read_bytes() == b"first"
    _, calls = release_server
    calls.clear()
    assert datasets.fetch_pereira_2025(data_dir=cache, print_citation=False) == new
    assert calls == []


def test_refresh_same_release_does_not_redownload(
    tmp_path, release_server, publish_release
):
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"verified"})
    datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)
    _, calls = release_server
    calls.clear()
    datasets.fetch_pereira_2025(
        data_dir=tmp_path / "cache", refresh=True, print_citation=False
    )
    assert calls == ["/last_versions.conf", f"/datasets/{_NAME}/1.0.0/manifest.json"]


def test_cached_files_are_trusted_until_deleted(
    tmp_path, release_server, publish_release, monkeypatch
):
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"verified"})
    root = datasets.fetch_pereira_2025(
        data_dir=tmp_path / "cache", print_citation=False
    )
    (root / "README.md").write_bytes(b"modified")
    _, calls = release_server
    calls.clear()
    with monkeypatch.context() as context:
        context.setattr(
            "pooch.file_hash",
            lambda *args, **kwargs: pytest.fail("Cached file was rehashed"),
        )
        for refresh in (False, True):
            assert datasets.fetch_pereira_2025(
                data_dir=tmp_path / "cache",
                refresh=refresh,
                print_citation=False,
            ) == root
        assert (root / "README.md").read_bytes() == b"modified"
    assert calls == ["/last_versions.conf", f"/datasets/{_NAME}/1.0.0/manifest.json"]
    (root / "README.md").unlink()
    datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)
    assert (root / "README.md").read_bytes() == b"verified"


def test_refresh_redownloads_changed_upstream_hash(tmp_path, publish_release):
    cache = tmp_path / "cache"
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"first"})
    root = datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"second"})
    datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    assert (root / "README.md").read_bytes() == b"first"
    datasets.fetch_pereira_2025(data_dir=cache, refresh=True, print_citation=False)
    assert (root / "README.md").read_bytes() == b"second"


def test_wrong_download_hash_fails_without_recording_release(tmp_path, publish_release):
    remote = publish_release("datasets", _NAME, "1.0.0", {"README.md": b"verified"})
    (remote / "README.md").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256 hash"):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)
    assert not (tmp_path / "cache" / _NAME / "s3_index.json").exists()
    assert not (tmp_path / "cache" / _NAME / "1.0.0" / "README.md").exists()


def test_unpublished_recipe_raises(tmp_path, release_server):
    root, _ = release_server
    (root / "last_versions.conf").write_text("[datasets]\n")
    with pytest.raises(RuntimeError, match="No published S3 release"):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "a/../../outside",
        "a\\file",
        "a//file",
        "manifest.json",
        "C:/outside",
    ],
)
def test_unsafe_manifest_paths_rejected(tmp_path, publish_release, path):
    remote = publish_release("datasets", _NAME, "1.0.0", {"README.md": b"safe"})
    (remote / "manifest.json").write_text(
        json.dumps({path: {"size": 4, "sha256": "a" * 64}})
    )
    with pytest.raises(ValueError, match="Invalid release manifest"):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"size": -1, "sha256": "a" * 64},
        {"size": True, "sha256": "a" * 64},
        {"size": 1, "sha256": "bad"},
    ],
)
def test_invalid_manifest_metadata_rejected(tmp_path, publish_release, info):
    remote = publish_release("datasets", _NAME, "1.0.0", {"README.md": b"safe"})
    (remote / "manifest.json").write_text(json.dumps({"README.md": info}))
    with pytest.raises(ValueError, match="Invalid release manifest"):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)


@pytest.mark.parametrize("manifest", [{}, [], {"README.md": None}])
def test_invalid_manifest_shape_rejected(tmp_path, publish_release, manifest):
    remote = publish_release("datasets", _NAME, "1.0.0", {"README.md": b"safe"})
    (remote / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)


@pytest.mark.parametrize("version", ["../outside", "/outside", "1.0.0/other"])
def test_invalid_catalog_version_rejected(tmp_path, release_server, version):
    root, _ = release_server
    (root / "last_versions.conf").write_text(f"[datasets]\n{_NAME} = {version}\n")
    with pytest.raises(ValueError, match="Invalid release version"):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)


def test_unsafe_cached_inventory_rejected(tmp_path, publish_release):
    cache = tmp_path / "cache"
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"safe"})
    datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    index_path = cache / _NAME / "s3_index.json"
    index = json.loads(index_path.read_text())
    index["../outside"] = index.pop("README.md")
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Invalid release manifest"):
        datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    # An explicit refresh replaces damaged local metadata with the valid manifest.
    root = datasets.fetch_pereira_2025(
        data_dir=cache, refresh=True, print_citation=False
    )
    assert (root / "README.md").read_bytes() == b"safe"


@pytest.mark.parametrize(
    "url,error",
    [
        (None, "Cached inventory must define a URL"),
        ("https://example.com/README.md", "invalid release URL"),
        ("{base}/2.0.0/README.md", "mixes release versions"),
    ],
)
def test_invalid_cached_release_urls_rejected(
    tmp_path, release_server, publish_release, url, error
):
    cache = tmp_path / "cache"
    publish_release(
        "datasets",
        _NAME,
        "1.0.0",
        {"README.md": b"safe", "dataset_description.json": b"{}"},
    )
    datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    index_path = cache / _NAME / "s3_index.json"
    index = json.loads(index_path.read_text())
    base = index["README.md"]["url"].rsplit("/", 2)[0]
    index["README.md"]["url"] = url.format(base=base) if url is not None else None
    index_path.write_text(json.dumps(index))
    _, calls = release_server
    calls.clear()
    with pytest.raises(ValueError, match=error):
        datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    assert calls == []
    assert json.loads(index_path.read_text()) == index


def test_missing_manifest_is_not_an_empty_release(tmp_path, publish_release):
    remote = publish_release("datasets", _NAME, "1.0.0", {"README.md": b"safe"})
    (remote / "manifest.json").unlink()
    with pytest.raises(requests.HTTPError):
        datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)


def test_failed_refresh_keeps_selected_cached_release(tmp_path, publish_release):
    cache = tmp_path / "cache"
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"first"})
    old = datasets.fetch_pereira_2025(data_dir=cache, print_citation=False)
    remote = publish_release("datasets", _NAME, "2.0.0", {"README.md": b"second"})
    (remote / "README.md").write_bytes(b"wrong")
    with pytest.raises(ValueError):
        datasets.fetch_pereira_2025(data_dir=cache, refresh=True, print_citation=False)
    assert datasets.fetch_pereira_2025(data_dir=cache, print_citation=False) == old
    assert (old / "README.md").read_bytes() == b"first"


def test_progress_callback_reports_bytes(tmp_path, publish_release):
    content = b"verified"
    publish_release(
        "datasets", "nunez-elizalde-2022-bids", "1.0.0", {"README.md": content}
    )
    reports = []
    root = datasets.fetch_nunez_elizalde_2022(
        data_dir=tmp_path / "cache",
        progress_callback=lambda *args: reports.append(args),
        print_citation=False,
    )
    assert (root / "README.md").read_bytes() == content
    assert reports[0][:2] == (0, len(content))
    assert reports[-1] == (len(content), len(content), "Download complete.")


def test_transfers_reuse_unsigned_http_connection(
    tmp_path, publish_release, s3_requests
):
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"verified"})
    datasets.fetch_pereira_2025(data_dir=tmp_path, print_citation=False)
    assert [record[0] for record in s3_requests] == ["HEAD", "GET"]
    assert all(record[3] is None for record in s3_requests)
    assert len({record[4] for record in s3_requests}) == 1


def test_batch_downloads_run_concurrently(tmp_path, publish_release, s3_barrier):
    files = {f"file-{i}.bin": bytes([i]) * 1024 for i in range(12)}
    publish_release("datasets", _NAME, "1.0.0", files)
    # Serial GETs cannot pass the barrier before the next request starts.
    s3_barrier.append(threading.Barrier(2))
    root = datasets.fetch_pereira_2025(data_dir=tmp_path, print_citation=False)
    for relative, content in files.items():
        assert (root / relative).read_bytes() == content


def test_multipart_download_verifies_assembled_bytes(
    tmp_path, publish_release, s3_requests
):
    content = bytes(range(256)) * (9 * 1024**2 // 256)
    publish_release("datasets", _NAME, "1.0.0", {"large.bin": content})
    root = datasets.fetch_pereira_2025(data_dir=tmp_path, print_citation=False)
    assert (root / "large.bin").read_bytes() == content
    assert {record[2] for record in s3_requests if record[0] == "GET"} == {
        "bytes=0-8388607",
        "bytes=8388608-",
    }


@pytest.mark.parametrize("failures", [[503], ["disconnect"]])
def test_sdk_retries_and_reports_progress_on_calling_thread(
    tmp_path, publish_release, s3_failures, s3_requests, failures
):
    name = "nunez-elizalde-2022-bids"
    content = b"a" * (2 * 1024**2)
    publish_release("datasets", name, "1.0.0", {"README.md": content})
    s3_failures[f"/datasets/{name}/1.0.0/README.md"] = failures.copy()
    reports = []
    root = datasets.fetch_nunez_elizalde_2022(
        data_dir=tmp_path,
        print_citation=False,
        progress_callback=lambda *args: reports.append((threading.get_ident(), *args)),
    )
    assert (root / "README.md").read_bytes() == content
    assert sum(record[0] == "GET" for record in s3_requests) == 2
    assert all(record[0] == threading.get_ident() for record in reports)
    assert all(0 <= record[1] <= len(content) for record in reports)
    assert reports[-1][1:] == (len(content), len(content), "Download complete.")


def test_sdk_retry_exhaustion_does_not_leave_partial_cache(
    tmp_path, publish_release, s3_failures, s3_requests
):
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"verified"})
    s3_failures[f"/datasets/{_NAME}/1.0.0/README.md"] = [503, 503, 503]
    with pytest.raises(ClientError):
        datasets.fetch_pereira_2025(data_dir=tmp_path, print_citation=False)
    assert sum(record[0] == "GET" for record in s3_requests) == 3
    assert not (tmp_path / _NAME / "s3_index.json").exists()
    assert list((tmp_path / _NAME / "1.0.0").iterdir()) == []


def test_downloads_empty_files_and_escaped_keys(tmp_path, publish_release):
    files = {"empty.bin": b"", "notes %25.md": b"escaped key"}
    publish_release("datasets", _NAME, "1.0.0", files)
    root = datasets.fetch_pereira_2025(data_dir=tmp_path, print_citation=False)
    for relative, content in files.items():
        assert (root / relative).read_bytes() == content


def test_callback_failure_stops_without_accepting_inventory(tmp_path, publish_release):
    publish_release(
        "datasets", "nunez-elizalde-2022-bids", "1.0.0", {"README.md": b"verified"}
    )
    reports = []

    def report(*args):
        """Reject the second progress notification during the active transfer."""
        reports.append(args)
        if len(reports) > 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        datasets.fetch_nunez_elizalde_2022(
            data_dir=tmp_path, print_citation=False, progress_callback=report
        )
    assert not (tmp_path / "nunez-elizalde-2022-bids" / "s3_index.json").exists()
