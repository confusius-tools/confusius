"""Exercise release discovery, downloads, and caching through public fetchers."""

from __future__ import annotations

import json

import pytest
import requests

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


def test_modified_cache_is_repaired(tmp_path, publish_release):
    publish_release("datasets", _NAME, "1.0.0", {"README.md": b"verified"})
    root = datasets.fetch_pereira_2025(
        data_dir=tmp_path / "cache", print_citation=False
    )
    (root / "README.md").write_bytes(b"corrupted")
    datasets.fetch_pereira_2025(data_dir=tmp_path / "cache", print_citation=False)
    assert (root / "README.md").read_bytes() == b"verified"


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
