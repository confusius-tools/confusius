"""Tests for fetch_brainglobe_atlas (network-free via a fake BrainGlobeAtlas)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import brainglobe_atlasapi
import brainglobe_atlasapi.utils
import numpy as np
import pytest
import s3fs
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

from confusius.datasets import fetch_brainglobe_atlas
from confusius.validation import validate_atlas


class _FakeBgAtlas:
    """Minimal BrainGlobeAtlas stand-in that records its construction arguments.

    With `mesh_dir`, the two structures point their `mesh_filename` at not-yet-existing
    files under it, mimicking a fresh BrainGlobe cache before any mesh download.
    """

    def __init__(
        self, atlas_name, brainglobe_dir=None, check_latest=True, mesh_dir=None
    ):
        self.atlas_name = atlas_name
        self.construction = {
            "brainglobe_dir": brainglobe_dir,
            "check_latest": check_latest,
        }
        shape = (4, 6, 8)
        self.template = np.ones(shape, dtype=np.uint16)
        self.annotation = np.zeros(shape, dtype=np.uint32)
        self.hemispheres = np.ones(shape, dtype=np.uint8)
        self.structures = StructuresDict(
            [
                {
                    "id": 997,
                    "acronym": "root",
                    "name": "whole brain",
                    "rgb_triplet": [200, 200, 200],
                    "structure_id_path": [997],
                    "mesh_filename": None if mesh_dir is None else mesh_dir / "997",
                },
                {
                    "id": 10,
                    "acronym": "ch",
                    "name": "child region",
                    "rgb_triplet": [255, 0, 0],
                    "structure_id_path": [997, 10],
                    "mesh_filename": None if mesh_dir is None else mesh_dir / "10",
                },
            ]
        )
        self.metadata = {
            "name": atlas_name,
            "citation": "Fake et al. (2026)",
            "species": "Mus musculus",
            "orientation": "asr",
            "shape": list(shape),
            "resolution": [25, 25, 25],
            "annotation_set": {"location": "/annotation-sets/fake-annotation/1_0"},
        }


@pytest.fixture
def fake_atlases(monkeypatch: pytest.MonkeyPatch) -> list[_FakeBgAtlas]:
    """Patch BrainGlobeAtlas to a fake and collect every instance it creates."""
    created: list[_FakeBgAtlas] = []

    def factory(atlas_name, brainglobe_dir=None, check_latest=True):
        atlas = _FakeBgAtlas(atlas_name, brainglobe_dir, check_latest)
        created.append(atlas)
        return atlas

    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", factory)
    return created


def test_returns_valid_atlas_dataset(fake_atlases: list[_FakeBgAtlas]) -> None:
    result = fetch_brainglobe_atlas("allen_mouse_25um")
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"reference", "annotation", "hemispheres"}
    assert result.attrs["name"] == "allen_mouse_25um"
    # The builder output must satisfy the atlas validator.
    validate_atlas(result)


def test_defaults_check_latest_off_and_brainglobe_default_cache(
    fake_atlases: list[_FakeBgAtlas],
) -> None:
    fetch_brainglobe_atlas("allen_mouse_25um")
    assert fake_atlases[0].construction == {
        "brainglobe_dir": None,
        "check_latest": False,
    }


def test_forwards_data_dir_and_check_latest(
    fake_atlases: list[_FakeBgAtlas], tmp_path
) -> None:
    fetch_brainglobe_atlas("allen_mouse_25um", data_dir=tmp_path, check_latest=True)
    assert fake_atlases[0].construction == {
        "brainglobe_dir": tmp_path,
        "check_latest": True,
    }


# ── Mesh prefetch ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def mesh_fake_atlases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Patch BrainGlobeAtlas to a fake whose meshes are not cached yet, S3 reachable."""

    def factory(atlas_name, brainglobe_dir=None, check_latest=True):
        return _FakeBgAtlas(atlas_name, brainglobe_dir, check_latest, mesh_dir=tmp_path)

    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", factory)
    monkeypatch.setattr(brainglobe_atlasapi.utils, "check_s3_status", lambda **kw: True)
    return tmp_path


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch s3fs with a fake that lists `remote_ids` and records every `get` call."""
    state = SimpleNamespace(remote_ids={"997", "10"}, calls=[])

    class FakeS3FileSystem:
        def __init__(self, *args, **kwargs):
            pass

        def ls(self, path):
            return [f"{path}/{rid}" for rid in state.remote_ids]

        def get(self, rpaths, lpaths, callback=None):
            state.calls.append((list(rpaths), list(lpaths)))
            for lpath in lpaths:
                Path(lpath).write_bytes(b"draco")

    monkeypatch.setattr(s3fs, "S3FileSystem", FakeS3FileSystem)
    state.cls = FakeS3FileSystem
    return state


def test_prefetches_missing_meshes_in_one_batched_call(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    fetch_brainglobe_atlas("allen_mouse_25um")

    [(rpaths, lpaths)] = fake_s3.calls
    remote_dir = (
        "s3://brainglobe/atlas/annotation-sets/fake-annotation/1_0/"
        "annotations.precomputed/mesh"
    )
    assert sorted(rpaths) == [f"{remote_dir}/10", f"{remote_dir}/997"]
    # Remote keys and local files are paired by structure id, in the same order.
    assert [Path(r).name for r in rpaths] == [Path(lp).name for lp in lpaths]
    assert [Path(lp).parent for lp in lpaths] == [mesh_fake_atlases] * 2
    assert (mesh_fake_atlases / "997").exists() and (mesh_fake_atlases / "10").exists()


def test_cached_meshes_skip_the_network(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    (mesh_fake_atlases / "997").write_bytes(b"draco")
    (mesh_fake_atlases / "10").write_bytes(b"draco")

    fetch_brainglobe_atlas("allen_mouse_25um")

    assert fake_s3.calls == []


def test_regions_without_remote_mesh_are_skipped_with_warning(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    fake_s3.remote_ids.discard("10")

    with pytest.warns(UserWarning, match=r"no mesh for region id\(s\) \['10'\]"):
        fetch_brainglobe_atlas("allen_mouse_25um")

    [(rpaths, _)] = fake_s3.calls
    assert [Path(r).name for r in rpaths] == ["997"]
    assert not (mesh_fake_atlases / "10").exists()


def test_failed_download_removes_partial_files(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_get(self, rpaths, lpaths, callback=None):
        Path(lpaths[0]).write_bytes(b"partial")
        raise ConnectionError("network dropped mid-download")

    monkeypatch.setattr(fake_s3.cls, "get", failing_get)

    with pytest.raises(ConnectionError):
        fetch_brainglobe_atlas("allen_mouse_25um")

    assert not (mesh_fake_atlases / "997").exists()
    assert not (mesh_fake_atlases / "10").exists()


def test_unreachable_bucket_skips_prefetch_with_warning(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        brainglobe_atlasapi.utils, "check_s3_status", lambda **kw: False
    )

    with pytest.warns(UserWarning, match="unreachable"):
        result = fetch_brainglobe_atlas("allen_mouse_25um")

    assert fake_s3.calls == []
    validate_atlas(result)
