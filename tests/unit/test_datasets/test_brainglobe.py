"""Tests for fetch_brainglobe_atlas (network-free via a fake BrainGlobeAtlas)."""

from __future__ import annotations

import errno
import warnings
from pathlib import Path
from types import SimpleNamespace

import brainglobe_atlasapi
import brainglobe_atlasapi.utils
import dask.array as da
import numpy as np
import pytest
import s3fs
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

import confusius.datasets._brainglobe as brainglobe_module
from confusius.datasets import fetch_brainglobe_atlas
from confusius.validation import validate_atlas


class _FakeFs:
    """Minimal filesystem stand-in that records downloads."""

    def __init__(self) -> None:
        self.downloads: list[tuple[object, object, bool, object | None]] = []

    def get(
        self,
        remote_path: object,
        resolution_path: object,
        recursive: bool = True,
        callback: object | None = None,
    ) -> None:
        self.downloads.append((remote_path, resolution_path, recursive, callback))


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
        # Not read directly by fetch_brainglobe_atlas (that goes through the
        # lazily-loaded arrays below), kept only in case a test wants the raw data.
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
            "symmetric": True,
            "annotation_set": {
                "location": "/annotation-sets/fake/1_0",
                "template": {"location": "/templates/fake/1_0"},
            },
        }
        self._template_pyramid_level = 0
        self._annotation_pyramid_level = 0
        self.root_dir = Path(".")
        self.fs = _FakeFs()


@pytest.fixture
def fake_atlases(monkeypatch: pytest.MonkeyPatch) -> list[_FakeBgAtlas]:
    """Patch BrainGlobeAtlas to a fake and collect every instance it creates."""
    created: list[_FakeBgAtlas] = []

    def factory(atlas_name, brainglobe_dir=None, check_latest=True):
        atlas = _FakeBgAtlas(atlas_name, brainglobe_dir, check_latest)
        created.append(atlas)
        return atlas

    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", factory)

    def fake_load_lazy_ngff_array(atlas, location, name, pyramid_level):
        source = (
            atlas.template
            if "template" in name.lower()
            else atlas.annotation
            if "annotation" in name.lower()
            else atlas.hemispheres
        )
        return da.from_array(source)

    monkeypatch.setattr(
        brainglobe_module, "_load_lazy_ngff_array", fake_load_lazy_ngff_array
    )
    return created


def test_returns_valid_atlas_dataset(fake_atlases: list[_FakeBgAtlas]) -> None:
    result = fetch_brainglobe_atlas("allen_mouse_25um")
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"template", "annotation", "hemispheres"}
    assert result.attrs["name"] == "allen_mouse_25um"
    # The builder output must satisfy the atlas validator.
    validate_atlas(result)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("template", np.float32),
        ("annotation", np.int32),
        ("hemispheres", np.int8),
    ],
)
def test_converts_brainglobe_arrays_to_voxeldata(
    fake_atlases: list[_FakeBgAtlas], source: str, expected: type[np.generic]
) -> None:
    result = fetch_brainglobe_atlas("allen_mouse_25um")
    assert result[source].dtype == expected
    assert result[source].dims == ("k", "j", "i")
    np.testing.assert_allclose(
        result[source].x.isel(k=0, j=0).to_numpy(), np.arange(8) * 0.025
    )


def test_fetch_loads_ngff_arrays_lazily_and_downloads_missing_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    atlas = _FakeBgAtlas("allen_mouse_25um")
    atlas.metadata["symmetric"] = False
    atlas.root_dir = tmp_path
    monkeypatch.setattr(
        brainglobe_atlasapi, "BrainGlobeAtlas", lambda *args, **kwargs: atlas
    )

    def fake_from_ngff_zarr(path):
        path = str(path)
        if "template" in path:
            data = atlas.template
        elif "hemispheres" in path:
            data = atlas.hemispheres
        else:
            data = atlas.annotation
        return SimpleNamespace(
            metadata=SimpleNamespace(datasets=[SimpleNamespace(path="0")]),
            images=[SimpleNamespace(data=da.from_array(data))],
        )

    import ngff_zarr

    monkeypatch.setattr(ngff_zarr, "from_ngff_zarr", fake_from_ngff_zarr)

    result = fetch_brainglobe_atlas("allen_mouse_25um")

    assert isinstance(result.template.data, da.Array)
    assert isinstance(result.annotation.data, da.Array)
    assert isinstance(result.hemispheres.data, da.Array)
    assert len(atlas.fs.downloads) == 3


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
def mesh_fake_atlases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_atlases: list[_FakeBgAtlas]
) -> Path:
    """Patch BrainGlobeAtlas to a fake whose meshes are not cached yet, S3 reachable."""

    def factory(atlas_name, brainglobe_dir=None, check_latest=True):
        return _FakeBgAtlas(atlas_name, brainglobe_dir, check_latest, mesh_dir=tmp_path)

    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", factory)
    monkeypatch.setattr(brainglobe_atlasapi.utils, "check_s3_status", lambda **kw: True)
    return tmp_path


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch s3fs with a fake that lists `remote_ids` and records every `get` call."""
    state = SimpleNamespace(remote_ids={"997", "10"}, calls=[], listings=[])

    class FakeS3FileSystem:
        def __init__(self, *args, **kwargs):
            pass

        def ls(self, path):
            state.listings.append(path)
            return [f"{path}/{rid}" for rid in state.remote_ids]

        def get(self, rpaths, lpaths, callback=None):
            state.calls.append((list(rpaths), list(lpaths)))
            for lpath in lpaths:
                Path(lpath).parent.mkdir(parents=True, exist_ok=True)
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
        "s3://brainglobe/atlas/annotation-sets/fake/1_0/annotations.precomputed/mesh"
    )
    assert sorted(rpaths) == [f"{remote_dir}/10", f"{remote_dir}/997"]
    # Remote keys and local files are paired by structure id, in the same order.
    assert [Path(r).name for r in rpaths] == [Path(lp).name for lp in lpaths]
    assert [Path(lp).parent for lp in lpaths] == [mesh_fake_atlases] * 2
    assert (mesh_fake_atlases / "997").exists() and (mesh_fake_atlases / "10").exists()


def test_cached_meshes_skip_the_network(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (mesh_fake_atlases / "997").write_bytes(b"draco")
    (mesh_fake_atlases / "10").write_bytes(b"draco")

    def unexpected_status_check(**kwargs):
        pytest.fail("Cached meshes must not trigger a network check.")

    monkeypatch.setattr(
        brainglobe_atlasapi.utils, "check_s3_status", unexpected_status_check
    )
    fetch_brainglobe_atlas("allen_mouse_25um")

    assert fake_s3.calls == []
    assert fake_s3.listings == []


def test_regions_without_remote_mesh_are_skipped_without_warning(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    fake_s3.remote_ids.discard("10")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fetch_brainglobe_atlas("allen_mouse_25um")

    [(rpaths, _)] = fake_s3.calls
    assert [Path(r).name for r in rpaths] == ["997"]
    assert not (mesh_fake_atlases / "10").exists()


def test_no_remote_meshes_at_all_skips_download(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    fake_s3.remote_ids.clear()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fetch_brainglobe_atlas("allen_mouse_25um")

    assert fake_s3.calls == []


@pytest.mark.parametrize("remote_ids", [{"997"}, set()])
def test_known_unavailable_meshes_skip_network_on_later_fetches(
    mesh_fake_atlases: Path,
    fake_s3: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    remote_ids: set[str],
) -> None:
    fake_s3.remote_ids = remote_ids
    fetch_brainglobe_atlas("allen_mouse_25um")

    fake_s3.calls.clear()
    fake_s3.listings.clear()

    def unexpected_status_check(**kwargs):
        pytest.fail("Known-unavailable meshes must not trigger a network check.")

    monkeypatch.setattr(
        brainglobe_atlasapi.utils, "check_s3_status", unexpected_status_check
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Each fetch constructs a fresh BrainGlobeAtlas; only disk state is shared.
        fetch_brainglobe_atlas("allen_mouse_25um")

    assert fake_s3.calls == []
    assert fake_s3.listings == []


def test_deleted_available_mesh_is_downloaded_again(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    fake_s3.remote_ids.discard("10")
    fetch_brainglobe_atlas("allen_mouse_25um")
    (mesh_fake_atlases / "997").unlink()
    fake_s3.calls.clear()

    fetch_brainglobe_atlas("allen_mouse_25um")

    [(rpaths, lpaths)] = fake_s3.calls
    assert [Path(r).name for r in rpaths] == ["997"]
    assert Path(lpaths[0]).read_bytes() == b"draco"
    assert not (mesh_fake_atlases / "10").exists()


def test_deleted_mesh_directory_is_downloaded_again(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace
) -> None:
    fetch_brainglobe_atlas("allen_mouse_25um")
    for path in mesh_fake_atlases.iterdir():
        path.unlink()
    mesh_fake_atlases.rmdir()
    fake_s3.calls.clear()

    fetch_brainglobe_atlas("allen_mouse_25um")

    [(rpaths, _)] = fake_s3.calls
    assert {Path(r).name for r in rpaths} == {"10", "997"}
    assert (mesh_fake_atlases / "997").read_bytes() == b"draco"
    assert (mesh_fake_atlases / "10").read_bytes() == b"draco"
    assert (
        mesh_fake_atlases / ".confusius-unavailable-meshes.json"
    ).read_text() == "[]"


def test_new_annotation_version_rechecks_mesh_availability(
    mesh_fake_atlases: Path,
    fake_s3: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_s3.remote_ids.discard("10")
    fetch_brainglobe_atlas("allen_mouse_25um")
    fake_s3.remote_ids.add("10")
    fake_s3.calls.clear()
    fake_s3.listings.clear()
    new_mesh_dir = mesh_fake_atlases / "2_0" / "annotations.precomputed" / "mesh"
    new_mesh_dir.mkdir(parents=True)

    def factory(atlas_name, **kwargs):
        atlas = _FakeBgAtlas(atlas_name, mesh_dir=new_mesh_dir)
        annotation_set = atlas.metadata["annotation_set"]
        assert isinstance(annotation_set, dict)
        annotation_set["location"] = "/annotation-sets/fake/2_0"
        return atlas

    monkeypatch.setattr(brainglobe_atlasapi, "BrainGlobeAtlas", factory)
    fetch_brainglobe_atlas("allen_mouse_25um")

    assert fake_s3.listings == [
        "s3://brainglobe/atlas/annotation-sets/fake/2_0/annotations.precomputed/mesh"
    ]
    [(rpaths, _)] = fake_s3.calls
    assert {Path(r).name for r in rpaths} == {"10", "997"}
    assert (new_mesh_dir / "10").read_bytes() == b"draco"


@pytest.mark.parametrize("cache_content", ['["10"', "null", '{"10": true}', "[10]"])
def test_invalid_mesh_availability_record_is_rebuilt(
    mesh_fake_atlases: Path, fake_s3: SimpleNamespace, cache_content: str
) -> None:
    record = mesh_fake_atlases / ".confusius-unavailable-meshes.json"
    record.write_text(cache_content)
    fake_s3.remote_ids.discard("10")

    fetch_brainglobe_atlas("allen_mouse_25um")

    assert record.read_text() == '["10"]'
    assert (mesh_fake_atlases / "997").read_bytes() == b"draco"
    assert not (mesh_fake_atlases / "10").exists()


@pytest.mark.parametrize("error_code", [errno.EACCES, errno.EROFS])
def test_read_only_mesh_cache_still_loads_atlas(
    mesh_fake_atlases: Path,
    fake_s3: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    error_code: int,
) -> None:
    (mesh_fake_atlases / "997").write_bytes(b"draco")
    fake_s3.remote_ids.discard("10")
    write_text = Path.write_text

    def fail_record_write(path, *args, **kwargs):
        if path.name == ".confusius-unavailable-meshes.json":
            raise OSError(error_code, "Cache is read-only")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_record_write)
    with pytest.warns(UserWarning, match="Could not cache unavailable mesh IDs"):
        result = fetch_brainglobe_atlas("allen_mouse_25um")

    validate_atlas(result)
    np.testing.assert_array_equal(result.template.compute(), np.ones((4, 6, 8)))
    assert fake_s3.calls == []
    assert (mesh_fake_atlases / "997").read_bytes() == b"draco"


def test_unreadable_mesh_availability_record_is_ignored(
    mesh_fake_atlases: Path,
    fake_s3: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (mesh_fake_atlases / "997").write_bytes(b"draco")
    fake_s3.remote_ids.discard("10")
    read_text = Path.read_text

    def fail_record_read(path, *args, **kwargs):
        if path.name == ".confusius-unavailable-meshes.json":
            raise PermissionError("Cache record belongs to another user")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_record_read)
    result = fetch_brainglobe_atlas("allen_mouse_25um")

    validate_atlas(result)
    np.testing.assert_array_equal(result.template.compute(), np.ones((4, 6, 8)))
    assert fake_s3.calls == []


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
    assert fake_s3.listings == []
    assert not (mesh_fake_atlases / ".confusius-unavailable-meshes.json").exists()
    validate_atlas(result)

    monkeypatch.setattr(
        brainglobe_atlasapi.utils, "check_s3_status", lambda **kwargs: True
    )
    fetch_brainglobe_atlas("allen_mouse_25um")
    assert (mesh_fake_atlases / "997").read_bytes() == b"draco"
    assert (mesh_fake_atlases / "10").read_bytes() == b"draco"
