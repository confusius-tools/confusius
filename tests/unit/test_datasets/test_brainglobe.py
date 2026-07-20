"""Tests for fetch_brainglobe_atlas (network-free via a fake BrainGlobeAtlas)."""

from __future__ import annotations

import brainglobe_atlasapi
import numpy as np
import pytest
import xarray as xr
from brainglobe_atlasapi.structure_class import StructuresDict

from confusius.datasets import fetch_brainglobe_atlas
from confusius.validation import validate_atlas_dataset


class _FakeBgAtlas:
    """Minimal BrainGlobeAtlas stand-in that records its construction arguments."""

    def __init__(self, atlas_name, brainglobe_dir=None, check_latest=True):
        self.atlas_name = atlas_name
        self.construction = {
            "brainglobe_dir": brainglobe_dir,
            "check_latest": check_latest,
        }
        shape = (4, 6, 8)
        self.reference = np.ones(shape, dtype=np.uint16)
        self.annotation = np.zeros(shape, dtype=np.int32)
        self.hemispheres = np.ones(shape, dtype=np.int8)
        self.structures = StructuresDict(
            [
                {
                    "id": 997,
                    "acronym": "root",
                    "name": "whole brain",
                    "rgb_triplet": [200, 200, 200],
                    "structure_id_path": [997],
                    "mesh_filename": None,
                }
            ]
        )
        self.metadata = {
            "name": atlas_name,
            "citation": "Fake et al. (2026)",
            "species": "Mus musculus",
            "orientation": "asr",
            "shape": list(shape),
            "resolution": [25, 25, 25],
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
    validate_atlas_dataset(result)


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
